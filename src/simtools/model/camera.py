"""Definition and modeling of camera."""

import logging
from pathlib import Path

import astropy.units as u
import numpy as np

from simtools.utils.geometry import rotate


class Camera:
    """
    Camera class, defining pixel layout.

    This includes rotation, finding neighbor pixels, calculating FoV and plotting the camera.
    Assume that all pixels have the same shape and size.

    Parameters
    ----------
    telescope_name: str
        Telescope name (e.g., LSTN-01)
    camera_config_file: str or Path
        The sim_telarray camera configuration file.
    focal_length: float
        The focal length of the camera in (preferably the effective focal length), assumed to be \
        in the same unit as the pixel positions in the camera_config_file, usually cm.
    camera_config_dict: dict, optional
        Dictionary (as provided by eventio) with camera configuration information.
        Used alternatively to camera_config_file, which is ignored when this is provided.
    """

    # Constants for finding neighbor pixels.
    PMT_NEIGHBOR_RADIUS_FACTOR = 1.1
    SIPM_NEIGHBOR_RADIUS_FACTOR = 1.4
    SIPM_ROW_COLUMN_DIST_FACTOR = 0.2

    def __init__(
        self,
        telescope_name,
        camera_config_file,
        focal_length,
        camera_config_dict=None,
    ):
        """Initialize Camera class, defining pixel layout."""
        self._logger = logging.getLogger(__name__)

        self.telescope_name = telescope_name
        self.focal_length = focal_length
        if self.focal_length <= 0:
            raise ValueError("The focal length must be larger than zero")
        if camera_config_dict is not None:
            self.pixels = self.read_pixel_list_from_dict(camera_config_dict)
        else:
            self.pixels = self.read_pixel_list(camera_config_file)

        self.pixels = self._rotate_pixels(self.pixels)

        # Empty list of neighbors, to be calculated only when necessary.
        self._neighbors = None
        # Empty list of edge pixels, to be calculated only when necessary.
        self._edge_pixel_indices = None

    @staticmethod
    def read_pixel_list(camera_config_file: str | Path) -> dict:
        """
        Read the pixel layout from the camera config file, assumed to be in a sim_telarray format.

        Parameters
        ----------
        camera_config_file: str or Path
            The sim_telarray file name.

        Returns
        -------
        dict: pixels
            A dictionary with the pixel positions, the camera rotation angle, the pixel shape, \
            the pixel diameter, the pixel IDs and their "on" status.

        Notes
        -----
        The pixel shape can be hexagonal (denoted as 1 or 3) or a square (denoted as 2). \
        The hexagonal shapes differ in their orientation, where those denoted as 3 are rotated
        clockwise by 30 degrees with respect to those denoted as 1.
        """
        pixels = Camera.initialize_pixel_dict()

        with open(camera_config_file, encoding="utf-8") as dat_file:
            for line in dat_file:
                Camera.process_line(line, pixels)

        Camera.validate_pixels(pixels, camera_config_file)

        return pixels

    @staticmethod
    def read_pixel_list_from_dict(camera_config_dict: dict) -> dict:
        """
        Read pixel list from pyeventio camera configuration dictionary.

        Parameters
        ----------
        camera_config_dict: dict
            Dictionary (as provided by eventio) with camera configuration information.

        Returns
        -------
        dict: pixels
            A dictionary with the pixel positions, the camera rotation angle, the pixel shape, \
            the pixel diameter, the pixel IDs and their "on" status.
        """
        pixels = Camera.initialize_pixel_dict()

        # assume all pixels have same size and shape
        unique_diam = np.unique(camera_config_dict["pixel_size"])
        if unique_diam.size != 1:
            raise ValueError(f"Pixel diameter is not unique: {unique_diam}")
        pixels["pixel_diameter"] = float(unique_diam[0])

        pixels["pixel_shape"] = camera_config_dict["common_pixel_shape"]
        pixels["x"] = camera_config_dict["pixel_x"]
        pixels["y"] = camera_config_dict["pixel_y"]
        pixels["pix_id"] = list(range(len(pixels["x"])))
        pixels["pix_on"] = np.ones(len(pixels["x"]), dtype=int).tolist()

        Camera.validate_pixels(pixels, "from camera dict")

        return pixels

    @staticmethod
    def initialize_pixel_dict() -> dict:
        """
        Initialize the pixel dictionary with default values.

        Returns
        -------
        dict
            A dictionary with default pixel properties.
        """
        return {
            "pixel_diameter": 9999,
            "pixel_shape": 9999,
            "pixel_spacing": 9999,
            "lightguide_efficiency_angle_file": "none",
            "lightguide_efficiency_wavelength_file": "none",
            "rotate_angle": 0,
            "x": [],
            "y": [],
            "pix_id": [],
            "pix_on": [],
        }

    @staticmethod
    def process_line(line: str, pixels: dict):
        """
        Process a line from the camera config file and update the pixels dictionary.

        Parameters
        ----------
        line: str
            A line from the camera config file.
        pixels: dict
            The dictionary to update with pixel information.
        """
        pix_info = line.split()

        if line.startswith("PixType"):
            pixels["pixel_shape"] = int(pix_info[5].strip())
            pixels["pixel_diameter"] = float(pix_info[6].strip())
            pixels["lightguide_efficiency_angle_file"] = pix_info[8].strip().replace('"', "")

            if len(pix_info) > 9:
                pixels["lightguide_efficiency_wavelength_file"] = (
                    pix_info[9].strip().replace('"', "")
                )

        elif line.startswith("Rotate"):
            pixels["rotate_angle"] = np.deg2rad(float(pix_info[1].strip()))

        elif line.startswith("Pixel"):
            pixels["x"].append(float(pix_info[3].strip()))
            pixels["y"].append(float(pix_info[4].strip()))
            pixels["pix_id"].append(int(pix_info[1].strip()))

            if len(pix_info) > 9:
                pixels["pix_on"].append(int(pix_info[9].strip()) != 0)
            else:
                pixels["pix_on"].append(True)

    @staticmethod
    def validate_pixels(pixels: dict, camera_config_file: str | Path):
        """
        Validate the pixel dictionary to ensure all required fields are present.

        Parameters
        ----------
        pixels: dict
            The pixel dictionary to validate.
        camera_config_file: string
            The sim_telarray file name for error messages.

        Raises
        ------
        ValueError
            If the pixel diameter or pixel shape is invalid.
        """
        if pixels["pixel_diameter"] == 9999:
            raise ValueError(f"Could not read the pixel diameter from {camera_config_file} file")

        if pixels["pixel_shape"] not in [1, 2, 3]:
            raise ValueError(
                f"Pixel shape in {camera_config_file} unrecognized (has to be 1, 2 or 3)"
            )

    def _rotate_pixels(self, pixels: dict) -> dict:
        """
        Rotate the pixels according to the rotation angle given in pixels['rotate_angle'].

        Additional rotation is added to get to the camera view of an observer facing the camera.
        The angle for the axes rotation depends on the coordinate system in which the original
        data was provided.

        Parameters
        ----------
        pixels: dict
            The dictionary produced by the read_pixel_list method of this class

        Returns
        -------
        pixels: dict
            The pixels dictionary with rotated pixels.
            The pixels orientation for plotting is added to the dictionary in pixels['orientation'].
            The orientation is determined by the pixel shape (see read_pixel_list for details).
        """
        rotate_angle = pixels["rotate_angle"] * u.rad  # So not to change the original angle
        # The original pixel list is given such that
        # x -> North, y -> West, z -> Up in the ground system.
        # At azimuth=0, zenith angle=0 all coordinate systems are aligned.
        # When the telescope turns the "normal" way towards
        # the horizon, the x-axis points downwards, the y-axis points right
        # (when looking from the camera onto the dish),
        # and the z-axis points in any case from (primary) dish towards camera.
        # To get the camera for an observer facing the camera, need to rotate by 90 degrees.
        rotate_angle += (90 * u.deg).to(u.rad)

        self._logger.debug(f"Rotating pixels by {rotate_angle.to(u.deg)} (clockwise rotation)")

        if rotate_angle != 0:
            pixels["x"], pixels["y"] = rotate(pixels["x"], pixels["y"], rotate_angle)

        pixels["orientation"] = 0
        if pixels["pixel_shape"] == 1 or pixels["pixel_shape"] == 3:
            if pixels["pixel_shape"] == 3:
                pixels["orientation"] = 30
            if rotate_angle > 0:
                pixels["orientation"] -= rotate_angle.to(u.deg).value

        return pixels

    def get_number_of_pixels(self) -> int:
        """
        Get the number of pixels in the camera (all pixels, including those defined as "off").

        Returns
        -------
        int
            number of pixels.
        """
        return len(self.pixels["x"])

    def get_pixel_diameter(self) -> float:
        """
        Get pixel diameter contained in _pixels.

        Returns
        -------
        float
            Pixel diameter (usually in cm).
        """
        return self.pixels["pixel_diameter"]

    def get_pixel_active_solid_angle(self) -> float:
        """
        Get the active solid angle of a pixel in sr.

        Returns
        -------
        float
            active solid angle of a pixel in sr.
        """
        pixel_area = self.get_pixel_diameter() ** 2
        # In case we have hexagonal pixels:
        if self.get_pixel_shape() == 1 or self.get_pixel_shape() == 3:
            pixel_area *= np.sqrt(3) / 2
        return pixel_area / (self.focal_length**2)

    def get_pixel_shape(self) -> int:
        """
        Get pixel shape code 1, 2 or 3.

        Where 1 and 3 are hexagonal pixels, where one is rotated by\
        30 degrees with respect to the other. A square pixel is denoted as 2.

        Returns
        -------
        int (1, 2 or 3)
            Pixel shape.
        """
        return self.pixels["pixel_shape"]

    def get_lightguide_efficiency_angle_file_name(self) -> str:
        """
        Get the file name of the light guide efficiency as a function of incidence angle.

        Returns
        -------
        str
            File name of the light guide efficiency as a function of incidence angle.
        """
        return self.pixels["lightguide_efficiency_angle_file"]

    def get_lightguide_efficiency_wavelength_file_name(self) -> str:
        """
        Get the file name of the light guide efficiency as a function of wavelength.

        Returns
        -------
        str
            File name of the light guide efficiency as a function of wavelength.
        """
        return self.pixels["lightguide_efficiency_wavelength_file"]

    def get_camera_fill_factor(self) -> float:
        """
        Calculate the fill factor of the camera, defined as (pixel_diameter/pixel_spacing)**2.

        Returns
        -------
        float
            The camera fill factor.
        """
        from scipy.spatial import distance  # pylint: disable=import-outside-toplevel

        if self.pixels["pixel_spacing"] == 9999:
            points = np.array([self.pixels["x"], self.pixels["y"]]).T
            pixel_distances = distance.cdist(points, points, "euclidean")
            # pylint: disable=unsubscriptable-object
            pixel_distances = pixel_distances[pixel_distances > 0]
            pixel_spacing = np.min(pixel_distances)
            self.pixels["pixel_spacing"] = pixel_spacing

        return (self.pixels["pixel_diameter"] / self.pixels["pixel_spacing"]) ** 2

    def calc_fov(self) -> tuple[float, float]:
        """
        Calculate the FOV of the camera in degrees, taking into account the focal length.

        Returns
        -------
        fov: float
            The FOV of the camera in the degrees.
        average_edge_distance: float
            The average edge distance of the camera.

        Notes
        -----
        The x,y pixel positions and focal length are assumed to have the same unit (usually cm)
        """
        self._logger.debug("Calculating the FoV")
        return self._calc_fov(
            self.pixels["x"],
            self.pixels["y"],
            self.get_edge_pixels(),
            self.focal_length,
        )

    def _calc_fov(
        self,
        x_pixel: list[float],
        y_pixel: list[float],
        edge_pixel_indices: list[int],
        focal_length: float,
    ) -> tuple[float, float]:
        """
        Calculate the FOV of the camera in degrees, taking into account the focal length.

        Parameters
        ----------
        x_pixel: list
            List of positions of the pixels on the x-axis
        y_pixel: list
            List of positions of the pixels on the y-axis
        edge_pixel_indices: list
            List of indices of the edge pixels
        focal_length: float
            The focal length of the camera in (preferably the effective focal length), assumed to \
            be in the same unit as the pixel positions.

        Returns
        -------
        fov: float
            The FOV of the camera in the degrees.
        average_edge_distance: float
            The average edge distance of the camera

        Notes
        -----
        The x,y pixel positions and focal length are assumed to have the same unit (usually cm)
        """
        self._logger.debug("Calculating the FoV")

        average_edge_distance = 0
        for i_pix in edge_pixel_indices:
            average_edge_distance += np.sqrt(x_pixel[i_pix] ** 2 + y_pixel[i_pix] ** 2)
        average_edge_distance /= len(edge_pixel_indices)

        fov = 2 * np.rad2deg(np.arctan(average_edge_distance / focal_length))

        return fov, average_edge_distance

    @staticmethod
    def _find_neighbors(x_pos: np.ndarray, y_pos: np.ndarray, radius: float) -> list[list[int]]:
        """
        Use a KD-Tree to quickly find nearest neighbors.

        This applies to e.g., of the pixels in a camera or mirror facets.

        Parameters
        ----------
        x_pos : numpy.array_like
            x position of each e.g., pixel
        y_pos : numpy.array_like
            y position of each e.g., pixel
        radius : float
            radius to consider neighbor it should be slightly larger than the pixel diameter or \
            mirror facet.

        Returns
        -------
        list of lists
            Array of neighbor indices in a list for each pixel
        """
        from scipy.spatial import cKDTree as KDTree  # pylint: disable=import-outside-toplevel

        tree = KDTree(np.column_stack([x_pos, y_pos]))
        neighbors = tree.query_ball_tree(tree, radius)
        return [list(np.setdiff1d(neigh, [i])) for i, neigh in enumerate(neighbors)]

    def _find_adjacent_neighbor_pixels(
        self, x_pos: np.ndarray, y_pos: np.ndarray, radius: float, row_column_dist: float
    ) -> list[list[int]]:
        """
        Find adjacent neighbor pixels in cameras with square pixels.

        Only directly adjacent neighbors are allowed, no diagonals.

        Parameters
        ----------
        x_pos: np.ndarray
            x position of each pixel
        y_pos: np.ndarray
            y position of each pixel
        radius: float
            Radius within which to find neighbors
        row_column_dist: float
            Distance to consider for row/column adjacency.
            Should be around 20% of the pixel diameter.

        Returns
        -------
        list of lists
            Array of neighbor indices in a list for each pixel
        """
        # First find the neighbors with the usual method and the original radius
        # which does not allow for diagonal neighbors.
        neighbors = self._find_neighbors(x_pos, y_pos, radius)

        for i_pix, nn in enumerate(neighbors):
            # Find pixels defined as edge pixels now
            if len(nn) < 4:
                # Go over all other pixels and search for ones which are adjacent
                # but further than sqrt(2) away
                self._add_additional_neighbors(i_pix, nn, x_pos, y_pos, radius, row_column_dist)

        return neighbors

    def _add_additional_neighbors(
        self,
        i_pix: int,
        nn: list[int],
        x_pos: np.ndarray,
        y_pos: np.ndarray,
        radius: float,
        row_column_dist: float,
    ):
        """
        Add neighbors for a given pixel if they are not already neighbors and are adjacent.

        Parameters
        ----------
        i_pix: int
            Index of the pixel to find neighbors for
        nn: list
            Current list of neighbors for the pixel
        x_pos: np.ndarray
            x position of each pixel
        y_pos: np.ndarray
            y position of each pixel
        radius: float
            Radius within which to find neighbors
        row_column_dist: float
            Distance to consider for row/column adjacency
        """
        for j_pix, _ in enumerate(x_pos):
            # No need to look at the pixel itself
            # nor at any pixels already in the neighbors list
            if j_pix != i_pix and j_pix not in nn:
                dist = np.sqrt(
                    (x_pos[i_pix] - x_pos[j_pix]) ** 2 + (y_pos[i_pix] - y_pos[j_pix]) ** 2
                )
                # Check if this pixel is in the same row or column
                # and allow it to be ~1.68*diameter away (1.4*1.2 = 1.68)
                # Need to increase the distance because of the curvature
                # of the CHEC camera
                if (
                    abs(x_pos[i_pix] - x_pos[j_pix]) < row_column_dist
                    or abs(y_pos[i_pix] - y_pos[j_pix]) < row_column_dist
                ) and dist < 1.2 * radius:
                    nn.append(j_pix)

    def _calc_neighbor_pixels(self, pixels: dict) -> list[list[int]]:
        """
        Find adjacent neighbor pixels in cameras with hexagonal or square pixels.

        Only directly adjacent neighbors are searched for, no diagonals.

        Parameters
        ----------
        pixels: dict
            The dictionary produced by the read_pixel_list method of this class

        Returns
        -------
        neighbors: list of lists
            Array of neighbor indices in a list for each pixel
        """
        self._logger.debug("Searching for neighbor pixels")

        if pixels["pixel_shape"] == 1 or pixels["pixel_shape"] == 3:
            self._neighbors = self._find_neighbors(
                pixels["x"],
                pixels["y"],
                self.PMT_NEIGHBOR_RADIUS_FACTOR * pixels["pixel_diameter"],
            )
        elif pixels["pixel_shape"] == 2:
            # Distance increased by 40% to take into account gaps in the SiPM cameras
            # Pixels in the same row/column can be 20% shifted from one another
            # Inside find_adjacent_neighbor_pixels the distance is increased
            # further for pixels in the same row/column to 1.68*diameter.
            self._neighbors = self._find_adjacent_neighbor_pixels(
                pixels["x"],
                pixels["y"],
                self.SIPM_NEIGHBOR_RADIUS_FACTOR * pixels["pixel_diameter"],
                self.SIPM_ROW_COLUMN_DIST_FACTOR * pixels["pixel_diameter"],
            )

        return self._neighbors

    def get_neighbor_pixels(self, pixels: dict | None = None) -> list[list[int]]:
        """
        Get a list of neighbor pixels by calling calc_neighbor_pixels() when necessary.

        The purpose of this function is to ensure the calculation occurs only once and only when
        necessary.

        Parameters
        ----------
        pixels: dict
            The dictionary produced by the read_pixel_list method of this class.

        Returns
        -------
        neighbors: list of lists
            Array of neighbor indices in a list for each pixel.
        """
        if self._neighbors is None:
            if pixels is None:
                pixels = self.pixels
            return self._calc_neighbor_pixels(pixels)

        return self._neighbors

    def _calc_edge_pixels(self, pixels: dict, neighbors: list[list[int]]) -> list[int]:
        """
        Find the edge pixels of the camera.

        Parameters
        ----------
        pixels: dict
            The dictionary produced by the read_pixel_list method of this class.
        neighbors: list of lists
            Array of neighbor indices in a list for each pixel.

        Returns
        -------
        edge_pixel_indices: list
            Array of edge pixel indices.
        """
        self._logger.debug("Searching for edge pixels")
        edge_pixel_indices = []

        def is_edge_pixel(i_pix):
            pixel_shape = pixels["pixel_shape"]
            pix_on = pixels["pix_on"][i_pix]
            num_neighbors = len(neighbors[i_pix])

            shape_condition = (pixel_shape in [1, 3] and num_neighbors < 6) or (
                pixel_shape == 2 and num_neighbors < 4
            )
            return pix_on and shape_condition

        for i_pix, _ in enumerate(pixels["x"]):
            if is_edge_pixel(i_pix):
                edge_pixel_indices.append(i_pix)

        return edge_pixel_indices

    def get_edge_pixels(
        self, pixels: dict | None = None, neighbors: list[list[int]] | None = None
    ) -> list[int]:
        """
        Get the indices of the edge pixels of the camera.

        Parameters
        ----------
        pixels: dict
            The dictionary produced by the read_pixel_list method of this class.
        neighbors: list of lists
            Array of neighbor indices in a list for each pixel.

        Returns
        -------
        edge_pixel_indices: list
            Array of edge pixel indices.
        """
        if self._edge_pixel_indices is None:
            if pixels is None:
                pixels = self.pixels
            if neighbors is None:
                neighbors = self.get_neighbor_pixels()
            return self._calc_edge_pixels(pixels, neighbors)

        return self._edge_pixel_indices
