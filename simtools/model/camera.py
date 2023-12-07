import logging

import astropy.units as u
import matplotlib as mlp
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from scipy.spatial import cKDTree as KDTree
from scipy.spatial import distance

import simtools.visualization.legend_handlers as leg_h
from simtools.model.model_utils import (
    get_camera_name,
    get_telescope_class,
    is_two_mirror_telescope,
)
from simtools.utils.geometry import rotate

__all__ = ["Camera"]


class Camera:
    """
    Camera class, defining pixel layout including rotation, finding neighbour pixels, calculating\
    FoV and plotting the camera.

    Parameters
    ----------
    telescope_model_name: string
        As provided by the telescope model method TelescopeModel (ex South-LST-1).
    camera_config_file: string
        The sim_telarray file name.
    focal_length: float
        The focal length of the camera in (preferably the effective focal length), assumed to be \
        in the same unit as the pixel positions in the camera_config_file, usually cm.
    """

    # Constants for finding neighbour pixels.
    PMT_NEIGHBOR_RADIUS_FACTOR = 1.1
    SIPM_NEIGHBOR_RADIUS_FACTOR = 1.4
    SIPM_ROW_COLUMN_DIST_FACTOR = 0.2

    def __init__(self, telescope_model_name, camera_config_file, focal_length):
        """
        Initialize Camera class, defining pixel layout including rotation, finding neighbour pixels,
        calculating FoV and plotting the camera.
        """

        self._logger = logging.getLogger(__name__)

        self._telescope_model_name = telescope_model_name
        self._camera_name = get_camera_name(self._telescope_model_name)
        self._camera_config_file = camera_config_file
        self._focal_length = focal_length
        if self._focal_length <= 0:
            raise ValueError("The focal length must be larger than zero")
        self._pixels = self.read_pixel_list(camera_config_file)

        self._pixels = self._rotate_pixels(self._pixels)

        # Initialize an empty list of neighbours, to be calculated only when necessary.
        self._neighbours = None

        # Initialize an empty list of edge pixels, to be calculated only when necessary.
        self._edge_pixel_indices = None

    @staticmethod
    def read_pixel_list(camera_config_file):
        """
        Read the pixel layout from the camera config file, assumed to be in a sim_telarray format.

        Parameters
        ----------
        camera_config_file: string
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

        pixels = {}
        pixels["pixel_diameter"] = 9999
        pixels["pixel_shape"] = 9999
        pixels["pixel_spacing"] = 9999
        pixels["lightguide_efficiency_angle_file"] = "none"
        pixels["lightguide_efficiency_wavelength_file"] = "none"
        pixels["rotate_angle"] = 0
        pixels["x"] = []
        pixels["y"] = []
        pixels["pix_id"] = []
        pixels["pix_on"] = []

        with open(camera_config_file, "r", encoding="utf-8") as dat_file:
            for line in dat_file:
                pix_info = line.split()
                if line.startswith("PixType"):
                    pixels["pixel_shape"] = int(pix_info[5].strip())
                    pixels["pixel_diameter"] = float(pix_info[6].strip())
                    pixels["lightguide_efficiency_angle_file"] = (
                        pix_info[8].strip().replace('"', "")
                    )
                    if len(pix_info) > 9:
                        pixels["lightguide_efficiency_wavelength_file"] = (
                            pix_info[9].strip().replace('"', "")
                        )
                if line.startswith("Rotate"):
                    pixels["rotate_angle"] = np.deg2rad(float(pix_info[1].strip()))
                if line.startswith("Pixel"):
                    pixels["x"].append(float(pix_info[3].strip()))
                    pixels["y"].append(float(pix_info[4].strip()))
                    pixels["pix_id"].append(int(pix_info[1].strip()))
                    if len(pix_info) > 9:
                        if int(pix_info[9].strip()) != 0:
                            pixels["pix_on"].append(True)
                        else:
                            pixels["pix_on"].append(False)
                    else:
                        pixels["pix_on"].append(True)

        if pixels["pixel_diameter"] == 9999:
            raise ValueError(f"Could not read the pixel diameter from {camera_config_file} file")
        if pixels["pixel_shape"] not in [1, 2, 3]:
            raise ValueError(
                f"Pixel shape in {camera_config_file} unrecognized (has to be 1, 2 or 3)"
            )

        return pixels

    def _rotate_pixels(self, pixels):
        """
        Rotate the pixels according to the rotation angle given in pixels['rotate_angle'].
        Additional rotation is added to get to the camera view of an observer facing the camera.
        The angle for the axes rotation depends on the coordinate system in which the original
        data was provided.

        Parameters
        ----------
        pixels: dictionary
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

    def get_number_of_pixels(self):
        """
        Get the number of pixels in the camera (all pixels, including those defined as "off".

        Returns
        -------
        int
            number of pixels.
        """

        return len(self._pixels["x"])

    def get_pixel_diameter(self):
        """
        Get pixel diameter contained in _pixels.

        Returns
        -------
        float
            Pixel diameter (usually in cm).
        """

        return self._pixels["pixel_diameter"]

    def get_pixel_active_solid_angle(self):
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
        return pixel_area / (self._focal_length**2)

    def get_pixel_shape(self):
        """
        Get pixel shape code 1, 2 or 3, where 1 and 3 are hexagonal pixels, where one is rotated by\
        30 degrees with respect to the other. A square pixel is denoted as 2.

        Returns
        -------
        int (1, 2 or 3)
            Pixel shape.
        """
        return self._pixels["pixel_shape"]

    def get_lightguide_efficiency_angle_file_name(self):
        """
        Get the file name of the light guide efficiency as a function of incidence angle.

        Returns
        -------
        str
            File name of the light guide efficiency as a function of incidence angle.
        """

        return self._pixels["lightguide_efficiency_angle_file"]

    def get_lightguide_efficiency_wavelength_file_name(self):
        """
        Get the file name of the light guide efficiency as a function of wavelength.

        Returns
        -------
        str
            File name of the light guide efficiency as a function of wavelength.
        """
        return self._pixels["lightguide_efficiency_wavelength_file"]

    def get_camera_fill_factor(self):
        """
        Calculate the fill factor of the camera, defined as (pixel_diameter/pixel_spacing)**2

        Returns
        -------
        float
            The camera fill factor.
        """

        if self._pixels["pixel_spacing"] == 9999:
            points = np.array([self._pixels["x"], self._pixels["y"]]).T
            pixel_distances = distance.cdist(points, points, "euclidean")
            self._pixels["pixel_spacing"] = np.min(pixel_distances[pixel_distances > 0])

        return (self._pixels["pixel_diameter"] / self._pixels["pixel_spacing"]) ** 2

    def calc_fov(self):
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
            self._pixels["x"],
            self._pixels["y"],
            self.get_edge_pixels(),
            self._focal_length,
        )

    def _calc_fov(self, x_pixel, y_pixel, edge_pixel_indices, focal_length):
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
    def _find_neighbours(x_pos, y_pos, radius):
        """
        use a KD-Tree to quickly find nearest neighbours (e.g., of the pixels in a camera or mirror\
        facets)

        Parameters
        ----------
        x_pos : numpy.array_like
            x position of each e.g., pixel
        y_pos : numpy.array_like
            y position of each e.g., pixel
        radius : float
            radius to consider neighbour it should be slightly larger than the pixel diameter or \
            mirror facet.

        Returns
        -------
        neighbours: numpy.array_like
            Array of neighbour indices in a list for each e.g., pixel.
        """

        points = np.array([x_pos, y_pos]).T
        indices = np.arange(len(x_pos))
        kdtree = KDTree(points)
        neighbours = [kdtree.query_ball_point(p, r=radius) for p in points]

        for neighbour_now, index_now in zip(neighbours, indices):
            neighbour_now.remove(index_now)  # get rid of the pixel or mirror itself

        return neighbours

    def _find_adjacent_neighbour_pixels(self, x_pos, y_pos, radius, row_coloumn_dist):
        """
        Find adjacent neighbour pixels in cameras with square pixels. Only directly adjacent \
        neighbours are allowed, no diagonals.

        Parameters
        ----------
        x_pos : numpy.array_like
            x position of each pixel
        y_pos : numpy.array_like
            y position of each pixels
        radius : float
            radius to consider neighbour.
            Should be slightly larger than the pixel diameter.
        row_coloumn_dist : float
            Maximum distance for pixels in the same row/column to consider when looking for a \
            neighbour. Should be around 20% of the pixel diameter.

        Returns
        -------
        neighbours: numpy.array_like
            Array of neighbour indices in a list for each pixel
        """

        # First find the neighbours with the usual method and the original radius
        # which does not allow for diagonal neighbours.
        neighbours = self._find_neighbours(x_pos, y_pos, radius)
        for i_pix, nn in enumerate(neighbours):
            # Find pixels defined as edge pixels now
            if len(nn) < 4:
                # Go over all other pixels and search for ones which are adjacent
                # but further than sqrt(2) away
                for j_pix, _ in enumerate(x_pos):
                    # No need to look at the pixel itself
                    # nor at any pixels already in the neighbours list
                    if j_pix != i_pix and j_pix not in nn:
                        dist = np.sqrt(
                            (x_pos[i_pix] - x_pos[j_pix]) ** 2 + (y_pos[i_pix] - y_pos[j_pix]) ** 2
                        )
                        # Check if this pixel is in the same row or column
                        # and allow it to be ~1.68*diameter away (1.4*1.2 = 1.68)
                        # Need to increase the distance because of the curvature
                        # of the CHEC camera
                        if (
                            abs(x_pos[i_pix] - x_pos[j_pix]) < row_coloumn_dist
                            or abs(y_pos[i_pix] - y_pos[j_pix]) < row_coloumn_dist
                        ) and dist < 1.2 * radius:
                            nn.append(j_pix)

        return neighbours

    def _calc_neighbour_pixels(self, pixels):
        """
        Find adjacent neighbour pixels in cameras with hexagonal or square pixels. Only directly \
        adjacent neighbours are searched for, no diagonals.

        Parameters
        ----------
        pixels: dictionary
            The dictionary produced by the read_pixel_list method of this class

        Returns
        -------
        neighbours: numpy.array_like
            Array of neighbour indices in a list for each pixel
        """

        self._logger.debug("Searching for neighbour pixels")

        if pixels["pixel_shape"] == 1 or pixels["pixel_shape"] == 3:
            self._neighbours = self._find_neighbours(
                pixels["x"],
                pixels["y"],
                self.PMT_NEIGHBOR_RADIUS_FACTOR * pixels["pixel_diameter"],
            )
        elif pixels["pixel_shape"] == 2:
            # Distance increased by 40% to take into account gaps in the SiPM cameras
            # Pixels in the same row/column can be 20% shifted from one another
            # Inside find_adjacent_neighbour_pixels the distance is increased
            # further for pixels in the same row/column to 1.68*diameter.
            self._neighbours = self._find_adjacent_neighbour_pixels(
                pixels["x"],
                pixels["y"],
                self.SIPM_NEIGHBOR_RADIUS_FACTOR * pixels["pixel_diameter"],
                self.SIPM_ROW_COLUMN_DIST_FACTOR * pixels["pixel_diameter"],
            )

        return self._neighbours

    def get_neighbour_pixels(self, pixels=None):
        """
        Get a list of neighbour pixels by calling calc_neighbour_pixels() when necessary. The \
        purpose of this function is to ensure the calculation occurs only once and only when \
        necessary.

        Parameters
        ----------
        pixels: dict
            The dictionary produced by the read_pixel_list method of this class.

        Returns
        -------
        neighbours: numpy.array_like
            Array of neighbour indices in a list for each pixel.
        """

        if self._neighbours is None:
            if pixels is None:
                pixels = self._pixels
            return self._calc_neighbour_pixels(pixels)

        return self._neighbours

    def _calc_edge_pixels(self, pixels, neighbours):
        """
        Find the edge pixels of the camera.

        Parameters
        ----------
        pixels: dictionary
            The dictionary produced by the read_pixel_list method of this class.
        neighbours: numpy.array_like
            Array of neighbour indices in a list for each pixel.

        Returns
        -------
        edge_pixel_indices: numpy.array_like
            Array of edge pixel indices.
        """

        self._logger.debug("Searching for edge pixels")

        edge_pixel_indices = []

        for i_pix, _ in enumerate(pixels["x"]):
            if pixels["pixel_shape"] == 1 or pixels["pixel_shape"] == 3:
                if pixels["pix_on"][i_pix]:
                    if len(neighbours[i_pix]) < 6:
                        edge_pixel_indices.append(i_pix)
            elif pixels["pixel_shape"] == 2:
                if pixels["pix_on"][i_pix]:
                    if len(neighbours[i_pix]) < 4:
                        edge_pixel_indices.append(i_pix)

        return edge_pixel_indices

    def get_edge_pixels(self, pixels=None, neighbours=None):
        """
        Get the indices of the edge pixels of the camera.

        Parameters
        ----------
        pixels: dict
            The dictionary produced by the read_pixel_list method of this class.
        neighbours: numpy.array_like
            Array of neighbour indices in a list for each pixel.

        Returns
        -------
        edge_pixel_indices: numpy.array_like
            Array of edge pixel indices.
        """

        if self._edge_pixel_indices is None:
            if pixels is None:
                pixels = self._pixels
            if neighbours is None:
                neighbours = self.get_neighbour_pixels()
            return self._calc_edge_pixels(pixels, neighbours)

        return self._edge_pixel_indices

    def _plot_axes_def(self, plot, rotate_angle):
        """
        Plot three axes definitions on the pyplot.plt instance provided. The three axes are Alt/Az,\
        the camera coordinate system and the original coordinate system the pixel list was provided.

        Parameters
        ----------
        plot: pyplot.plt instance
            A pyplot.plt instance where to add the axes definitions.
        rotate_angle: float
            The rotation angle applied
        """

        invert_yaxis = False
        x_left = 0.7  # Position of the left most axis
        if not is_two_mirror_telescope(self._telescope_model_name):
            invert_yaxis = True
            x_left = 0.8

        x_title = r"$x_{\!pix}$"
        y_title = r"$y_{\!pix}$"
        x_pos, y_pos = (x_left, 0.12)
        # The rotation of LST (above 100 degrees) raises the axes.
        # In this case, lower the starting point.
        if np.rad2deg(rotate_angle) > 100:
            y_pos -= 0.09
            x_pos -= 0.05
        kwargs = {
            "x_title": x_title,
            "y_title": y_title,
            "x_pos": x_pos,
            "y_pos": y_pos,
            "rotate_angle": rotate_angle - (1 / 2.0) * np.pi,
            "fc": "black",
            "ec": "black",
            "invert_yaxis": invert_yaxis,
        }
        self._plot_one_axis_def(plot, **kwargs)

        x_title = r"$x_{\!cam}$"
        y_title = r"$y_{\!cam}$"
        x_pos, y_pos = (x_left + 0.15, 0.12)
        kwargs = {
            "x_title": x_title,
            "y_title": y_title,
            "x_pos": x_pos,
            "y_pos": y_pos,
            "rotate_angle": (3 / 2.0) * np.pi,
            "fc": "blue",
            "ec": "blue",
            "invert_yaxis": invert_yaxis,
        }
        self._plot_one_axis_def(plot, **kwargs)

        x_title = "Alt"
        y_title = "Az"
        x_pos, y_pos = (x_left + 0.15, 0.25)
        kwargs = {
            "x_title": x_title,
            "y_title": y_title,
            "x_pos": x_pos,
            "y_pos": y_pos,
            "rotate_angle": (3 / 2.0) * np.pi,
            "fc": "red",
            "ec": "red",
            "invert_yaxis": invert_yaxis,
        }
        self._plot_one_axis_def(plot, **kwargs)

    @staticmethod
    def _plot_one_axis_def(plot, **kwargs):
        """
        Plot an axis on the pyplot.plt instance provided.

        Parameters
        ----------
        plot: pyplot.plt instance
            A pyplot.plt instance where to add the axes definitions.
        **kwargs: dict
             x_title: str
                x-axis title
             y_title: str
                y-axis title,
             x_pos: float
                x position of the axis to draw
             y_pos: float
                y position of the axis to draw
             rotate_angle: float
                rotation angle of the axis in radians
             fc: str
                face colour of the axis
             ec: str
                edge colour of the axis
             invert_yaxis: bool
                Flag to invert the y-axis (for dual mirror telescopes).
        """

        x_title = kwargs["x_title"]
        y_title = kwargs["y_title"]
        x_pos, y_pos = (kwargs["x_pos"], kwargs["y_pos"])

        r = 0.1  # size of arrow
        sign = 1.0
        if kwargs["invert_yaxis"]:
            sign *= -1.0
        x_text1 = x_pos + sign * r * np.cos(kwargs["rotate_angle"])
        y_text1 = y_pos + r * np.sin(0 + kwargs["rotate_angle"])
        x_text2 = x_pos + sign * r * np.cos(np.pi / 2.0 + kwargs["rotate_angle"])
        y_text2 = y_pos + r * np.sin(np.pi / 2.0 + kwargs["rotate_angle"])

        plot.gca().annotate(
            x_title,
            xy=(x_pos, y_pos),
            xytext=(x_text1, y_text1),
            xycoords="axes fraction",
            ha="center",
            va="center",
            size="xx-large",
            arrowprops={
                "arrowstyle": "<|-",
                "shrinkA": 0,
                "shrinkB": 0,
                "fc": kwargs["fc"],
                "ec": kwargs["ec"],
            },
        )

        plot.gca().annotate(
            y_title,
            xy=(x_pos, y_pos),
            xytext=(x_text2, y_text2),
            xycoords="axes fraction",
            ha="center",
            va="center",
            size="xx-large",
            arrowprops={
                "arrowstyle": "<|-",
                "shrinkA": 0,
                "shrinkB": 0,
                "fc": kwargs["fc"],
                "ec": kwargs["ec"],
            },
        )

    def plot_pixel_layout(self, camera_in_sky_coor=False, pixels_id_to_print=50):
        """
        Plot the pixel layout for an observer facing the camera. Including in the plot edge pixels,\
        off pixels, pixel ID for the first 50 pixels, coordinate systems, FOV, focal length and the\
        average edge radius.

        Returns
        -------
        fig: plt.figure instance
            Figure with the pixel layout.
        """

        self._logger.info(f"Plotting the {self._telescope_model_name} camera")

        fig, ax = plt.subplots()
        plt.gcf().set_size_inches(8, 8)

        if not is_two_mirror_telescope(self._telescope_model_name):
            if not camera_in_sky_coor:
                self._pixels["y"] = [(-1) * y_val for y_val in self._pixels["y"]]

        on_pixels, edge_pixels, off_pixels = [], [], []

        for i_pix, xy_pix_pos in enumerate(zip(self._pixels["x"], self._pixels["y"])):
            if self._pixels["pixel_shape"] == 1 or self._pixels["pixel_shape"] == 3:
                hexagon = mpatches.RegularPolygon(
                    (xy_pix_pos[0], xy_pix_pos[1]),
                    numVertices=6,
                    radius=self._pixels["pixel_diameter"] / np.sqrt(3),
                    orientation=np.deg2rad(self._pixels["orientation"]),
                )
                if self._pixels["pix_on"][i_pix]:
                    if len(self.get_neighbour_pixels()[i_pix]) < 6:
                        edge_pixels.append(hexagon)
                    else:
                        on_pixels.append(hexagon)
                else:
                    off_pixels.append(hexagon)
            elif self._pixels["pixel_shape"] == 2:
                square = mpatches.Rectangle(
                    (
                        xy_pix_pos[0] - self._pixels["pixel_diameter"] / 2.0,
                        xy_pix_pos[1] - self._pixels["pixel_diameter"] / 2.0,
                    ),
                    width=self._pixels["pixel_diameter"],
                    height=self._pixels["pixel_diameter"],
                )
                if self._pixels["pix_on"][i_pix]:
                    if len(self.get_neighbour_pixels()[i_pix]) < 4:
                        edge_pixels.append(square)
                    else:
                        on_pixels.append(square)
                else:
                    off_pixels.append(square)

            if self._pixels["pix_id"][i_pix] < pixels_id_to_print + 1:
                font_size = 4
                if get_telescope_class(self._telescope_model_name) == "SCT":
                    font_size = 2
                plt.text(
                    xy_pix_pos[0],
                    xy_pix_pos[1],
                    self._pixels["pix_id"][i_pix],
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=font_size,
                )

        ax.add_collection(
            PatchCollection(on_pixels, facecolor="none", edgecolor="black", linewidth=0.2)
        )
        ax.add_collection(
            PatchCollection(
                edge_pixels,
                facecolor=mcolors.to_rgb("brown") + (0.5,),
                edgecolor=mcolors.to_rgb("black") + (1,),
                linewidth=0.2,
            )
        )
        ax.add_collection(
            PatchCollection(off_pixels, facecolor="black", edgecolor="black", linewidth=0.2)
        )

        legend_objects = [leg_h.PixelObject(), leg_h.EdgePixelObject()]
        legend_labels = ["Pixel", "Edge pixel"]
        if isinstance(on_pixels[0], mlp.patches.RegularPolygon):
            legend_handler_map = {
                leg_h.PixelObject: leg_h.HexPixelHandler(),
                leg_h.EdgePixelObject: leg_h.HexEdgePixelHandler(),
                leg_h.OffPixelObject: leg_h.HexOffPixelHandler(),
            }
        elif isinstance(on_pixels[0], mlp.patches.Rectangle):
            legend_handler_map = {
                leg_h.PixelObject: leg_h.SquarePixelHandler(),
                leg_h.EdgePixelObject: leg_h.SquareEdgePixelHandler(),
                leg_h.OffPixelObject: leg_h.SquareOffPixelHandler(),
            }

        if len(off_pixels) > 0:
            legend_objects.append(leg_h.OffPixelObject())
            legend_labels.append("Disabled pixel")

        plt.axis("equal")
        plt.grid(True)
        ax.set_axisbelow(True)
        plt.axis(
            [
                min(self._pixels["x"]),
                max(self._pixels["x"]),
                min(self._pixels["y"]) * 1.42,
                max(self._pixels["y"]) * 1.42,
            ]
        )
        plt.xlabel("Horizontal scale [cm]", fontsize=18, labelpad=0)
        plt.ylabel("Vertical scale [cm]", fontsize=18, labelpad=0)
        ax.set_title(
            f"Pixels layout in {self._telescope_model_name:s} camera",
            fontsize=15,
            y=1.02,
        )
        plt.tick_params(axis="both", which="major", labelsize=15)

        self._plot_axes_def(plt, self._pixels["rotate_angle"])
        description = "For an observer facing the camera"
        if camera_in_sky_coor and not is_two_mirror_telescope(self._telescope_model_name):
            description = "For an observer behind the camera looking through"
        if is_two_mirror_telescope(self._telescope_model_name):
            description = "For an observer looking from secondary to camera"
        ax.text(
            0.02,
            0.02,
            description,
            transform=ax.transAxes,
            color="black",
            fontsize=12,
        )

        fov, r_edge_avg = self.calc_fov()
        ax.text(
            0.02,
            0.96,
            r"$f_{\mathrm{eff}}$ = " + f"{self._focal_length:.3f} cm",
            transform=ax.transAxes,
            color="black",
            fontsize=12,
        )
        ax.text(
            0.02,
            0.92,
            f"Avg. edge radius = {r_edge_avg:.3f} cm",
            transform=ax.transAxes,
            color="black",
            fontsize=12,
        )
        ax.text(
            0.02,
            0.88,
            f"FoV = {fov:.3f} deg",
            transform=ax.transAxes,
            color="black",
            fontsize=12,
        )

        plt.legend(
            legend_objects,
            legend_labels,
            handler_map=legend_handler_map,
            prop={"size": 11},
            loc="upper right",
        )

        ax.set_aspect("equal", "datalim")
        plt.tight_layout()

        return fig
