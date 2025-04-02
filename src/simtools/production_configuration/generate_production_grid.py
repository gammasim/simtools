"""
Module defines the `GridGeneration` class.

Used to generate a grid of simulation points based on flexible axes definitions such as energy,
 azimuth, zenith angle, night-sky background,
and camera offset. The module handles axis scaling, binning,
and distribution types, allowing for adaptable simulation configurations. Additionally,
it provides functionality for converting between Altitude/Azimuth and Right Ascension
Declination coordinates.
"""

import json
import logging

import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.table import Table
from astropy.time import Time
from astropy.units import Quantity
from scipy.interpolate import griddata


class GridGeneration:
    """
    Defines and generates a grid of simulation points based on flexible axes definitions.

    This class generates a grid of points for a simulation based on parameters like energy,
    azimuth, zenith angle, night-sky background, and camera offset,
    taking into account axis definitions, scaling, and units.
    """

    def __init__(
        self,
        axes: dict,
        coordinate_system: str = "zenith_azimuth",
        observing_location=None,
        observing_time=None,
        lookup_table: str | None = None,
        telescope_ids: list | None = None,
    ):
        """
        Initialize the grid with the given axes and coordinate system.

        Parameters
        ----------
        axes : dict
            Dictionary where each key is the axis name and the value is a dictionary
            defining the axis properties (range, binning, scaling, etc.).
        coordinate_system : str, optional
            The coordinate system for the grid generation (default is 'zenith_azimuth').
        observing_location : EarthLocation, optional
            The location of the observation (latitude, longitude, height).
        observing_time : Time, optional
            The time of the observation.
        lookup_table : str, optional
            Path to the lookup table file (ECSV format).
        telescope_ids : list of int, optional
            List of telescope IDs to get the limits for.
        """
        self._logger = logging.getLogger(__name__)

        self.axes = axes["axes"] if "axes" in axes else axes
        self.coordinate_system = coordinate_system
        self.observing_location = (
            observing_location
            if observing_location is not None
            else EarthLocation(lat=0.0 * u.deg, lon=0.0 * u.deg, height=0 * u.m)
        )
        self.observing_time = observing_time if observing_time is not None else Time.now()
        self.lookup_table = lookup_table
        self.telescope_ids = telescope_ids

        if self.lookup_table:
            self._apply_lookup_table_limits()

    def _apply_lookup_table_limits(self):
        """Apply interpolated limits from the provided lookup table to the axes."""
        lookup_table = Table.read(self.lookup_table, format="ascii.ecsv")

        if isinstance(lookup_table["telescope_ids"][0], str):
            lookup_table["telescope_ids"] = [
                json.loads(telescope_ids) for telescope_ids in lookup_table["telescope_ids"]
            ]

        matching_rows = [
            row for row in lookup_table if set(self.telescope_ids) == set(row["telescope_ids"])
        ]

        if not matching_rows:
            raise ValueError(
                f"No matching rows in the lookup table for telescope_ids: {self.telescope_ids}"
            )

        zeniths = np.array([row["zenith"] for row in matching_rows])
        azimuths = np.array([row["azimuth"] for row in matching_rows]) % 360  # Normalize azimuths
        lower_energy_thresholds = np.array([row["lower_energy_threshold"] for row in matching_rows])
        upper_radius_thresholds = np.array([row["upper_radius_threshold"] for row in matching_rows])
        viewcone_radii = np.array([row["viewcone_radius"] for row in matching_rows])

        target_zeniths = np.linspace(
            self.axes["zenith_angle"]["range"][0],
            self.axes["zenith_angle"]["range"][1],
            self.axes["zenith_angle"]["binning"],
        )
        target_azimuths = self.create_circular_binning(
            self.axes["azimuth"]["range"], self.axes["azimuth"]["binning"]
        )

        is_flat_zenith = len(np.unique(zeniths)) == 1
        is_flat_azimuth = len(np.unique(azimuths)) == 1

        if is_flat_zenith and is_flat_azimuth:
            self._logger.info("Both zenith and azimuth are flat, assigning constant values.")
            interpolated_energy = np.full(
                target_zeniths.size * target_azimuths.size, lower_energy_thresholds[0]
            )
            interpolated_radius = np.full(
                target_zeniths.size * target_azimuths.size, upper_radius_thresholds[0]
            )
            interpolated_viewcone = np.full(
                target_zeniths.size * target_azimuths.size, viewcone_radii[0]
            )
        elif is_flat_zenith:
            self._logger.info("Zenith is flat, interpolating only along azimuth.")
            interpolated_energy = np.interp(
                target_azimuths, azimuths, lower_energy_thresholds, left=np.nan, right=np.nan
            )
            interpolated_radius = np.interp(
                target_azimuths, azimuths, upper_radius_thresholds, left=np.nan, right=np.nan
            )
            interpolated_viewcone = np.interp(
                target_azimuths, azimuths, viewcone_radii, left=np.nan, right=np.nan
            )
            interpolated_energy = np.tile(interpolated_energy, (len(target_zeniths), 1))
            interpolated_radius = np.tile(interpolated_radius, (len(target_zeniths), 1))
            interpolated_viewcone = np.tile(interpolated_viewcone, (len(target_zeniths), 1))
        elif is_flat_azimuth:
            self._logger.info("Azimuth is flat, interpolating only along zenith.")
            interpolated_energy = np.interp(
                target_zeniths, zeniths, lower_energy_thresholds, left=np.nan, right=np.nan
            )
            interpolated_radius = np.interp(
                target_zeniths, zeniths, upper_radius_thresholds, left=np.nan, right=np.nan
            )
            interpolated_viewcone = np.interp(
                target_zeniths, zeniths, viewcone_radii, left=np.nan, right=np.nan
            )
            interpolated_energy = np.tile(
                interpolated_energy[:, np.newaxis], (1, len(target_azimuths))
            )
            interpolated_radius = np.tile(
                interpolated_radius[:, np.newaxis], (1, len(target_azimuths))
            )
            interpolated_viewcone = np.tile(
                interpolated_viewcone[:, np.newaxis], (1, len(target_azimuths))
            )
        else:
            self._logger.info("Neither zenith nor azimuth is flat, performing 2D interpolation.")
            target_grid = (
                np.array(np.meshgrid(target_zeniths, target_azimuths, indexing="ij"))
                .reshape(2, -1)
                .T
            )
            interpolated_energy = griddata(
                points=np.column_stack((zeniths, azimuths)),
                values=lower_energy_thresholds,
                xi=target_grid,
                method="linear",
                fill_value=np.nan,
            )
            interpolated_radius = griddata(
                points=np.column_stack((zeniths, azimuths)),
                values=upper_radius_thresholds,
                xi=target_grid,
                method="linear",
                fill_value=np.nan,
            )
            interpolated_viewcone = griddata(
                points=np.column_stack((zeniths, azimuths)),
                values=viewcone_radii,
                xi=target_grid,
                method="linear",
                fill_value=np.nan,
            )
            interpolated_energy = interpolated_energy.reshape(
                len(target_zeniths), len(target_azimuths)
            )
            interpolated_radius = interpolated_radius.reshape(
                len(target_zeniths), len(target_azimuths)
            )
            interpolated_viewcone = interpolated_viewcone.reshape(
                len(target_zeniths), len(target_azimuths)
            )

        self.interpolated_limits = {
            "energy": interpolated_energy,
            "radius": interpolated_radius,
            "viewcone": interpolated_viewcone,
            "target_zeniths": target_zeniths,
            "target_azimuths": target_azimuths,
        }

    def create_circular_binning(self, azimuth_range, num_bins):
        """
        Create bin centers for azimuth angles, handling circular wraparound (0° to 360°).

        Parameters
        ----------
        azimuth_range : tuple
            (min_azimuth, max_azimuth), can wrap around 0°.
        num_bins : int
            Number of bins.

        Returns
        -------
        np.ndarray
            Array of bin centers.
        """
        azimuth_min, azimuth_max = azimuth_range
        azimuth_min %= 360  # Normalize to [0, 360)
        azimuth_max %= 360

        if azimuth_min > azimuth_max:  # Handles wraparound case (e.g., 310° to 20°)
            # Total range is split into two parts: [azimuth_min, 360) and [0, azimuth_max]
            total_range = (360 - azimuth_min) + azimuth_max

            bin_edges = np.linspace(azimuth_min, azimuth_min + total_range, num_bins, endpoint=True)
            bin_centers = bin_edges % 360  # Wrap around to [0, 360)
        else:
            # range does not cross 360°
            bin_centers = np.linspace(azimuth_min, azimuth_max, num_bins, endpoint=True)

        return bin_centers

    def generate_grid(self) -> list[dict]:
        """
        Generate the grid based on the defined axes and include interpolated limits.

        Takes energy threshold, viewcone, and radius from the interpolated lookup table.

        Returns
        -------
        list of dict
            A list of grid points, each represented as a dictionary with axis names
            as keys and axis values as values. Axis values may include units where defined.
        """
        axis_values = {}

        for axis_name, axis in self.axes.items():
            if axis_name in ["energy", "viewcone", "radius"]:
                continue  # Skip fixed coordinates
            axis_range = axis["range"]
            binning = axis["binning"]
            scaling = axis.get("scaling", "linear")
            units = axis.get("units", None)

            if scaling == "log":
                values = np.logspace(np.log10(axis_range[0]), np.log10(axis_range[1]), binning)
            else:
                values = np.linspace(axis_range[0], axis_range[1], binning)

            if units:
                values = values * u.Unit(units)

            axis_values[axis_name] = values

        value_arrays = [value.value for value in axis_values.values()]
        units = [value.unit for value in axis_values.values()]
        grid = np.meshgrid(*value_arrays, indexing="ij")
        combinations = np.vstack(list(map(np.ravel, grid))).T

        grid_points = []
        for combination in combinations:
            grid_point = {
                key: Quantity(combination[i], units[i]) for i, key in enumerate(axis_values.keys())
            }

            if "energy" in self.interpolated_limits:
                zenith_idx = np.searchsorted(
                    self.interpolated_limits["target_zeniths"], grid_point["zenith_angle"].value
                )
                azimuth_idx = np.searchsorted(
                    self.interpolated_limits["target_azimuths"], grid_point["azimuth"].value
                )

                zenith_idx = np.clip(
                    zenith_idx, 0, len(self.interpolated_limits["target_zeniths"]) - 1
                )
                azimuth_idx = np.clip(
                    azimuth_idx, 0, len(self.interpolated_limits["target_azimuths"]) - 1
                )

                energy_lower = self.interpolated_limits["energy"][zenith_idx, azimuth_idx]
                grid_point["energy_range"] = {
                    "lower": energy_lower * u.TeV,
                    "upper": self.axes["energy"]["range"][1] * u.TeV,
                }

            if "radius" in self.interpolated_limits:
                radius_value = self.interpolated_limits["radius"][zenith_idx, azimuth_idx]
                grid_point["radius"] = radius_value * u.m

            if "viewcone" in self.interpolated_limits:
                viewcone_value = self.interpolated_limits["viewcone"][zenith_idx, azimuth_idx]
                grid_point["viewcone"] = viewcone_value * u.deg

            grid_points.append(grid_point)

        return grid_points

    def generate_power_law_values(self, axis_range, binning, power_law_index=3):
        """
        Generate axis values following a power-law distribution.

        Parameters
        ----------
        axis_range : tuple
            Range of the axis values.
        binning : int
            Number of bins (points) to generate.
        power_law_index : int
            Index of the power-law function.

        Returns
        -------
        numpy.ndarray
            Array of axis values following a power-law distribution.
        """
        lin_space = np.linspace(0, 1, binning)
        lin_space = np.clip(lin_space, 1e-10, 1 - 1e-10)  # Avoid division by zero
        return axis_range[0] + (axis_range[1] - axis_range[0]) * (lin_space) ** (power_law_index)

    def convert_altaz_to_radec(self, alt, az):
        """
        Convert Altitude/Azimuth (AltAz) coordinates to Right Ascension/Declination (RA/Dec).

        Parameters
        ----------
        alt : float
            Altitude angle in degrees.
        az : float
            Azimuth angle in degrees.

        Returns
        -------
        SkyCoord
            SkyCoord object containing the RA/Dec coordinates.
        """
        alt_rad = alt.to(u.rad)
        az_rad = az.to(u.rad)
        aa = AltAz(
            alt=alt_rad,
            az=az_rad,
            location=self.observing_location,
            obstime=self.observing_time,
        )
        skycoord = SkyCoord(aa)
        return skycoord.icrs  # Return RA/Dec in ICRS frame

    def convert_coordinates(self, grid_points: list[dict]) -> list[dict]:
        """
        Convert the grid points to RA/Dec coordinates if necessary.

        Parameters
        ----------
        grid_points : list of dict
            List of grid points, each represented as a dictionary with axis
              names as keys and values.

        Returns
        -------
        list of dict
            The grid points with converted RA/Dec coordinates.
        """
        if self.coordinate_system == "ra_dec":
            for point in grid_points:
                if "zenith_angle" in point and "azimuth" in point:
                    alt = (90.0 * u.deg) - point.pop("zenith_angle")
                    az = point.pop("azimuth")
                    radec = self.convert_altaz_to_radec(alt, az)
                    point["ra"] = radec.ra.deg * u.deg
                    point["dec"] = radec.dec.deg * u.deg
        return grid_points

    def clean_grid_output(self, grid_points, output_file=None):
        """Clean the grid output and save to a file or print to the console."""
        cleaned_points = []

        def serialize_quantity(value):
            """Serialize Quantity objects."""
            if hasattr(value, "unit"):
                return {"value": value.value, "unit": str(value.unit)}
            return value

        for point in grid_points:
            cleaned_point = {}
            for key, value in point.items():
                if isinstance(value, dict):
                    # nested dictionaries
                    cleaned_point[key] = {k: serialize_quantity(v) for k, v in value.items()}
                else:
                    cleaned_point[key] = serialize_quantity(value)

            cleaned_points.append(cleaned_point)

        output_data = json.dumps(cleaned_points, indent=4)

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(output_data)
            self._logger.info(f"Output saved to {output_file}")
        else:
            self._logger.info(output_data)
