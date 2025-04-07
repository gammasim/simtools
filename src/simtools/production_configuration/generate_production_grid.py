"""
Module defines the `GridGeneration` class.

Used to generate a grid of simulation points based on flexible axes definitions such
azimuth, zenith angle, night-sky background, and camera offset.
The module handles axis binning, scaling and interpolation of energy thresholds, viewcone,
and radius limits from a lookup table.
Additionally, it allows for converting between Altitude/Azimuth and Right Ascension
Declination coordinates. The resulting grid points are saved to a file.
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

    This class generates a grid of points for a simulation based on parameters such as
    azimuth, zenith angle, night-sky background, and camera offset,
    taking into account axis definitions, scaling, and units and interpolating values
    for simulations from a lookup table.
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

        # Store target values for each axis
        self.target_values = self._generate_target_values()

        if self.lookup_table:
            self._apply_lookup_table_limits()

    def _generate_target_values(self):
        """
        Generate target axis values and store them as Quantities.

        Returns
        -------
        dict
            Dictionary of target values for each axis, stored as Quantity objects.
        """
        target_values = {}
        for axis_name, axis in self.axes.items():
            axis_range = axis["range"]
            binning = axis["binning"]
            scaling = axis.get("scaling", "linear")
            units = axis.get("units", None)

            if axis_name == "azimuth":
                # Use circular binning for azimuth
                values = self.create_circular_binning(axis_range, binning)
            elif scaling == "log":
                # Log scaling
                values = np.logspace(np.log10(axis_range[0]), np.log10(axis_range[1]), binning)
            elif scaling == "1/cos":
                # 1/cos scaling
                cos_min = np.cos(np.radians(axis_range[0]))
                cos_max = np.cos(np.radians(axis_range[1]))
                inv_cos_values = np.linspace(1 / cos_min, 1 / cos_max, binning)
                values = np.degrees(np.arccos(1 / inv_cos_values))
            else:
                # Linear scaling
                values = np.linspace(axis_range[0], axis_range[1], binning)

            if units:
                values = values * u.Unit(units)

            target_values[axis_name] = values

        return target_values

    def _apply_lookup_table_limits(self):
        """Apply limits from the lookup table and interpolate values."""
        lookup_table = Table.read(self.lookup_table, format="ascii.ecsv")

        matching_rows = [
            row for row in lookup_table if set(self.telescope_ids) == set(row["telescope_ids"])
        ]

        if not matching_rows:
            raise ValueError(
                f"No matching rows in the lookup table for telescope_ids: {self.telescope_ids}"
            )

        zeniths = np.array([row["zenith"] for row in matching_rows])
        azimuths = np.array([row["azimuth"] for row in matching_rows]) % 360
        nsb_values = np.array([1 if row["nsb"] == "dark" else 5 for row in matching_rows])
        lower_energy_thresholds = np.array([row["lower_energy_threshold"] for row in matching_rows])
        upper_radius_thresholds = np.array([row["upper_radius_threshold"] for row in matching_rows])
        viewcone_radii = np.array([row["viewcone_radius"] for row in matching_rows])

        azimuths_wrapped = np.concatenate([azimuths, azimuths + 360, azimuths - 360])
        zeniths_wrapped = np.tile(zeniths, 3)
        nsb_wrapped = np.tile(nsb_values, 3)
        lower_energy_thresholds_wrapped = np.tile(lower_energy_thresholds, 3)
        upper_radius_thresholds_wrapped = np.tile(upper_radius_thresholds, 3)
        viewcone_radii_wrapped = np.tile(viewcone_radii, 3)

        target_zeniths = self.target_values["zenith_angle"].value
        target_azimuths = self.target_values["azimuth"].value
        target_nsb = self.target_values["nsb"].value

        target_grid = (
            np.array(np.meshgrid(target_zeniths, target_azimuths, target_nsb, indexing="ij"))
            .reshape(3, -1)
            .T
        )
        interpolated_energy = griddata(
            points=np.column_stack((zeniths_wrapped, azimuths_wrapped, nsb_wrapped)),
            values=lower_energy_thresholds_wrapped,
            xi=target_grid,
            method="linear",
            fill_value=np.nan,
        )
        interpolated_radius = griddata(
            points=np.column_stack((zeniths_wrapped, azimuths_wrapped, nsb_wrapped)),
            values=upper_radius_thresholds_wrapped,
            xi=target_grid,
            method="linear",
            fill_value=np.nan,
        )
        interpolated_viewcone = griddata(
            points=np.column_stack((zeniths_wrapped, azimuths_wrapped, nsb_wrapped)),
            values=viewcone_radii_wrapped,
            xi=target_grid,
            method="linear",
            fill_value=np.nan,
        )

        interpolated_energy = interpolated_energy.reshape(
            len(target_zeniths), len(target_azimuths), len(target_nsb)
        )
        interpolated_radius = interpolated_radius.reshape(
            len(target_zeniths), len(target_azimuths), len(target_nsb)
        )
        interpolated_viewcone = interpolated_viewcone.reshape(
            len(target_zeniths), len(target_azimuths), len(target_nsb)
        )

        self.interpolated_limits = {
            "energy": interpolated_energy,
            "radius": interpolated_radius,
            "viewcone": interpolated_viewcone,
        }

    def create_circular_binning(self, azimuth_range, num_bins):
        """
        Create bin centers for azimuth angles, handling circular wraparound (0 deg to 360 deg).

        Parameters
        ----------
        azimuth_range : tuple
            (min_azimuth, max_azimuth), can wrap around 0 deg.
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

        clockwise_distance = (azimuth_max - azimuth_min) % 360
        counterclockwise_distance = (azimuth_min - azimuth_max) % 360

        if clockwise_distance <= counterclockwise_distance:
            bin_centers = (
                np.linspace(azimuth_min, azimuth_min + clockwise_distance, num_bins, endpoint=True)
                % 360
            )
        else:
            bin_centers = (
                np.linspace(
                    azimuth_min, azimuth_min - counterclockwise_distance, num_bins, endpoint=True
                )
                % 360
            )

        return bin_centers

    def generate_grid(self) -> list[dict]:
        """
        Generate the grid based on the required axes and include interpolated limits.

        Takes energy threshold, viewcone, and radius from the interpolated lookup table.

        Returns
        -------
        list of dict
            A list of grid points, each represented as a dictionary with axis names
            as keys and axis values as values. Axis values may include units where defined.
        """
        value_arrays = [value.value for value in self.target_values.values()]
        units = [value.unit for value in self.target_values.values()]
        grid = np.meshgrid(*value_arrays, indexing="ij")
        combinations = np.vstack(list(map(np.ravel, grid))).T
        grid_points = []
        for combination in combinations:
            grid_point = {
                key: Quantity(combination[i], units[i])
                for i, key in enumerate(self.target_values.keys())
            }

            if "energy" in self.interpolated_limits:
                zenith_idx = np.searchsorted(
                    self.target_values["zenith_angle"].value, grid_point["zenith_angle"].value
                )
                azimuth_idx = np.searchsorted(
                    self.target_values["azimuth"].value, grid_point["azimuth"].value
                )
                nsb_idx = np.searchsorted(self.target_values["nsb"].value, grid_point["nsb"].value)
                energy_lower = self.interpolated_limits["energy"][zenith_idx, azimuth_idx, nsb_idx]
                grid_point["energy_threshold"] = {"lower": energy_lower * u.TeV}

            if "radius" in self.interpolated_limits:
                radius_value = self.interpolated_limits["radius"][zenith_idx, azimuth_idx, nsb_idx]
                grid_point["radius"] = radius_value * u.m

            if "viewcone" in self.interpolated_limits:
                viewcone_value = self.interpolated_limits["viewcone"][
                    zenith_idx, azimuth_idx, nsb_idx
                ]
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

        for point in grid_points:
            cleaned_point = {}
            for key, value in point.items():
                if isinstance(value, dict):
                    # Nested dictionaries
                    cleaned_point[key] = {k: self.serialize_quantity(v) for k, v in value.items()}
                else:
                    cleaned_point[key] = self.serialize_quantity(value)

            cleaned_points.append(cleaned_point)

        output_data = json.dumps(cleaned_points, indent=4)

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(output_data)
            self._logger.info(f"Output saved to {output_file}")
        else:
            self._logger.info(output_data)
        return output_data

    def serialize_quantity(self, value):
        """Serialize Quantity."""
        if isinstance(value, u.Quantity):
            return {"value": value.value, "unit": str(value.unit)}
        self._logger.warning(f"Unsupported type {type(value)} for serialization. Returning as is.")
        return value
