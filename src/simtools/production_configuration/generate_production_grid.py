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

    This class generates a grid of points for a simulation based on parameters like energy, azimuth,
    zenith angle, night-sky background, and camera offset, taking into account axis definitions,
      scaling, and units.
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
        azimuths = np.array([row["azimuth"] for row in matching_rows])
        lower_energy_thresholds = np.array([row["lower_energy_threshold"] for row in matching_rows])
        upper_radius_thresholds = np.array([row["upper_radius_threshold"] for row in matching_rows])
        viewcone_radii = np.array([row["viewcone_radius"] for row in matching_rows])

        if "energy" in self.axes:
            interpolated_lower = self._interpolate_limits(
                zeniths, azimuths, lower_energy_thresholds, self.axes["energy"]["range"]
            )
            self.axes["energy"]["range"] = (interpolated_lower, self.axes["energy"]["range"][1])

        if "radius" in self.axes:
            interpolated_upper = self._interpolate_limits(
                zeniths, azimuths, upper_radius_thresholds, self.axes["radius"]["range"]
            )
            self.axes["radius"]["range"] = (0, interpolated_upper)

        if "viewcone" in self.axes:
            interpolated_viewcone = self._interpolate_limits(
                zeniths, azimuths, viewcone_radii, self.axes["viewcone"]["range"]
            )
            self.axes["viewcone"]["range"] = (0, interpolated_viewcone)

    def _interpolate_limits(self, zeniths, azimuths, values, axis_range):
        """
        Interpolate limits for a given axis based on zenith and azimuth.

        Parameters
        ----------
        zeniths : np.ndarray
            Array of zenith values from the lookup table.
        azimuths : np.ndarray
            Array of azimuth values from the lookup table.
        values : np.ndarray
            Array of limit values corresponding to the zenith and azimuth.
        axis_range : tuple
            The current range of the axis.

        Returns
        -------
        float
            The interpolated limit value.
        """
        # Create a grid of zenith and azimuth values
        points = np.column_stack((zeniths, azimuths))
        grid_point = np.array([[axis_range[0], axis_range[1]]])

        # Interpolate the value at the grid point
        interpolated_value = griddata(points, values, grid_point, method="linear")

        # If interpolation fails, fallback to the closest value
        if np.isnan(interpolated_value):
            distances = np.sqrt((zeniths - axis_range[0]) ** 2 + (azimuths - axis_range[1]) ** 2)
            closest_index = np.argmin(distances)
            interpolated_value = values[closest_index]

        return interpolated_value

    def generate_grid(self) -> list[dict]:
        """
        Generate the grid based on the defined axes.

        Returns
        -------
        list of dict
            A list of grid points, each represented as a dictionary with axis names
              as keys and axis values as values. Axis values may include units where defined.
        """
        axis_values = {}

        for axis_name, axis in self.axes.items():
            print("axis_name", axis_name)
            print("axis", axis)
            axis_range = axis["range"]
            binning = axis["binning"]
            scaling = axis.get("scaling", "linear")
            distribution = axis.get("distribution", "uniform")
            units = axis.get("units", None)

            # Create axis values based on scaling
            if scaling == "log":
                values = np.logspace(np.log10(axis_range[0]), np.log10(axis_range[1]), binning)
            else:
                values = np.linspace(axis_range[0], axis_range[1], binning)

            # Apply distribution type
            if distribution == "power-law":
                values = self.generate_power_law_values(axis_range=axis_range, binning=binning)

            if units:
                values = values * u.Unit(units)

            axis_values[axis_name] = values

        value_arrays = [value.value for value in axis_values.values()]
        units = [value.unit for value in axis_values.values()]

        grid = np.meshgrid(*value_arrays, indexing="ij")
        combinations = np.vstack(list(map(np.ravel, grid))).T
        return [
            {key: Quantity(combination[i], units[i]) for i, key in enumerate(axis_values.keys())}
            for combination in combinations
        ]

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
