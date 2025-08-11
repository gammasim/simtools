"""
Module defines the `GridGeneration` class.

Used to generate a grid of simulation points based on flexible axes definitions such
azimuth, zenith angle, night-sky background, and camera offset.
The module handles axis binning, scaling and interpolation of energy thresholds, viewcone,
and radius limits from a lookup table.
Additionally, it allows for converting between Altitude/Azimuth and Right Ascension
Declination coordinates. The resulting grid points are saved to a file.
"""

import logging

import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.table import Table
from astropy.units import Quantity
from scipy.interpolate import griddata

from simtools.io import ascii_handler


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
            The time of the observation. If None, coordinate conversion to RA/Dec not working.
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
        self.observing_time = observing_time
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

        def extract_array(field, transform=lambda x: x):
            return np.array([transform(row[field]) for row in matching_rows])

        zeniths = extract_array("zenith")
        azimuths = extract_array("azimuth", lambda x: x % 360)
        nsb_values = extract_array("nsb", lambda x: 1 if x == "dark" else 5)
        lower_energy_thresholds = extract_array("lower_energy_threshold")
        upper_radius_thresholds = extract_array("upper_radius_threshold")
        viewcone_radii = extract_array("viewcone_radius")

        # Wrap azimuths and repeat others
        azimuths_wrapped = np.concatenate([azimuths + shift for shift in (0, 360, -360)])

        def repeat_3(arr):
            """Repeat an array three times."""
            return np.tile(arr, 3)

        points = np.column_stack(
            (
                repeat_3(zeniths),
                azimuths_wrapped,
                repeat_3(nsb_values),
            )
        )

        target_grid = (
            np.array(
                np.meshgrid(
                    self.target_values["zenith_angle"].value,
                    self.target_values["azimuth"].value,
                    self.target_values["nsb"].value,
                    indexing="ij",
                )
            )
            .reshape(3, -1)
            .T
        )

        def interpolate(values):
            return griddata(
                points, repeat_3(values), target_grid, method="linear", fill_value=np.nan
            ).reshape(
                len(self.target_values["zenith_angle"]),
                len(self.target_values["azimuth"]),
                len(self.target_values["nsb"]),
            )

        self.interpolated_limits = {
            "energy": interpolate(lower_energy_thresholds),
            "radius": interpolate(upper_radius_thresholds),
            "viewcone": interpolate(viewcone_radii),
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

        Raises
        ------
        ValueError
            If observing_time is not set.
        """
        if self.observing_time is None:
            raise ValueError(
                "Observing time is not set. "
                "Please provide an observing_time to convert coordinates."
            )

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

    def serialize_grid_points(self, grid_points, output_file):
        """Serialize the grid output and save to a file or print to the console."""
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

        ascii_handler.write_data_to_file(
            data=cleaned_points,
            output_file=output_file,
            sort_keys=False,
        )
        self._logger.info(f"Output saved to {output_file}")

    def serialize_quantity(self, value):
        """Serialize Quantity."""
        if isinstance(value, u.Quantity):
            return {"value": value.value, "unit": str(value.unit)}
        self._logger.warning(f"Unsupported type {type(value)} for serialization. Returning as is.")
        return value
