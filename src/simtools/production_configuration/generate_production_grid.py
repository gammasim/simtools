"""
Module defines the `GridGeneration` class.

Used to generate a grid of simulation points based on flexible axes definitions.
The grid can be adapted based on different science cases and data levels. The
axes include parameters such as energy, azimuth, zenith angle, night-sky background,
and camera offset. The class handles various axis scaling, binning,
and distribution types, allowing for adaptable simulation configurations. Additionally,
it provides functionality for converting between Altitude/Azimuth and Right Ascension
/Declination coordinates.

Key Components:
---------------
- `GridGeneration`: Class to handle the generation and adaptation of grid points for simulations.
  - Attributes:
    - `axes` (list of dict): List of dictionaries defining each axis with properties
      such as name, range, binning, scaling, etc.
    - `coordinate_system` (str): The coordinate system being used
    (e.g., 'zenith_azimuth' or 'ra_dec').
    - `observing_location` (EarthLocation): The location of the observation
      (latitude, longitude, height).
    - `observing_time` (Time): The time of the observation.

Example Usage:
--------------
```python
from astropy.coordinates import EarthLocation
from astropy.time import Time
# Define axes for the grid
axes = [
    {"name": "energy", "range": (1e9, 1e14), "binning": 10,
      "scaling": "log", "distribution": "uniform"},
    {"name": "azimuth", "range": (70, 80), "binning": 3,
      "scaling": "linear", "distribution": "uniform"},
    {"name": "zenith_angle", "range": (20, 30), "binning": 3,
      "scaling": "linear", "distribution": "uniform"},
    {"name": "night_sky_background", "range": (5, 6), "binning": 2,
      "scaling": "linear", "distribution": "uniform"},
    {"name": "offset", "range": (0, 10), "binning": 5,
      "scaling": "linear", "distribution": "uniform"},
]


# Define coordinate system
coordinate_system = "ra_dec"

# Define the observing location and time
latitude = 28.7622  # degrees
longitude = -17.8920  # degrees

# Create EarthLocation object
observing_location = EarthLocation(lon=longitude*u.deg, lat=latitude*u.deg, height=2000*u.m)

observing_time = Time('2017-09-16 0:0:0')

# Create grid generation instance
grid_gen = GridGeneration(axes, coordinate_system, observing_location, observing_time)

# Generate grid points
grid_points = grid_gen.generate_grid()

# If coordinate system is 'ra_dec', convert the generated grid points to RA/Dec
grid_points_converted = grid_gen.convert_coordinates(grid_points)

# Example of converting AltAz coordinates to RA/Dec
alt = 45.0  # Altitude in degrees
az = 30.0   # Azimuth in degrees
radec = grid_gen.convert_altaz_to_radec(alt, az)
print(f"RA: {radec.ra.deg} degrees, Dec: {radec.dec.deg} degrees")
"""

import json
import logging

import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from astropy.units import Quantity


class GridGeneration:
    """
    Defines and generates a grid of simulation points based on flexible axes definitions.

    This class generates a grid of points for a simulation based on parameters like energy, azimuth,
    zenith angle, night-sky background, and camera offset. The grid adapts to different science
    cases and data levels, taking into account flexible axis definitions, scaling, and units.

    Attributes
    ----------
    axes : list of dict
        List of dictionaries defining each axis with properties such as name,
          range, binning, scaling, distribution, and unit.
    coordinate_system : str
        The coordinate system being used (e.g., 'zenith_azimuth' or 'declination_azimuth').
    observing_location : EarthLocation
        The location of the observation (latitude, longitude, and height).
    observing_time : Time
        The time of the observation.
    """

    def __init__(
        self,
        axes: list[dict],
        coordinate_system: str = "zenith_azimuth",
        observing_location=None,
        observing_time=None,
        db_config=None,
    ):
        """
        Initialize the grid with the given axes and coordinate system.

        Parameters
        ----------
        axes : list of dict
            List of dictionaries where each dictionary defines an axis with properties
              such as name, range, binning, scaling, distribution type, and unit.
        coordinate_system : str, optional
            The coordinate system for the grid generation (default is 'zenith_azimuth').
        observing_location : EarthLocation, optional
            The location of the observation (latitude, longitude, height).
        observing_time : Time, optional
            The time of the observation.
        db_config : dict, optional
            Database configuration for fetching default limits.
        """
        self._logger = logging.getLogger(__name__)

        self.axes = axes
        self.coordinate_system = coordinate_system
        self.observing_location = (
            observing_location
            if observing_location is not None
            else EarthLocation(lat=0.0 * u.deg, lon=0.0 * u.deg, height=0 * u.m)
        )
        self.observing_time = observing_time if observing_time is not None else Time.now()
        self.db_config = db_config

        # Fetch default limits from the database if db_config is provided
        if self.db_config:
            self._apply_default_limits_from_db()

    def _apply_default_limits_from_db(self):
        """Fetch default limits from the database and apply them to the axes."""
        # Simulate fetching limits from the database
        db_limits = self._fetch_limits_from_db()

        # Update axes with db limits
        for axis in self.axes:
            if axis["name"] == "energy":
                axis["range"] = (db_limits["lower_energy_threshold"], axis["range"][1])
            elif axis["name"] == "viewcone":
                axis["range"] = (0, db_limits["viewcone"])
            elif axis["name"] == "radius":
                axis["range"] = (0, db_limits["upper_radius_threshold"])

    def _fetch_limits_from_db(self):
        """
        Simulate fetching limits from the database.

        Returns
        -------
        dict
            A dictionary containing default limits.
        """
        # Replace this with actual database queries
        return {
            "lower_energy_threshold": 1e9,  # Example: 1 GeV
            "upper_energy_threshold": 1e14,  # Example: 100 TeV
            "viewcone": 10,  # Example: 10 degrees
            "upper_radius_threshold": 5000,  # Example: 5000 meters
        }

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

        for axis in self.axes:
            name = axis["name"]
            axis_range = axis["range"]
            binning = axis["binning"]
            scaling = axis.get("scaling", "linear")
            distribution = axis.get("distribution", "uniform")
            unit = axis.get("unit", None)
            # Create axis values based on scaling
            if scaling == "log":
                values = np.logspace(np.log10(axis_range[0]), np.log10(axis_range[1]), binning)
            else:
                values = np.linspace(axis_range[0], axis_range[1], binning)

            # Apply distribution type
            if distribution == "power-law":
                values = self.generate_power_law_values(axis_range=axis_range, binning=binning)

            if unit:
                values = values * u.Unit(unit)

            axis_values[name] = values

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
