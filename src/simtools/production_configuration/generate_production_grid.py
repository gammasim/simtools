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
import os
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.table import Table
from astropy.units import Quantity
from astropy.utils import iers
from scipy.interpolate import LinearNDInterpolator, griddata
from scipy.spatial import QhullError  # pylint: disable=no-name-in-module

from simtools.simtel.simtel_io_metadata import (
    get_sim_telarray_telescope_id_to_telescope_name_mapping,
)

DEFAULT_SERIALIZATION_ROUND_DECIMALS = 6


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
        axes,
        coordinate_system="zenith_azimuth",
        observing_location=None,
        observing_time=None,
        lookup_table=None,
        telescope_ids=None,
        simtel_file=None,
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
        telescope_ids : list of str, optional
            List of telescope IDs to get the limits for.
        simtel_file : str, optional
            Path to a sim_telarray file used to map sim_telarray telescope IDs to
            telescope names when matching lookup-table telescope selections.
        """
        self._logger = logging.getLogger(__name__)
        if os.getenv("SIMTOOLS_OFFLINE_IERS", "0") == "1":
            iers.conf.auto_download = False
            iers.conf.auto_max_age = None

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
        self.simtel_file = simtel_file
        self.interpolated_limits = {}
        self.serialization_round_decimals = DEFAULT_SERIALIZATION_ROUND_DECIMALS
        self._simtel_id_to_name = (
            get_sim_telarray_telescope_id_to_telescope_name_mapping(simtel_file)
            if simtel_file
            else {}
        )

        # Store target values for each axis
        self.target_values = self._generate_target_values()

        if self.lookup_table:
            if self.coordinate_system == "ra_dec":
                self._prepare_lookup_table_limits_for_point_interpolation()
            else:
                self._apply_lookup_table_limits()

    @staticmethod
    def _coerce_identifier_container(value):
        """Coerce identifier input into a list."""
        if value is None:
            return []
        if isinstance(value, str):
            stripped = value.strip()
            return json.loads(stripped) if stripped.startswith("[") else [stripped]
        if isinstance(value, (list, tuple, set)):
            return list(value)
        return [value]

    def _normalize_lookup_identifier(self, identifier):
        """Normalize one telescope identifier and report if it is numeric."""
        if isinstance(identifier, (int, np.integer)):
            return self._simtel_id_to_name.get(int(identifier), str(int(identifier))), True

        text = str(identifier).strip()
        if text.lstrip("+-").isdigit():
            return self._simtel_id_to_name.get(int(text), text), True
        return text, False

    def _normalized_identifier_set(self, identifiers):
        """Return normalized telescope identifiers as a set."""
        return {
            self._normalize_lookup_identifier(identifier)[0]
            for identifier in self._coerce_identifier_container(identifiers)
        }

    def _lookup_contains_numeric_telescope_ids(self, lookup_table):
        """Return True when any lookup-table telescope identifier is numeric."""
        return any(
            any(
                self._normalize_lookup_identifier(identifier)[1]
                for identifier in self._coerce_identifier_container(row["telescope_ids"])
            )
            for row in lookup_table
        )

    def _load_matching_lookup_arrays(self):
        """Load and filter lookup-table arrays for selected telescope IDs."""
        lookup_table = Table.read(self.lookup_table, format="ascii.ecsv")
        selected_telescope_ids = self._normalized_identifier_set(self.telescope_ids)

        matching_rows = [
            row
            for row in lookup_table
            if selected_telescope_ids == self._normalized_identifier_set(row["telescope_ids"])
        ]

        if not matching_rows:
            has_numeric_lookup_ids = self._lookup_contains_numeric_telescope_ids(lookup_table)

            if has_numeric_lookup_ids and not self.simtel_file:
                raise ValueError(
                    "Lookup table telescope selections contain numeric IDs. "
                    "Provide --simtel_file to map those IDs to telescope names."
                )

            raise ValueError(
                f"No matching rows in the lookup table for telescope_ids: {self.telescope_ids}"
            )

        def extract_array(field, transform=lambda x: x):
            return np.array([transform(row[field]) for row in matching_rows])

        zeniths = extract_array("zenith")
        azimuths = extract_array("azimuth", lambda x: x % 360)
        nsb_values = extract_array("nsb_level", float)

        return {
            "points": np.column_stack((zeniths, azimuths, nsb_values)),
            "lower_energy_threshold": extract_array("lower_energy_limit"),
            "upper_scatter_radius": extract_array("upper_radius_limit"),
            "viewcone_radius": extract_array("viewcone_radius"),
        }

    def _require_observing_time(self):
        """Return observing time if available, else raise a clear error."""
        if self.observing_time is None:
            raise ValueError("Observing time is required for ra_dec grid generation.")
        return self.observing_time

    def _get_max_zenith_for_radec_mode(self):
        """Read maximum zenith from axes for RA/Dec direction sampling."""
        zenith_axis = self.axes.get("zenith_angle")
        if not zenith_axis or "range" not in zenith_axis or len(zenith_axis["range"]) != 2:
            raise ValueError(
                "RA/Dec direction sampling requires 'zenith_angle' axis with a valid "
                "two-element 'range' in the axes definition."
            )
        return float(zenith_axis["range"][1])

    def _prepare_lookup_table_limits_for_point_interpolation(self):
        """Prepare lookup arrays for per-point interpolation in RA/Dec grid mode."""
        lookup_arrays = self._load_matching_lookup_arrays()

        points = lookup_arrays["points"]
        azimuths = points[:, 1]
        azimuths_wrapped = np.concatenate([azimuths + shift for shift in (0, 360, -360)])

        def repeat_3(arr):
            """Repeat an array three times."""
            return np.tile(arr, 3)

        self.lookup_points_for_interpolation = np.column_stack(
            (
                repeat_3(points[:, 0]),
                azimuths_wrapped,
                repeat_3(points[:, 2]),
            )
        )
        self.lookup_values_for_interpolation = {
            "lower_energy_threshold": repeat_3(lookup_arrays["lower_energy_threshold"]),
            "upper_scatter_radius": repeat_3(lookup_arrays["upper_scatter_radius"]),
            "viewcone_radius": repeat_3(lookup_arrays["viewcone_radius"]),
        }
        try:
            self.lookup_interpolators_for_point = {
                key: LinearNDInterpolator(
                    self.lookup_points_for_interpolation,
                    self.lookup_values_for_interpolation[key],
                    fill_value=np.nan,
                )
                for key in ("lower_energy_threshold", "upper_scatter_radius", "viewcone_radius")
            }
        except QhullError as exc:
            raise ValueError(
                "Lookup table does not contain enough unique points for 3D interpolation "
                "(zenith, azimuth, nsb_level). Provide a denser lookup table "
                "or run without --lookup_table if limits are not required for this use case."
            ) from exc

    def _has_radec_axes(self):
        """Return True if axes define a native RA/Dec grid."""
        return "ra" in self.axes and "dec" in self.axes

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
        lookup_arrays = self._load_matching_lookup_arrays()
        points_base = lookup_arrays["points"]
        lower_energy_thresholds = lookup_arrays["lower_energy_threshold"]
        upper_scatter_radii = lookup_arrays["upper_scatter_radius"]
        viewcone_radii = lookup_arrays["viewcone_radius"]

        zeniths = points_base[:, 0]
        azimuths = points_base[:, 1]
        nsb_values = points_base[:, 2]

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
                    self.target_values["nsb_level"].value,
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
                len(self.target_values["nsb_level"]),
            )

        self.interpolated_limits = {
            "lower_energy_threshold": interpolate(lower_energy_thresholds),
            "upper_scatter_radius": interpolate(upper_scatter_radii),
            "viewcone_radius": interpolate(viewcone_radii),
        }

    def _generate_radec_grid_direction_points(self):
        """Generate direction points from declination lines and hour-angle spacing."""
        observing_time = self._require_observing_time()
        max_zenith = self._get_max_zenith_for_radec_mode()
        lst_deg = observing_time.sidereal_time(
            "apparent", longitude=self.observing_location.lon
        ).deg

        direction_points = []
        for declination in np.arange(-90.0, 91.0, 1.0):
            cos_dec = np.cos(np.deg2rad(declination))
            step_ha = 1.0 / cos_dec if cos_dec > 1e-6 else 360.0
            n_ha = max(1, int(np.ceil(360.0 / step_ha)))
            hour_angles = np.linspace(-180.0, 180.0, n_ha, endpoint=False)
            ra_values = (lst_deg - hour_angles) % 360.0

            skycoord = SkyCoord(
                ra=ra_values * u.deg,
                dec=np.full_like(ra_values, declination) * u.deg,
                frame="icrs",
            )
            altaz = skycoord.transform_to(
                AltAz(location=self.observing_location, obstime=observing_time)
            )

            zenith_values = (90.0 * u.deg - altaz.alt).to(u.deg).value
            mask = (zenith_values >= 0.0) & (zenith_values <= max_zenith)

            for idx in np.nonzero(mask)[0]:
                direction_points.append(
                    {
                        "zenith_angle": zenith_values[idx] * u.deg,
                        "azimuth": altaz.az.deg[idx] * u.deg,
                    }
                )
        return direction_points

    def _generate_extra_axis_combinations(self, excluded_keys):
        """Generate combinations for all axes except the excluded ones."""
        extra_axes = {
            key: value for key, value in self.target_values.items() if key not in excluded_keys
        }
        if not extra_axes:
            return list(extra_axes.keys()), [], [np.array([])]

        extra_value_arrays = [value.value for value in extra_axes.values()]
        extra_units = [value.unit for value in extra_axes.values()]
        extra_grid = np.meshgrid(*extra_value_arrays, indexing="ij")
        extra_combinations = np.vstack(list(map(np.ravel, extra_grid))).T
        return list(extra_axes.keys()), extra_units, extra_combinations

    def _add_lookup_limits_to_point(self, point, zenith, azimuth):
        """Interpolate and attach lookup-table limits to a grid point."""
        if not self.lookup_table:
            return

        nsb_value = point.get("nsb_level", 1)
        if isinstance(nsb_value, Quantity):
            nsb_value = nsb_value.value
        limits = self._interpolate_limits_for_point(
            zenith=zenith,
            azimuth=azimuth,
            nsb=float(nsb_value),
        )
        point["lower_energy_threshold"] = limits["lower_energy_threshold"] * u.TeV
        point["scatter_radius"] = limits["upper_scatter_radius"] * u.m
        point["viewcone_radius"] = limits["viewcone_radius"] * u.deg

    def _generate_grid_from_radec_axes(self):
        """Generate grid points from explicit RA/Dec axes definitions.

        All explicit RA/Dec combinations defined by the input axes are preserved,
        even when their transformed Alt/Az coordinates fall below the local horizon.
        """
        observing_time = self._require_observing_time()

        axis_keys = [key for key in self.target_values if key not in ("zenith_angle", "azimuth")]
        value_arrays = [self.target_values[key].value for key in axis_keys]
        units = [self.target_values[key].unit for key in axis_keys]
        grid = np.meshgrid(*value_arrays, indexing="ij")
        combinations = np.vstack(list(map(np.ravel, grid))).T

        grid_points = []
        for combination in combinations:
            grid_point = {
                key: Quantity(combination[i], units[i]) for i, key in enumerate(axis_keys)
            }

            skycoord = SkyCoord(
                ra=grid_point["ra"].to(u.deg),
                dec=grid_point["dec"].to(u.deg),
                frame="icrs",
            )
            altaz = skycoord.transform_to(
                AltAz(location=self.observing_location, obstime=observing_time)
            )
            zenith = (90.0 * u.deg - altaz.alt).to(u.deg).value

            self._add_lookup_limits_to_point(grid_point, zenith=zenith, azimuth=altaz.az.deg)

            grid_points.append(grid_point)

        return grid_points

    def _interpolate_limits_for_point(self, zenith, azimuth, nsb):
        """Interpolate lookup-table limits for a single point."""
        target = np.array([[zenith, azimuth % 360.0, nsb]], dtype=float)
        return {
            "lower_energy_threshold": float(
                self.lookup_interpolators_for_point["lower_energy_threshold"](target)[0]
            ),
            "upper_scatter_radius": float(
                self.lookup_interpolators_for_point["upper_scatter_radius"](target)[0]
            ),
            "viewcone_radius": float(
                self.lookup_interpolators_for_point["viewcone_radius"](target)[0]
            ),
        }

    def _generate_grid_radec_mode(self):
        """Generate grid points for RA/Dec mode.

        If explicit ``ra``/``dec`` axes are provided, use those points directly.
        Otherwise, sample directions along declination lines with hour-angle spacing
        and keep only points that satisfy the configured zenith-angle limits.
        """
        if self._has_radec_axes():
            return self._generate_grid_from_radec_axes()

        direction_points = self._generate_radec_grid_direction_points()
        extra_keys, extra_units, extra_combinations = self._generate_extra_axis_combinations(
            excluded_keys=("zenith_angle", "azimuth")
        )

        grid_points = []
        for direction_point in direction_points:
            for extra_combination in extra_combinations:
                point = dict(direction_point)
                for i, key in enumerate(extra_keys):
                    point[key] = Quantity(extra_combination[i], extra_units[i])

                self._add_lookup_limits_to_point(
                    point,
                    zenith=point["zenith_angle"].value,
                    azimuth=point["azimuth"].value,
                )

                grid_points.append(point)

        return grid_points

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

    def generate_grid(self):
        """
        Generate the grid based on the required axes and include interpolated limits.

        Takes energy threshold, viewcone, and radius from the interpolated lookup table.

        Returns
        -------
        list of dict
            A list of grid points, each represented as a dictionary with axis names
            as keys and axis values as values. Axis values may include units where defined.
        """
        if self.coordinate_system == "ra_dec":
            return self._generate_grid_radec_mode()

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

            if "lower_energy_threshold" in self.interpolated_limits:
                zenith_idx = np.searchsorted(
                    self.target_values["zenith_angle"].value, grid_point["zenith_angle"].value
                )
                azimuth_idx = np.searchsorted(
                    self.target_values["azimuth"].value, grid_point["azimuth"].value
                )
                nsb_idx = np.searchsorted(
                    self.target_values["nsb_level"].value,
                    grid_point["nsb_level"].value,
                )
                energy_lower = self.interpolated_limits["lower_energy_threshold"][
                    zenith_idx, azimuth_idx, nsb_idx
                ]
                grid_point["lower_energy_threshold"] = energy_lower * u.TeV

            if "upper_scatter_radius" in self.interpolated_limits:
                radius_value = self.interpolated_limits["upper_scatter_radius"][
                    zenith_idx, azimuth_idx, nsb_idx
                ]
                grid_point["scatter_radius"] = radius_value * u.m

            if "viewcone_radius" in self.interpolated_limits:
                viewcone_value = self.interpolated_limits["viewcone_radius"][
                    zenith_idx, azimuth_idx, nsb_idx
                ]
                grid_point["viewcone_radius"] = viewcone_value * u.deg

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

    def convert_coordinates(self, grid_points):
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
        """Serialize the grid output and save to an ECSV table file."""
        if Path(output_file).suffix.lower() != ".ecsv":
            raise ValueError("Grid output file must use '.ecsv' extension.")

        all_keys = self._collect_point_keys(grid_points)
        rows, units = self._build_serialized_rows(grid_points, all_keys)

        output_table = Table(rows=rows, names=all_keys)
        for column_name, unit in units.items():
            output_table[column_name].unit = u.Unit(unit)

        output_table.meta = self._build_grid_metadata()

        output_table.write(output_file, format="ascii.ecsv", overwrite=True)
        self._logger.info(f"Output saved to {output_file}")

    @staticmethod
    def _collect_point_keys(grid_points):
        """Collect all grid-point keys while preserving first-seen order."""
        all_keys = []
        for point in grid_points:
            for key in point:
                if key not in all_keys:
                    all_keys.append(key)
        return all_keys

    def _serialize_grid_value(self, value):
        """Serialize one grid value and return (value, unit)."""
        if isinstance(value, u.Quantity):
            serialized = round(float(value.value), self.serialization_round_decimals)
            return serialized, str(value.unit)

        if isinstance(value, dict) and "value" in value:
            return value["value"], value.get("unit")

        if value is None:
            return np.nan, None

        if isinstance(value, (np.floating, float)):
            return round(float(value), self.serialization_round_decimals), None

        if isinstance(value, (np.integer, int)):
            return int(value), None

        return value, None

    def _build_serialized_rows(self, grid_points, all_keys):
        """Build serialized row dictionaries and collect units."""
        rows = []
        units = {}

        for point in grid_points:
            row = {}
            for key in all_keys:
                serialized_value, unit = self._serialize_grid_value(point.get(key))
                row[key] = serialized_value
                if unit is not None:
                    units.setdefault(key, unit)
            rows.append(row)

        return rows, units

    def _build_grid_metadata(self):
        """Build metadata for the output grid table."""
        return {
            "coordinate_system": self.coordinate_system,
            "reference_frame": "ICRS (J2000)",
            "observing_time_utc": self.observing_time.isot if self.observing_time else None,
            "observing_time_scale": self.observing_time.scale if self.observing_time else None,
            "telescope_ids": self.telescope_ids,
            "lookup_table": str(Path(self.lookup_table)) if self.lookup_table else None,
        }

    def serialize_quantity(self, value):
        """Serialize Quantity."""
        if isinstance(value, u.Quantity):
            serialized_value = float(value.value)
            rounded_value = round(serialized_value, self.serialization_round_decimals)
            return {"value": rounded_value, "unit": str(value.unit)}
        if isinstance(value, float):
            return round(value, self.serialization_round_decimals)
        if isinstance(value, np.floating):
            return round(float(value), self.serialization_round_decimals)
        if isinstance(value, np.integer):
            return int(value)
        return value
