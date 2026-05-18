"""Backend-independent engine for simulation production-grid generation."""

import logging

import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.units import Quantity

from simtools.production_configuration.corsika_limits_lookup import CorsikaLimitsLookup
from simtools.production_configuration.production_grid_helpers import (
    DEFAULT_SERIALIZATION_ROUND_DECIMALS,
)
from simtools.production_configuration.production_grid_serialization import (
    build_grid_metadata,
    build_serialized_rows,
    collect_point_keys,
    serialize_grid_points,
    serialize_grid_value,
)


class ProductionGridEngine:
    """
    Backend-independent engine for simulation production-grid generation.

    This engine handles axis expansion, coordinate-system specific grid
    generation, lookup-table interpolation, coordinate conversion, and ECSV
    serialization helpers used by backend adapters.
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
        Initialize the production-grid engine.

        Parameters
        ----------
        axes : dict
            Dictionary where each key is the axis name and the value is a dictionary
            defining the axis properties (range, binning, scaling, etc.).
        coordinate_system : str, optional
            The coordinate system for the grid generation.
        observing_location : EarthLocation, optional
            The location of the observation.
        observing_time : Time, optional
            The observing time used for RA/Dec transforms.
        lookup_table : str, optional
            Path to the lookup table file (ECSV format).
        telescope_ids : list of str, optional
            Telescope selection used to filter lookup rows.
        simtel_file : str, optional
            Path to a sim_telarray file used to map numeric telescope IDs.
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
        self.simtel_file = simtel_file
        self.interpolated_limits = {}
        self.serialization_round_decimals = DEFAULT_SERIALIZATION_ROUND_DECIMALS
        self._limits_lookup = CorsikaLimitsLookup(
            lookup_table=lookup_table,
            telescope_ids=telescope_ids,
            simtel_file=simtel_file,
        )
        self._simtel_id_to_name = self._limits_lookup.simtel_id_to_name
        self.target_values = self._generate_target_values()

        if self.lookup_table:
            if self.coordinate_system == "ra_dec":
                self._prepare_lookup_table_limits_for_point_interpolation()
            else:
                self._apply_lookup_table_limits()

    @staticmethod
    def _coerce_identifier_container(value):
        """Coerce identifier input into a list."""
        return CorsikaLimitsLookup.coerce_identifier_container(value)

    def _normalize_lookup_identifier(self, identifier):
        """Normalize one telescope identifier and report if it is numeric."""
        return self._limits_lookup.normalize_lookup_identifier(identifier)

    def _normalized_identifier_set(self, identifiers):
        """Return normalized telescope identifiers as a set."""
        return self._limits_lookup.normalized_identifier_set(identifiers)

    def _lookup_contains_numeric_telescope_ids(self, lookup_table):
        """Return True when any lookup-table telescope identifier is numeric."""
        return self._limits_lookup.lookup_contains_numeric_telescope_ids(lookup_table)

    def _sync_limits_lookup(self):
        """Synchronize mutable lookup settings with the shared lookup helper."""
        self._limits_lookup.lookup_table = self.lookup_table
        self._limits_lookup.telescope_ids = self.telescope_ids
        self._limits_lookup.simtel_file = self.simtel_file
        self._limits_lookup.simtel_id_to_name = self._simtel_id_to_name

    def _load_matching_lookup_arrays(self):
        """Load and filter lookup-table arrays for selected telescope IDs."""
        self._sync_limits_lookup()
        return self._limits_lookup.load_matching_lookup_arrays()

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
        self._sync_limits_lookup()
        self.lookup_interpolators_for_point = self._limits_lookup.prepare_point_interpolators()
        self.lookup_points_for_interpolation = self._limits_lookup.lookup_points_for_interpolation
        self.lookup_values_for_interpolation = self._limits_lookup.lookup_values_for_interpolation

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
                values = self.create_circular_binning(axis_range, binning)
            elif scaling == "log":
                values = np.logspace(np.log10(axis_range[0]), np.log10(axis_range[1]), binning)
            elif scaling == "1/cos":
                cos_min = np.cos(np.radians(axis_range[0]))
                cos_max = np.cos(np.radians(axis_range[1]))
                inv_cos_values = np.linspace(1 / cos_min, 1 / cos_max, binning)
                values = np.degrees(np.arccos(1 / inv_cos_values))
            else:
                values = np.linspace(axis_range[0], axis_range[1], binning)

            if units:
                values = values * u.Unit(units)

            target_values[axis_name] = values

        return target_values

    def _apply_lookup_table_limits(self):
        """Apply limits from the lookup table and interpolate values."""
        self._sync_limits_lookup()
        self.interpolated_limits = self._limits_lookup.interpolate_grid_limits(self.target_values)

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

    def _interpolate_limits_for_point(self, zenith, azimuth, nsb):
        """Interpolate lookup-table limits for a single point."""
        self._sync_limits_lookup()
        return self._limits_lookup.interpolate_point(zenith, azimuth, nsb)

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
        """Generate grid points from explicit RA/Dec axes definitions."""
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

    def _generate_grid_radec_mode(self):
        """Generate grid points for RA/Dec mode."""
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
        Create bin centers for azimuth angles, handling circular wraparound.

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
        azimuth_min %= 360
        azimuth_max %= 360

        clockwise_distance = (azimuth_max - azimuth_min) % 360
        counterclockwise_distance = (azimuth_min - azimuth_max) % 360

        if clockwise_distance <= counterclockwise_distance:
            return (
                np.linspace(azimuth_min, azimuth_min + clockwise_distance, num_bins, endpoint=True)
                % 360
            )
        return (
            np.linspace(
                azimuth_min, azimuth_min - counterclockwise_distance, num_bins, endpoint=True
            )
            % 360
        )

    def generate_grid(self):
        """
        Generate the grid based on the required axes and include interpolated limits.

        Returns
        -------
        list of dict
            A list of generated grid points.
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
                grid_point["lower_energy_threshold"] = (
                    self.interpolated_limits["lower_energy_threshold"][
                        zenith_idx, azimuth_idx, nsb_idx
                    ]
                    * u.TeV
                )

            if "upper_scatter_radius" in self.interpolated_limits:
                grid_point["scatter_radius"] = (
                    self.interpolated_limits["upper_scatter_radius"][
                        zenith_idx, azimuth_idx, nsb_idx
                    ]
                    * u.m
                )

            if "viewcone_radius" in self.interpolated_limits:
                grid_point["viewcone_radius"] = (
                    self.interpolated_limits["viewcone_radius"][zenith_idx, azimuth_idx, nsb_idx]
                    * u.deg
                )

            grid_points.append(grid_point)

        return grid_points

    def convert_altaz_to_radec(self, alt, az):
        """
        Convert Altitude/Azimuth (AltAz) coordinates to RA/Dec.

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
        return skycoord.icrs

    def convert_coordinates(self, grid_points):
        """
        Convert the grid points to RA/Dec coordinates if necessary.

        Parameters
        ----------
        grid_points : list of dict
            List of grid points.

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
        serialize_grid_points(
            grid_points=grid_points,
            output_file=output_file,
            coordinate_system=self.coordinate_system,
            observing_time=self.observing_time,
            telescope_ids=self.telescope_ids,
            lookup_table=self.lookup_table,
            serialization_round_decimals=self.serialization_round_decimals,
        )
        self._logger.info(f"Output saved to {output_file}")

    @staticmethod
    def _collect_point_keys(grid_points):
        """Collect all grid-point keys while preserving first-seen order."""
        return collect_point_keys(grid_points)

    def _serialize_grid_value(self, value):
        """Serialize one grid value and return (value, unit)."""
        return serialize_grid_value(
            value,
            serialization_round_decimals=self.serialization_round_decimals,
        )

    def _build_serialized_rows(self, grid_points, all_keys):
        """Build serialized row dictionaries and collect units."""
        return build_serialized_rows(
            grid_points,
            all_keys,
            serialization_round_decimals=self.serialization_round_decimals,
        )

    def _build_grid_metadata(self):
        """Build metadata for the output grid table."""
        return build_grid_metadata(
            coordinate_system=self.coordinate_system,
            observing_time=self.observing_time,
            telescope_ids=self.telescope_ids,
            lookup_table=self.lookup_table,
        )

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
