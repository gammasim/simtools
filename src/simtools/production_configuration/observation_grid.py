"""
Generate observation sampling grids with direction-dependent CORSIKA physics limits.

Samples observing positions (Alt/Az or RA/Dec) and interpolates physics limits
(minimum energy, maximum scatter radius, maximum viewcone) from lookup tables.
Supports linear/log/1/cos binning modes.
"""

import logging

import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.units import Quantity

from simtools.production_configuration.corsika_limits_lookup import (
    CorsikaLimitsLookup,
    attach_lookup_limits_to_point,
)


class ProductionGridEngine:
    """
    Generate observation grids with direction-dependent CORSIKA physics limits.

    Samples observing positions per configured axes, interpolates physics limits from
    lookup tables, and optionally converts between coordinate systems (Alt/Az vs RA/Dec).
    Results feed into :func:`production_configuration.simulation_jobs.build_simulation_jobs`
    to expand into full job matrices.
    """

    def __init__(
        self,
        axes,
        coordinate_system="horizontal",
        observing_location=None,
        time_of_observation=None,
        lookup_table=None,
        array_layout_name=None,
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
            The location of the observation (latitude, longitude, height).
        time_of_observation : Time, optional
            The time of observation required for RA/Dec transforms.
        lookup_table : str, optional
            Path to the lookup table file (ECSV format).
        array_layout_name : str, optional
            Array layout name used to select lookup-table rows.
        """
        self._logger = logging.getLogger(__name__)
        self.axes = axes["axes"] if "axes" in axes else axes
        self.coordinate_system = coordinate_system
        self.observing_location = (
            observing_location
            if observing_location is not None
            else EarthLocation(lat=0.0 * u.deg, lon=0.0 * u.deg, height=0 * u.m)
        )
        self.time_of_observation = time_of_observation
        self.lookup_table = lookup_table
        self.array_layout_name = array_layout_name
        self.interpolated_limits = {}
        self._limits_lookup = None
        if self.lookup_table:
            self._limits_lookup = CorsikaLimitsLookup(
                lookup_table=lookup_table,
                array_layout_name=array_layout_name,
            )
        self.target_values = self._generate_target_values()

        if self.lookup_table:
            if self.coordinate_system == "ra_dec":
                self._prepare_lookup_table_limits_for_point_interpolation()
            else:
                self._apply_lookup_table_limits()

    def _require_time_of_observation(self):
        """Return observing time if available, else raise a clear error."""
        if self.time_of_observation is None:
            raise ValueError("Observing time is required for ra_dec grid generation.")
        return self.time_of_observation

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
        self._limits_lookup.prepare_point_interpolators()

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
        self.interpolated_limits = self._limits_lookup.interpolate_grid_limits(self.target_values)

    def _generate_radec_grid_direction_points(self):
        """Generate direction points from declination lines and hour-angle spacing."""
        time_of_observation = self._require_time_of_observation()
        max_zenith = self._get_max_zenith_for_radec_mode()
        lst_deg = time_of_observation.sidereal_time(
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
                AltAz(location=self.observing_location, obstime=time_of_observation)
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
        attach_lookup_limits_to_point(point, limits)

    def _generate_grid_from_radec_axes(self, include_horizontal_coordinates=False):
        """Generate grid points from explicit RA/Dec axes definitions."""
        if self._is_adaptive_radec_density_enabled():
            return self._generate_adaptive_radec_grid(include_horizontal_coordinates)

        time_of_observation = self._require_time_of_observation()

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
                AltAz(location=self.observing_location, obstime=time_of_observation)
            )
            zenith = (90.0 * u.deg - altaz.alt).to(u.deg)
            azimuth = altaz.az.to(u.deg)

            if include_horizontal_coordinates:
                grid_point["zenith_angle"] = zenith
                grid_point["azimuth"] = azimuth

            self._add_lookup_limits_to_point(
                grid_point,
                zenith=zenith.value,
                azimuth=azimuth.value,
            )
            grid_points.append(grid_point)

        return grid_points

    def _generate_grid_radec_mode(self, include_horizontal_coordinates=False):
        """Generate grid points for RA/Dec mode."""
        if "ra" in self.axes and "dec" in self.axes:
            return self._generate_grid_from_radec_axes(
                include_horizontal_coordinates=include_horizontal_coordinates
            )

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

        return self.convert_coordinates(
            grid_points,
            keep_horizontal_coordinates=include_horizontal_coordinates,
        )

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
        raw_span = abs(float(azimuth_range[1]) - float(azimuth_range[0]))
        if raw_span > 0.0 and np.isclose(raw_span % 360.0, 0.0):
            azimuth_start = float(azimuth_range[0]) % 360.0
            return (azimuth_start + np.linspace(0.0, 360.0, num_bins, endpoint=False)) % 360.0

        azimuth_min, azimuth_max = azimuth_range
        azimuth_start = float(azimuth_min) % 360.0
        directed_distance = float((azimuth_max - azimuth_min) % 360.0)

        return (
            np.linspace(
                azimuth_start,
                azimuth_start + directed_distance,
                num_bins,
                endpoint=True,
            )
            % 360.0
        )

    @staticmethod
    def _ceil_with_tolerance(value):
        """Ceil a float while avoiding near-integer floating-point artifacts."""
        nearest_integer = round(value)
        if np.isclose(value, nearest_integer):
            return int(nearest_integer)
        return int(np.ceil(value))

    @staticmethod
    def _directed_circular_span_degrees(azimuth_range):
        """Return directed circular span (degrees) from start to end."""
        raw_span = abs(float(azimuth_range[1]) - float(azimuth_range[0]))
        if raw_span > 0.0 and np.isclose(raw_span % 360.0, 0.0):
            return 360.0
        return float((azimuth_range[1] - azimuth_range[0]) % 360.0)

    @staticmethod
    def _is_in_directed_azimuth_range(azimuth_values_deg, azimuth_range):
        """Return mask of azimuth values inside directed circular range [start -> end]."""
        start, end = azimuth_range
        raw_span = abs(float(end) - float(start))
        if raw_span > 0.0 and np.isclose(raw_span % 360.0, 0.0):
            return np.ones_like(azimuth_values_deg, dtype=bool)

        start_norm = float(start) % 360.0
        span = float((end - start) % 360.0)
        offsets = (np.asarray(azimuth_values_deg) - start_norm) % 360.0
        return offsets <= (span + 1e-12)

    def _is_adaptive_radec_density_enabled(self):
        """Return whether RA/Dec grid generation should use per-declination adaptive RA bins."""
        return (
            self.coordinate_system == "ra_dec"
            and "ra" in self.axes
            and "dec" in self.axes
            and self.axes["ra"].get("direction_grid_density") is not None
        )

    def _compute_visible_radec_strip(self, dec_deg, spacing, lst_deg, time_of_observation):
        """Compute visible RA, zenith, and azimuth samples for one declination strip."""
        cos_dec = np.cos(np.deg2rad(dec_deg))
        if cos_dec <= 0.0:
            return None

        ha_step = spacing / cos_dec
        ha_values = np.arange(-180.0, 180.0, ha_step)
        if len(ha_values) == 0:
            ha_values = np.array([0.0])

        ra_values = ((lst_deg - ha_values) % 360.0) * u.deg
        skycoord = SkyCoord(
            ra=ra_values.to(u.deg),
            dec=np.full(len(ra_values), dec_deg) * u.deg,
            frame="icrs",
        )
        altaz = skycoord.transform_to(
            AltAz(location=self.observing_location, obstime=time_of_observation)
        )

        visible_mask = altaz.alt.to_value(u.deg) >= 0.0

        zenith_values = (90.0 * u.deg - altaz.alt).to(u.deg)
        azimuth_values = altaz.az.to(u.deg)

        ra_axis = self.axes.get("ra", {})

        zenith_range = ra_axis.get("local_zenith_range")
        if zenith_range is not None:
            zenith_min, zenith_max = sorted(
                (
                    float(zenith_range[0]),
                    float(zenith_range[1]),
                )
            )
            zenith_values_deg = zenith_values.to_value(u.deg)
            visible_mask &= (zenith_values_deg >= zenith_min) & (zenith_values_deg <= zenith_max)

        azimuth_range = ra_axis.get("local_azimuth_range")
        if azimuth_range is not None:
            azimuth_values_deg = azimuth_values.to_value(u.deg)
            visible_mask &= self._is_in_directed_azimuth_range(
                azimuth_values_deg,
                azimuth_range,
            )

        if not np.any(visible_mask):
            return None

        return (
            dec_deg * u.deg,
            ra_values[visible_mask],
            zenith_values[visible_mask],
            azimuth_values[visible_mask],
        )

    def _generate_adaptive_radec_grid(self, include_horizontal_coordinates=False):
        """Generate RA/Dec grid with Hour-Angle stepping adapted per declination strip.

        Pointings are arranged along declination lines evenly spaced in declination.
        For each strip the telescope pointings are stepped through Hour Angle (equivalent
        to stepping through time), tracing the actual arc a source at that declination
        sweeps across the local sky. Hour-angle spacing is scaled by 1 / cos(dec)
        to maintain uniform density per solid angle.

        Parameters
        ----------
        include_horizontal_coordinates : bool, optional
            If True, include zenith_angle and azimuth in each grid point.

        Returns
        -------
        list of dict
            Grid points with ra, dec and optionally zenith_angle, azimuth.
        """
        time_of_observation = self._require_time_of_observation()
        density = float(self.axes["ra"]["direction_grid_density"])
        spacing = 1.0 / np.sqrt(density)
        dec_min, dec_max = sorted(
            (float(self.axes["dec"]["range"][0]), float(self.axes["dec"]["range"][1]))
        )
        dec_values_deg = np.arange(dec_min, dec_max + 0.5 * spacing, spacing)

        lst_deg = time_of_observation.sidereal_time(
            "apparent", longitude=self.observing_location.lon
        ).deg

        extra_keys, extra_units, extra_combinations = self._generate_extra_axis_combinations(
            excluded_keys=("ra", "dec")
        )

        grid_points = []
        for dec_deg in dec_values_deg:
            strip = self._compute_visible_radec_strip(
                dec_deg=dec_deg,
                spacing=spacing,
                lst_deg=lst_deg,
                time_of_observation=time_of_observation,
            )
            if strip is None:
                continue
            dec, visible_ra_values, visible_zenith_values, visible_azimuth_values = strip

            for extra_combination in extra_combinations:
                point_base = {
                    key: Quantity(extra_combination[i], extra_units[i])
                    for i, key in enumerate(extra_keys)
                }
                for ra, zenith, azimuth in zip(
                    visible_ra_values,
                    visible_zenith_values,
                    visible_azimuth_values,
                    strict=True,
                ):
                    point = {**point_base, "ra": ra, "dec": dec}

                    if include_horizontal_coordinates:
                        point["zenith_angle"] = zenith
                        point["azimuth"] = azimuth

                    self._add_lookup_limits_to_point(
                        point,
                        zenith=zenith.value,
                        azimuth=azimuth.value,
                    )
                    grid_points.append(point)

        return grid_points

    def _is_adaptive_horizontal_density_enabled(self):
        """Return whether horizontal grid generation should use row-wise adaptive azimuth bins."""
        return (
            self.coordinate_system == "horizontal"
            and "azimuth" in self.axes
            and "zenith_angle" in self.axes
            and self.axes["azimuth"].get("direction_grid_density") is not None
        )

    def _generate_adaptive_horizontal_grid(self):
        """Generate horizontal grid with azimuth binning adapted per zenith row."""
        density = float(self.axes["azimuth"]["direction_grid_density"])
        azimuth_range = self.axes["azimuth"]["range"]
        azimuth_span = self._directed_circular_span_degrees(azimuth_range)
        zenith_values = self.target_values["zenith_angle"]

        zenith_step = 1.0 / np.sqrt(density)
        if len(zenith_values) > 1:
            zenith_step = float(np.mean(np.abs(np.diff(zenith_values.to_value(u.deg)))))

        extra_axis_keys = [
            key for key in self.target_values if key not in ("azimuth", "zenith_angle")
        ]
        extra_arrays = [self.target_values[key].value for key in extra_axis_keys]
        extra_units = [self.target_values[key].unit for key in extra_axis_keys]

        if extra_arrays:
            extra_grid = np.meshgrid(*extra_arrays, indexing="ij")
            extra_combinations = np.vstack(list(map(np.ravel, extra_grid))).T
        else:
            extra_combinations = [np.array([])]

        grid_points = []
        for zenith in zenith_values:
            altitude_cosine = np.cos(np.deg2rad(90.0 - zenith.to_value(u.deg)))
            azimuth_bins = 1
            if azimuth_span > 0.0 and altitude_cosine > 0.0:
                azimuth_bins = max(
                    1,
                    self._ceil_with_tolerance(
                        azimuth_span * density * zenith_step * altitude_cosine
                    ),
                )

            azimuth_values = self.create_circular_binning(azimuth_range, azimuth_bins) * u.deg
            for extra_combination in extra_combinations:
                point_base = {
                    key: Quantity(extra_combination[i], extra_units[i])
                    for i, key in enumerate(extra_axis_keys)
                }
                for azimuth in azimuth_values:
                    point = {
                        **point_base,
                        "azimuth": azimuth,
                        "zenith_angle": zenith,
                    }
                    self._add_lookup_limits_to_point(
                        point,
                        zenith=zenith.to_value(u.deg),
                        azimuth=azimuth.to_value(u.deg),
                    )
                    grid_points.append(point)

        return grid_points

    def _generate_horizontal_grid(self):
        """Generate grid points for horizontal (zenith/azimuth) mode."""
        if self._is_adaptive_horizontal_density_enabled():
            return self._generate_adaptive_horizontal_grid()

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

            zenith_idx = np.nonzero(
                np.isclose(
                    self.target_values["zenith_angle"].value,
                    grid_point["zenith_angle"].value,
                )
            )[0][0]
            azimuth_idx = np.nonzero(
                np.isclose(
                    self.target_values["azimuth"].value,
                    grid_point["azimuth"].value,
                )
            )[0][0]
            nsb_idx = np.nonzero(
                np.isclose(
                    self.target_values["nsb_level"].value,
                    grid_point["nsb_level"].value,
                )
            )[0][0]

            if "lower_energy_threshold" in self.interpolated_limits:
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

    def generate_simulation_grid(self):
        """Generate observation grid with CORSIKA limits. Always includes Alt/Az for backends."""
        if self.coordinate_system == "ra_dec":
            return self._generate_grid_radec_mode(include_horizontal_coordinates=True)
        return self._generate_horizontal_grid()

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
        if self.time_of_observation is None:
            raise ValueError("Conversion to RA/Dec requires time_of_observation to be set.")

        aa = AltAz(
            alt=alt.to(u.rad),
            az=az.to(u.rad),
            location=self.observing_location,
            obstime=self.time_of_observation,
        )
        sky_coord = SkyCoord(aa)
        return sky_coord.icrs  # Return RA/Dec in ICRS frame

    def convert_coordinates(self, grid_points, keep_horizontal_coordinates=False):
        """
        Convert the grid points RA/Dec coordinates if necessary.

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
                    alt = (90.0 * u.deg) - point["zenith_angle"]
                    az = point["azimuth"]
                    radec = self.convert_altaz_to_radec(alt, az)
                    if not keep_horizontal_coordinates:
                        point.pop("zenith_angle")
                        point.pop("azimuth")
                    point["ra"] = radec.ra.deg * u.deg
                    point["dec"] = radec.dec.deg * u.deg
        return grid_points
