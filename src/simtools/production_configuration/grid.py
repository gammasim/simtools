"""Expand observation grids into full simulation job matrices.

Combines grids (from ProductionGridEngine or explicit axes) with primaries, interactions,
model versions, energy ranges, and run counts into a complete job parameter set.
"""

import itertools
import logging
import shlex
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.units import Quantity

from simtools.configuration import defaults
from simtools.configuration.argument_helpers import parse_quantity_pair
from simtools.layout.array_layout_utils import resolve_array_layout_name
from simtools.model.site_model import SiteModel
from simtools.production_configuration.corsika_limits_lookup import (
    CorsikaLimitsLookup,
    attach_lookup_limits_to_point,
)
from simtools.production_configuration.jobs_io import (
    build_job_grid_summary,
    format_quantity_field,
    serialize_job_grid,
)
from simtools.production_configuration.production_definition import (
    production_definition_from_arguments,
)
from simtools.utils.value_conversion import ensure_list

logger = logging.getLogger(__name__)

TOTAL_SHOWERS_ROUNDING_WARNINGS_MAX_DEFAULT = 20
MAX_SHOWERS_PER_RUN_ARGUMENT = "max_showers_per_run"
NUMBER_OF_RUNS_ARGUMENT = "number_of_runs"
TOTAL_SHOWERS_ARGUMENT = "total_showers"

_GRID_AXIS_ENGINE_NAMES = {
    "azimuth": "azimuth",
    "zenith": "zenith_angle",
    "ha": "ha",
    "dec": "dec",
    "offset": "offset",
}
_HORIZONTAL_AXES = ("azimuth", "zenith")
_HADEC_AXES = ("ha", "dec")
_AXIS_SCALING_CHOICES = ("linear", "log", "1/cos")


def _is_full_circle(angle_range):
    """Return True if a (start, end) degree range covers the full circle."""
    raw_span = abs(float(angle_range[1]) - float(angle_range[0]))
    return bool(raw_span > 0.0 and np.isclose(raw_span % 360.0, 0.0))


def directed_circular_span_degrees(axis_range):
    """Return the directed circular span in degrees; full-circle ranges give ``360.0``."""
    if _is_full_circle(axis_range):
        return 360.0
    return float((axis_range[1] - axis_range[0]) % 360.0)


def ceil_with_tolerance(value):
    """Ceil a float while avoiding near-integer floating-point artifacts."""
    nearest_integer = round(value)
    if np.isclose(value, nearest_integer):
        return int(nearest_integer)
    return int(np.ceil(value))


def _spherical_angles(angle_x, angle_y, latitude):
    """Return ``(arcsin, arctan2)`` terms of the HA/Dec <-> zenith/azimuth rotation.

    With ``angle_x`` = hour angle and ``angle_y`` = declination this yields altitude and
    azimuth; with ``angle_x`` = azimuth and ``angle_y`` = altitude it yields declination
    and hour angle.
    """
    arcsin_term = np.arcsin(
        np.sin(latitude) * np.sin(angle_y) + np.cos(latitude) * np.cos(angle_y) * np.cos(angle_x)
    )
    arctan2_term = np.arctan2(
        -np.sin(angle_x) * np.cos(angle_y),
        np.sin(angle_y) * np.cos(latitude) - np.cos(angle_y) * np.sin(latitude) * np.cos(angle_x),
    )
    return arcsin_term, arctan2_term


class ProductionGridEngine:
    """Generate observation grids with direction-dependent CORSIKA physics limits."""

    def __init__(
        self,
        axes,
        coordinate_system="horizontal",
        observing_location=None,
        lookup_table=None,
        array_layout_name=None,
        lookup_nsb_rate=None,
    ):
        """Initialize the production-grid engine."""
        self.axes = axes["axes"] if "axes" in axes else axes
        self.coordinate_system = coordinate_system
        self.observing_location = (
            observing_location
            if observing_location is not None
            else EarthLocation(lat=0.0 * u.deg, lon=0.0 * u.deg, height=0 * u.m)
        )
        self.lookup_table = lookup_table
        self.lookup_nsb_rate = lookup_nsb_rate
        self._limits_lookup = None
        if self.lookup_table:
            self._limits_lookup = CorsikaLimitsLookup(
                lookup_table=lookup_table, array_layout_name=array_layout_name
            )
            self._prepare_lookup_table_limits_for_point_interpolation()
        self.target_values = self._generate_target_values()

    def _get_max_zenith_for_hadec_mode(self):
        """Read maximum zenith from axes for HA/Dec direction sampling."""
        zenith_range = (self.axes.get("zenith_angle") or {}).get("range", [])
        if len(zenith_range) != 2:
            raise ValueError(
                "HA/Dec direction sampling requires 'zenith_angle' axis with a valid "
                "two-element 'range' in the axes definition."
            )
        return float(zenith_range[1])

    def _prepare_lookup_table_limits_for_point_interpolation(self):
        """Prepare lookup arrays for per-point interpolation."""
        self._limits_lookup.prepare_point_interpolators()

    def _generate_target_values(self):
        """Generate target axis values and store them as Quantities."""
        target_values = {}
        for axis_name, axis in self.axes.items():
            axis_range, binning = axis["range"], axis["binning"]
            scaling = axis.get("scaling", "linear")
            if axis_name == "azimuth":
                values = self.create_circular_binning(axis_range, binning)
            elif scaling == "log":
                values = np.logspace(np.log10(axis_range[0]), np.log10(axis_range[1]), binning)
            elif scaling == "1/cos":
                inv_cos = np.linspace(
                    1 / np.cos(np.radians(axis_range[0])),
                    1 / np.cos(np.radians(axis_range[1])),
                    binning,
                )
                values = np.degrees(np.arccos(1 / inv_cos))
            else:
                values = np.linspace(axis_range[0], axis_range[1], binning)
            if axis.get("units"):
                values = values * u.Unit(axis["units"])
            target_values[axis_name] = values
        return target_values

    def _generate_hadec_grid_direction_points(self):
        """Generate direction points from declination lines and hour-angle spacing."""
        max_zenith = self._get_max_zenith_for_hadec_mode()
        direction_points = []
        for declination in np.arange(-90.0, 91.0, 1.0):
            cos_dec = np.cos(np.deg2rad(declination))
            step_ha = 1.0 / cos_dec if cos_dec > 1e-6 else 360.0
            n_ha = max(1, int(np.ceil(360.0 / step_ha)))
            hour_angles = np.linspace(-180.0, 180.0, n_ha, endpoint=False)
            zenith, azimuth = self._hadec_to_horizontal(
                hour_angles * u.deg, np.full_like(hour_angles, declination) * u.deg
            )
            zenith_deg = zenith.to_value(u.deg)
            for idx in np.nonzero((zenith_deg >= 0.0) & (zenith_deg <= max_zenith))[0]:
                direction_points.append({"zenith_angle": zenith[idx], "azimuth": azimuth[idx]})
        return direction_points

    def _generate_extra_axis_combinations(self, excluded_keys):
        """Generate combinations for all axes except the excluded ones."""
        extra_axes = {k: v for k, v in self.target_values.items() if k not in excluded_keys}
        if not extra_axes:
            return list(extra_axes.keys()), [], [np.array([])]
        extra_grid = np.meshgrid(*[v.value for v in extra_axes.values()], indexing="ij")
        extra_combinations = np.vstack(list(map(np.ravel, extra_grid))).T
        return list(extra_axes.keys()), [v.unit for v in extra_axes.values()], extra_combinations

    def _extra_axis_points(self, excluded_keys):
        """Return one ``{axis: Quantity}`` dict per combination of the remaining axes."""
        keys, units, combinations = self._generate_extra_axis_combinations(excluded_keys)
        return [
            {key: Quantity(combination[i], units[i]) for i, key in enumerate(keys)}
            for combination in combinations
        ]

    def _interpolate_limits_for_point(self, zenith, azimuth, nsb):
        """Interpolate lookup-table limits for a single point."""
        return self._limits_lookup.interpolate_point(zenith, azimuth, nsb)

    def _add_lookup_limits_to_point(self, point, zenith, azimuth):
        """Interpolate and attach lookup-table limits to a grid point."""
        if not self.lookup_table:
            return
        nsb_value = self.lookup_nsb_rate
        if nsb_value is None:
            nsb_value = point.get("nsb_level")
        if nsb_value is None:
            raise ValueError("nsb_rate is required to interpolate lookup limits.")
        if isinstance(nsb_value, Quantity):
            nsb_value = nsb_value.value
        point["nsb_rate"] = float(nsb_value)
        limits = self._interpolate_limits_for_point(
            zenith=zenith, azimuth=azimuth, nsb=float(nsb_value)
        )
        attach_lookup_limits_to_point(
            point, limits, getattr(self._limits_lookup, "lookup_field_units", None)
        )

    def _generate_grid_from_hadec_axes(self, include_horizontal_coordinates=False):
        """Generate grid points from explicit HA/Dec axes definitions."""
        if self._is_adaptive_hadec_density_enabled():
            return self._generate_adaptive_hadec_grid(include_horizontal_coordinates)

        grid_points = self._extra_axis_points(excluded_keys=("zenith_angle", "azimuth"))
        for grid_point in grid_points:
            zenith, azimuth = self._hadec_to_horizontal(grid_point["ha"], grid_point["dec"])
            if include_horizontal_coordinates:
                grid_point["zenith_angle"] = zenith
                grid_point["azimuth"] = azimuth
            self._add_lookup_limits_to_point(grid_point, zenith=zenith.value, azimuth=azimuth.value)
        return grid_points

    def _generate_grid_hadec_mode(self, include_horizontal_coordinates=False):
        """Generate grid points for HA/Dec mode."""
        if "ha" in self.axes and "dec" in self.axes:
            return self._generate_grid_from_hadec_axes(include_horizontal_coordinates)

        direction_points = self._generate_hadec_grid_direction_points()
        extra_points = self._extra_axis_points(excluded_keys=("zenith_angle", "azimuth"))
        grid_points = []
        for direction_point in direction_points:
            for extra_point in extra_points:
                point = {**direction_point, **extra_point}
                self._add_lookup_limits_to_point(
                    point, zenith=point["zenith_angle"].value, azimuth=point["azimuth"].value
                )
                grid_points.append(point)
        return self.convert_coordinates(grid_points, include_horizontal_coordinates)

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
        azimuth_start = float(azimuth_range[0]) % 360.0
        if _is_full_circle(azimuth_range):
            return (azimuth_start + np.linspace(0.0, 360.0, num_bins, endpoint=False)) % 360.0
        span = directed_circular_span_degrees(azimuth_range)
        return np.linspace(azimuth_start, azimuth_start + span, num_bins) % 360.0

    @staticmethod
    def _is_in_directed_azimuth_range(azimuth_values_deg, azimuth_range):
        """Return mask of azimuth values inside directed circular range [start -> end]."""
        if _is_full_circle(azimuth_range):
            return np.ones_like(azimuth_values_deg, dtype=bool)
        start = float(azimuth_range[0]) % 360.0
        span = directed_circular_span_degrees(azimuth_range)
        offsets = (np.asarray(azimuth_values_deg) - start) % 360.0
        return offsets <= (span + 1e-12)

    def _is_adaptive_hadec_density_enabled(self):
        """Return whether HA/Dec grid generation uses per-declination adaptive HA bins."""
        return (
            self.coordinate_system == "ha_dec"
            and "ha" in self.axes
            and "dec" in self.axes
            and self.axes["ha"].get("direction_grid_density") is not None
        )

    def _generate_adaptive_hadec_grid(self, include_horizontal_coordinates=False):
        """Generate HA/Dec grid with hour-angle steps adapted per declination strip."""
        ha_axis = self.axes["ha"]
        spacing = 1.0 / np.sqrt(float(ha_axis["direction_grid_density"]))
        dec_min, dec_max = sorted(float(value) for value in self.axes["dec"]["range"])
        point_bases = self._extra_axis_points(excluded_keys=("ha", "dec"))

        grid_points = []
        for dec_deg in np.arange(dec_min, dec_max + 0.5 * spacing, spacing):
            if np.cos(np.deg2rad(dec_deg)) <= 0.0:
                continue
            grid_points.extend(
                self._adaptive_hadec_strip_points(
                    dec_deg, spacing, point_bases, include_horizontal_coordinates
                )
            )
        return grid_points

    def _adaptive_hadec_strip_points(
        self, dec_deg, spacing, point_bases, include_horizontal_coordinates
    ):
        """Return the visible HA/Dec grid points of one declination strip."""
        cos_dec = np.cos(np.deg2rad(dec_deg))
        ha_values = np.arange(-180.0, 180.0, spacing / cos_dec)
        if len(ha_values) == 0:
            ha_values = np.array([0.0])
        hour_angles = ha_values * u.deg
        zenith, azimuth = self._hadec_to_horizontal(
            hour_angles, np.full(len(hour_angles), dec_deg) * u.deg
        )
        visible = self._adaptive_hadec_visibility(zenith, azimuth)
        if not np.any(visible):
            return []

        dec = dec_deg * u.deg
        grid_points = []
        for point_base in point_bases:
            for ha, zen, az in zip(
                hour_angles[visible], zenith[visible], azimuth[visible], strict=True
            ):
                point = {**point_base, "ha": ha, "dec": dec}
                if include_horizontal_coordinates:
                    point["zenith_angle"] = zen
                    point["azimuth"] = az
                self._add_lookup_limits_to_point(point, zenith=zen.value, azimuth=az.value)
                grid_points.append(point)
        return grid_points

    def _adaptive_hadec_visibility(self, zenith, azimuth):
        """Return the mask of HA/Dec samples inside the local zenith/azimuth constraints."""
        ha_axis = self.axes["ha"]
        visible = zenith.to_value(u.deg) <= 90.0
        if ha_axis.get("local_zenith_range") is not None:
            visible &= _in_range(zenith.to_value(u.deg), ha_axis["local_zenith_range"])
        if ha_axis.get("local_azimuth_range") is not None:
            visible &= self._is_in_directed_azimuth_range(
                azimuth.to_value(u.deg), ha_axis["local_azimuth_range"]
            )
        return visible

    def _generate_horizontal_grid(self):
        """Generate grid points for horizontal (zenith/azimuth) mode."""
        if not (
            self.coordinate_system == "horizontal"
            and "azimuth" in self.axes
            and "zenith_angle" in self.axes
            and self.axes["azimuth"].get("direction_grid_density") is not None
        ):
            grid_points = self._extra_axis_points(excluded_keys=())
            for point in grid_points:
                self._add_lookup_limits_to_point(
                    point,
                    zenith=point["zenith_angle"].to_value(u.deg),
                    azimuth=point["azimuth"].to_value(u.deg),
                )
            return grid_points

        # Adaptive mode: azimuth bins per zenith row follow the local sky area.
        density = float(self.axes["azimuth"]["direction_grid_density"])
        azimuth_range = self.axes["azimuth"]["range"]
        azimuth_span = directed_circular_span_degrees(azimuth_range)
        zenith_values = self.target_values["zenith_angle"]
        zenith_step = 1.0 / np.sqrt(density)
        if len(zenith_values) > 1:
            zenith_step = float(np.mean(np.abs(np.diff(zenith_values.to_value(u.deg)))))

        point_bases = self._extra_axis_points(excluded_keys=("azimuth", "zenith_angle"))
        grid_points = []
        for zenith in zenith_values:
            altitude_cosine = np.cos(np.deg2rad(90.0 - zenith.to_value(u.deg)))
            azimuth_bins = 1
            if azimuth_span > 0.0 and altitude_cosine > 0.0:
                azimuth_bins = max(
                    1, ceil_with_tolerance(azimuth_span * density * zenith_step * altitude_cosine)
                )
            azimuth_values = self.create_circular_binning(azimuth_range, azimuth_bins) * u.deg
            for point_base in point_bases:
                for azimuth in azimuth_values:
                    point = {**point_base, "azimuth": azimuth, "zenith_angle": zenith}
                    self._add_lookup_limits_to_point(
                        point, zenith=zenith.to_value(u.deg), azimuth=azimuth.to_value(u.deg)
                    )
                    grid_points.append(point)
        return grid_points

    def generate_simulation_grid(self):
        """Generate a grid for the configured coordinate system."""
        if self.coordinate_system == "ha_dec":
            return self._generate_grid_hadec_mode(include_horizontal_coordinates=True)
        return self._generate_horizontal_grid()

    def _hadec_to_horizontal(self, hour_angle, declination):
        """Convert local HA/Dec coordinates to zenith and azimuth angles."""
        altitude, azimuth = _spherical_angles(
            u.Quantity(hour_angle).to_value(u.rad),
            u.Quantity(declination).to_value(u.rad),
            self.observing_location.lat.to_value(u.rad),
        )
        return (90.0 * u.deg - altitude * u.rad).to(u.deg), (azimuth * u.rad).to(u.deg) % (
            360.0 * u.deg
        )

    def convert_altaz_to_hadec(self, alt, az):
        """
        Convert Altitude/Azimuth (AltAz) coordinates to HA/Dec.

        Parameters
        ----------
        alt : float
            Altitude angle in degrees.
        az : float
            Azimuth angle in degrees.

        Returns
        -------
        SimpleNamespace
            Object exposing ``ha`` and ``dec`` angle attributes.
        """
        declination, hour_angle = _spherical_angles(
            u.Quantity(az).to_value(u.rad),
            u.Quantity(alt).to_value(u.rad),
            self.observing_location.lat.to_value(u.rad),
        )
        return SimpleNamespace(ha=hour_angle * u.rad, dec=declination * u.rad)

    def convert_coordinates(self, grid_points, keep_horizontal_coordinates=False):
        """
        Convert the grid points HA/Dec coordinates if necessary.

        Parameters
        ----------
        grid_points : list of dict
            List of grid points.
        keep_horizontal_coordinates : bool
            Keep ``zenith_angle`` and ``azimuth`` in the converted points.

        Returns
        -------
        list of dict
            The grid points with converted HA/Dec coordinates.
        """
        for point in grid_points:
            if "zenith_angle" in point and "azimuth" in point:
                hadec = self.convert_altaz_to_hadec(
                    (90.0 * u.deg) - point["zenith_angle"], point["azimuth"]
                )
                if not keep_horizontal_coordinates:
                    point.pop("zenith_angle")
                    point.pop("azimuth")
                point["ha"] = hadec.ha.to(u.deg)
                point["dec"] = hadec.dec.to(u.deg)
        return grid_points


def _in_range(values_deg, angle_range):
    """Return mask of values within a (possibly reversed) closed interval."""
    low, high = sorted(float(value) for value in angle_range)
    return (values_deg >= low) & (values_deg <= high)


@dataclass
class _RunAllocationConfig:
    """Run-count settings shared across all energy ranges at one grid point."""

    total_showers: int | None = None
    total_showers_scaling: str = "fixed"
    zenith_angle_scaling_factor: float = defaults.ZENITH_ANGLE_SCALING_FACTOR_DEFAULT
    max_showers_per_run: int | None = None
    rounding_warning_state: dict | None = None


def _parse_axis_range_tokens(range_tokens):
    """Parse a quantity pair from CLI axis range tokens."""
    if len(range_tokens) == 1:
        if ".." in range_tokens[0]:
            return tuple(
                u.Quantity(value.strip()) for value in range_tokens[0].split("..", maxsplit=1)
            )
        return parse_quantity_pair(range_tokens[0])
    if len(range_tokens) == 2:
        return tuple(u.Quantity(value) for value in range_tokens)
    if len(range_tokens) == 4:
        return tuple(
            u.Quantity(f"{range_tokens[index]} {range_tokens[index + 1]}")
            for index in range(0, len(range_tokens), 2)
        )
    raise ValueError("Axis range must contain exactly two quantities.")


def _normalize_axis_spec_tokens(axis_spec):
    """Return axis specification tokens from CLI or configuration input."""
    if isinstance(axis_spec, str):
        return shlex.split(axis_spec)
    if isinstance(axis_spec, (list, tuple)):
        if len(axis_spec) == 1 and isinstance(axis_spec[0], str):
            return shlex.split(axis_spec[0])
        return [str(token) for token in axis_spec]
    raise TypeError("Axis definitions must be strings or lists of CLI-style tokens.")


def _parse_axis_spec(axis_spec):
    """Parse one compact axis definition ``NAME RANGE... BINNING [SCALING]``."""
    tokens = _normalize_axis_spec_tokens(axis_spec)
    if len(tokens) < 3:
        raise ValueError(
            "Axis definitions require at least an axis name, range, and binning value."
        )

    axis_name = tokens[0]
    if axis_name not in _GRID_AXIS_ENGINE_NAMES:
        supported_axes = ", ".join(sorted(_GRID_AXIS_ENGINE_NAMES))
        raise ValueError(f"Unknown axis '{axis_name}'. Supported axes: {supported_axes}.")

    if tokens[-1] in _AXIS_SCALING_CHOICES:
        scaling, binning_token, range_tokens = tokens[-1], tokens[-2], tokens[1:-2]
    else:
        scaling, binning_token, range_tokens = "linear", tokens[-1], tokens[1:-1]
    if not range_tokens:
        raise ValueError(f"Axis '{axis_name}' is missing its range definition.")
    try:
        binning = int(binning_token)
    except ValueError as exc:
        raise ValueError(f"Axis '{axis_name}' binning must be an integer.") from exc

    axis_range = _parse_axis_range_tokens(range_tokens)
    return axis_name, {
        "range": [u.Quantity(value).to_value("deg") for value in axis_range],
        "binning": binning,
        "scaling": scaling,
        "units": "deg",
    }


def _iter_compact_axis_specs(args_dict):
    """Iterate over compact axis definitions from CLI or configuration."""
    axis_specs = args_dict.get("axis") or []
    if isinstance(axis_specs, str):
        axis_specs = [axis_specs]

    normalized_specs = []
    for axis_spec in axis_specs:
        is_merged_list = (
            isinstance(axis_spec, (list, tuple))
            and len(axis_spec) > 1
            and all(isinstance(item, str) for item in axis_spec)
            and str(axis_spec[0]).strip() not in _GRID_AXIS_ENGINE_NAMES
            and " " in str(axis_spec[0])
        )
        if is_merged_list:
            normalized_specs.extend(axis_spec)
        else:
            normalized_specs.append(axis_spec)
    return normalized_specs


def _resolve_axis_configs(args_dict):
    """Resolve compact axis definitions into one normalized mapping."""
    return dict(map(_parse_axis_spec, _iter_compact_axis_specs(args_dict)))


def _parse_direction_grid_density(density_value):
    """Parse direction-grid density and normalize it to ``1/deg^2`` units."""
    if density_value is None:
        return None
    if isinstance(density_value, (int, float)):
        return float(density_value)
    if isinstance(density_value, (str, list, tuple)):
        tokens = _normalize_axis_spec_tokens(density_value)
        if len(tokens) == 1:
            return float(tokens[0])
        try:
            return u.Quantity(" ".join(tokens)).to_value(1 / u.deg**2)
        except (TypeError, ValueError, u.UnitConversionError) as exc:
            raise ValueError(
                "direction_grid_density must be a float or quantity in 1/deg^2."
            ) from exc
    raise TypeError("direction_grid_density must be a number, string, or CLI-style token list.")


def _mean_over_angle_span(angle_range_deg, function, antiderivative):
    """Return the absolute mean of ``function`` over an angle interval (degrees)."""
    low, high = sorted((float(angle_range_deg[0]), float(angle_range_deg[1])))
    span_rad = np.deg2rad(high - low)
    if np.isclose(span_rad, 0.0):
        return abs(function(np.deg2rad(low)))
    return abs((antiderivative(np.deg2rad(high)) - antiderivative(np.deg2rad(low))) / span_rad)


def _resolve_coordinate_system(axis_configs):
    """Resolve the coordinate system from axis definitions."""
    has_horizontal_axes = all(axis_name in axis_configs for axis_name in _HORIZONTAL_AXES)
    has_hadec_axes = all(axis_name in axis_configs for axis_name in _HADEC_AXES)
    if has_horizontal_axes and has_hadec_axes:
        raise ValueError("Cannot define both azimuth/zenith and ha/dec axes at the same time.")
    if has_hadec_axes:
        return "ha_dec"
    if has_horizontal_axes:
        return "horizontal"
    return None


def resolve_single_model_version(model_version):
    """Resolve one model version for helpers that require a scalar version."""
    if isinstance(model_version, list):
        return model_version[0]
    return model_version


def build_observing_location(site, model_version, context=None):
    """Build observing location from the site model.

    Parameters
    ----------
    site : str
        Site name.
    model_version : str or list[str]
        Model version used to read site coordinates.
    context : RunContext, optional
        Run configuration and model reader.
    """
    site_model = SiteModel(
        model_version=resolve_single_model_version(model_version), site=site, context=context
    )
    return EarthLocation(
        lat=site_model.get_parameter_value_with_unit("reference_point_latitude"),
        lon=site_model.get_parameter_value_with_unit("reference_point_longitude"),
        height=site_model.get_parameter_value_with_unit("reference_point_altitude"),
    )


def _apply_direction_grid_density(axis_configs, direction_axes, density):
    """Set the direction-grid density and derive direction-axis binning from it (points/deg^2)."""
    axis_configs[direction_axes[0]]["direction_grid_density"] = density
    if density <= 0:
        raise ValueError("direction_grid_density must be strictly positive.")
    if direction_axes == _HADEC_AXES:
        longitudinal_scale = _mean_over_angle_span(axis_configs["dec"]["range"], np.cos, np.sin)
    else:
        longitudinal_scale = _mean_over_angle_span(
            axis_configs["zenith"]["range"], np.sin, lambda angle: -np.cos(angle)
        )
    for axis_name in direction_axes:
        axis_range = axis_configs[axis_name]["range"]
        span_degrees = (
            directed_circular_span_degrees(axis_range)
            if axis_name == "azimuth"
            else abs(axis_range[1] - axis_range[0])
        )
        if axis_name in ("azimuth", "ha"):
            span_degrees *= longitudinal_scale
        axis_configs[axis_name]["binning"] = max(
            1, ceil_with_tolerance(span_degrees * np.sqrt(density))
        )


def _apply_local_hadec_constraints(axis_configs, args_dict):
    """Store optional local zenith/azimuth constraints on the HA axis."""
    for argument in ("local_zenith_range", "local_azimuth_range"):
        if args_dict.get(argument) is not None:
            range_quantities = _parse_axis_range_tokens(
                _normalize_axis_spec_tokens(args_dict[argument])
            )
            axis_configs["ha"][argument] = [
                u.Quantity(value).to_value("deg") for value in range_quantities
            ]


def build_axes_dict_from_cli_args(args_dict):
    """Build ProductionGridEngine-compatible axes configuration from CLI arguments."""
    axis_configs = _resolve_axis_configs(args_dict)
    coordinate_system = _resolve_coordinate_system(axis_configs)
    if coordinate_system is None:
        raise ValueError("Must provide either both azimuth/zenith or both ha/dec axis definitions.")
    if "offset" not in axis_configs:
        raise ValueError("Missing required shared axis definition(s): offset.")

    direction_axes = _HADEC_AXES if coordinate_system == "ha_dec" else _HORIZONTAL_AXES
    density = _parse_direction_grid_density(args_dict.get("direction_grid_density"))
    if density is not None:
        _apply_direction_grid_density(axis_configs, direction_axes, density)

    if coordinate_system == "ha_dec":
        _apply_local_hadec_constraints(axis_configs, args_dict)

    return {
        _GRID_AXIS_ENGINE_NAMES[axis_name]: axis_configs[axis_name]
        for axis_name in ("offset", *direction_axes)
    }


def _resolve_nsb_rate(args_dict, model_version, context=None):
    """Resolve the NSB interpolation rate for one model version."""
    if not args_dict.get("site") or not model_version:
        raise ValueError("site and model_version are required to resolve nsb_rate.")
    return float(
        SiteModel(
            model_version=model_version,
            site=args_dict["site"],
            context=context,
        ).get_nsb_integrated_flux()
    )


def build_production_grid_engine(
    args_dict, array_layout_name=None, model_version=None, context=None
):
    """
    Build a production-grid engine from application arguments.

    Parameters
    ----------
    args_dict : dict
        Application arguments (axis definitions, site, model version, optional CORSIKA
        lookup table, array layout name).
    array_layout_name : str, optional
        Resolved array layout name; resolved from ``args_dict`` if omitted.
    model_version : str, optional
        Model version for observing location, array layout, and NSB rate.
        Defaults to the first model version in ``args_dict``.
    context : RunContext, optional
        Run configuration and model reader.

    Returns
    -------
    ProductionGridEngine
        Configured production-grid engine.
    """
    axes = build_axes_dict_from_cli_args(args_dict)
    coordinate_system = _resolve_coordinate_system(_resolve_axis_configs(args_dict))
    resolved_model_version = model_version or resolve_single_model_version(
        args_dict.get("model_version")
    )
    observing_location = None
    if args_dict.get("site") and resolved_model_version:
        observing_location = build_observing_location(
            site=args_dict["site"], model_version=resolved_model_version, context=context
        )
    if coordinate_system == "ha_dec" and observing_location is None:
        raise ValueError("site is required when using HA/Dec axes.")
    resolved_layout_name = array_layout_name or resolve_array_layout_name(
        args_dict.get("array_layout_name"), resolved_model_version
    )

    return ProductionGridEngine(
        axes=axes,
        coordinate_system=coordinate_system,
        observing_location=observing_location,
        lookup_table=args_dict.get("corsika_limits"),
        array_layout_name=resolved_layout_name,
        lookup_nsb_rate=_resolve_nsb_rate(args_dict, resolved_model_version, context=context),
    )


def build_job_grid_metadata(args_dict):
    """Build metadata stored alongside serialized executable job grids."""
    coordinate_system = _resolve_coordinate_system(_resolve_axis_configs(args_dict))
    direction_grid_density = _parse_direction_grid_density(args_dict.get("direction_grid_density"))
    if coordinate_system == "ha_dec" and not args_dict.get("site"):
        raise ValueError("site is required when using HA/Dec axes.")
    corsika_limits = args_dict.get("corsika_limits")
    return {
        "site": args_dict.get("site"),
        "simulation_software": args_dict.get("simulation_software"),
        "coordinate_system": coordinate_system,
        "direction_grid_density": direction_grid_density,
        "direction_grid_density_unit": "1/deg^2" if direction_grid_density is not None else None,
        "corsika_limits": str(corsika_limits) if corsika_limits else None,
    }


def normalize_grid_axes(args_dict):
    """Return normalized grid axes for cartesian product expansion."""
    axis_defaults = {
        "corsika_le_interaction": defaults.CORSIKA_LE_INTERACTION,
        "corsika_he_interaction": defaults.CORSIKA_HE_INTERACTION,
    }
    return {
        axis: (
            ensure_list(args_dict[axis])
            if args_dict.get(axis) is not None
            else [axis_defaults.get(axis)]
        )
        for axis in (
            "primary",
            "azimuth_angle",
            "zenith_angle",
            "model_version",
            "corsika_le_interaction",
            "corsika_he_interaction",
        )
    }


def normalize_energy_ranges(energy_range):
    """Normalize energy range argument to a list of ``(e_min, e_max)`` pairs."""
    if isinstance(energy_range, tuple) and len(energy_range) == 2:
        return [energy_range]
    if isinstance(energy_range, list):
        if len(energy_range) == 2 and all(hasattr(item, "to") for item in energy_range):
            return [(energy_range[0], energy_range[1])]
        if all(isinstance(item, (list, tuple)) and len(item) == 2 for item in energy_range):
            return [tuple(item) for item in energy_range]
    raise ValueError(
        "energy_range must be one pair (e_min, e_max) or a list of (e_min, e_max) pairs."
    )


def calculate_log_energy_midpoint(energy_range_pair):
    """Return the geometric-mean energy for an energy range pair."""
    energy_min, energy_max = energy_range_pair
    if not isinstance(energy_min, u.Quantity) or not isinstance(energy_max, u.Quantity):
        raise TypeError("energy_range_pair must contain astropy Quantity values.")

    energy_min_tev = energy_min.to(u.TeV)
    energy_max_tev = energy_max.to(u.TeV)
    if energy_min_tev <= 0 * u.TeV or energy_max_tev <= 0 * u.TeV:
        raise ValueError("Energy range values must be strictly positive.")

    mean_log_energy = np.mean([np.log10(energy_min_tev.value), np.log10(energy_max_tev.value)])
    return 10**mean_log_energy * u.TeV


def calculate_scaled_showers_per_run(
    energy_range_pair, baseline_showers_per_run, showers_per_run_power_law=None
):
    """Return an energy-dependent showers per run value."""
    if baseline_showers_per_run < 1:
        raise ValueError("baseline_showers_per_run must be a positive integer.")
    if showers_per_run_power_law is None:
        return baseline_showers_per_run
    if len(showers_per_run_power_law) != 2:
        raise ValueError(
            "showers_per_run_power_law must contain exactly two values: "
            "(power_index, reference_energy)."
        )

    showers_per_run_power_index, reference_energy = showers_per_run_power_law
    midpoint_energy = calculate_log_energy_midpoint(energy_range_pair)
    scaling_factor = (midpoint_energy / reference_energy.to(midpoint_energy.unit)).to_value(
        u.dimensionless_unscaled
    ) ** showers_per_run_power_index
    scaled_showers_per_run = int(np.ceil(baseline_showers_per_run * scaling_factor))
    if scaled_showers_per_run < 1:
        raise ValueError("Scaled showers per run must be at least 1.")
    return scaled_showers_per_run


def calculate_zenith_scaled_showers_per_run(
    zenith_angle, baseline_showers_per_run, showers_per_run_scaling="fixed"
):
    """Return a zenith-angle-dependent showers per run value.

    Parameters
    ----------
    zenith_angle : astropy.units.Quantity
        Zenith angle for one simulation point.
    baseline_showers_per_run : int
        Showers-per-run value before zenith-angle scaling.
    showers_per_run_scaling : str
        Scaling mode ('fixed' or 'cosine_zenith').

    Returns
    -------
    int
        Zenith-scaled showers-per-run value.
    """
    if baseline_showers_per_run < 1:
        raise ValueError("baseline_showers_per_run must be a positive integer.")
    if showers_per_run_scaling == "fixed":
        return baseline_showers_per_run
    if showers_per_run_scaling == "cosine_zenith":
        cos_zenith = np.round(np.cos(zenith_angle.to(u.rad).value), decimals=12)
        scaled_showers_per_run = int(np.ceil(baseline_showers_per_run * cos_zenith))
        if scaled_showers_per_run < 1:
            raise ValueError("Scaled showers per run must be at least 1.")
        return scaled_showers_per_run
    raise ValueError(f"Unknown showers_per_run_scaling mode: {showers_per_run_scaling}")


def scale_energy_max_for_zenith_angle(zenith_angle, energy_range_pair, energy_max_scaling=None):
    """
    Scale energy_max with zenith angle and return an updated energy range pair.

    Parameters
    ----------
    zenith_angle : astropy.units.Quantity
        Zenith angle for one simulation point.
    energy_range_pair : tuple
        Tuple containing the minimum and maximum energy values.
    energy_max_scaling : tuple or None
        Tuple containing the scaling index and reference energy, or None.

    Returns
    -------
    tuple or None
        Updated energy range pair, or None if the scaling results in an invalid range.
    """
    if energy_max_scaling is None:
        return energy_range_pair

    energy_max_scaling_index, energy_max_scaling_reference = energy_max_scaling
    energy_min, energy_range_max = energy_range_pair
    cos_zenith = np.cos(zenith_angle.to(u.rad).value)

    if cos_zenith < 0:
        raise ValueError(
            "energy_max_scaling is only defined for zenith angles with cos(zenith) "
            ">= 0 (zenith <= 90 deg)."
        )
    if np.isclose(cos_zenith, 0.0) and energy_max_scaling_index < 0:
        raise ValueError(
            "energy_max_scaling with index < 0 is not defined for zenith angles with cos(zenith)=0."
        )

    scaling_reference = (
        energy_range_max
        if energy_max_scaling_reference is None
        else energy_max_scaling_reference.to(energy_range_max.unit)
    )
    scaled_energy_max = scaling_reference * (cos_zenith**energy_max_scaling_index)
    if scaled_energy_max <= energy_min.to(scaled_energy_max.unit):
        return None
    return energy_min, scaled_energy_max.to(energy_range_max.unit)


def _clip_energy_range_from_threshold(energy_range_pair, lower_energy_threshold):
    """Clip the lower energy bound of a configured energy range."""
    if lower_energy_threshold is None:
        return energy_range_pair

    energy_min, energy_max = energy_range_pair
    lower_energy_threshold = lower_energy_threshold.to(energy_min.unit)
    if lower_energy_threshold > energy_max.to(lower_energy_threshold.unit):
        return None
    if lower_energy_threshold <= energy_min:
        return energy_range_pair
    return lower_energy_threshold, energy_max


def _clip_energy_range_to_configured_bounds(energy_range_pair, configured_energy_range_pair):
    """Clip selected energy bounds to configured bounds; zero-width ranges are valid."""
    energy_min, energy_max = energy_range_pair
    configured_energy_min, configured_energy_max = configured_energy_range_pair

    selected_energy_min = max(energy_min.to(configured_energy_min.unit), configured_energy_min)
    selected_energy_max = min(energy_max.to(configured_energy_max.unit), configured_energy_max)
    if selected_energy_max < selected_energy_min.to(selected_energy_max.unit):
        return None
    return selected_energy_min, selected_energy_max


def _clip_max_quantity(configured_max, lookup_max):
    """Clip a configured maximum value against an interpolated lookup value."""
    if lookup_max is None:
        return configured_max
    return min(configured_max, lookup_max.to(configured_max.unit))


def _parse_power_index_quantity(tokens, parameter_name):
    """Parse ``<power_index> <reference_value> <reference_unit>`` tokens."""
    if isinstance(tokens, str):
        tokens = shlex.split(tokens)
    elif len(tokens) == 1 and isinstance(tokens[0], str):
        tokens = shlex.split(tokens[0])

    if len(tokens) != 3:
        raise ValueError(
            f"{parameter_name} must be provided as "
            "<power_index> <reference_energy_value> <reference_energy_unit>."
        )
    return float(tokens[0]), u.Quantity(f"{tokens[1]} {tokens[2]}")


def _resolve_shower_params(args_dict):
    """Extract and convert shower-statistics parameters from an args dict."""
    showers_per_run = args_dict.get("showers_per_run")
    if showers_per_run is None:
        showers_per_run = args_dict.get(MAX_SHOWERS_PER_RUN_ARGUMENT)
    if showers_per_run is None:
        raise ValueError("showers_per_run is required unless max_showers_per_run is configured.")
    showers_per_run_power_law = args_dict.get("showers_per_run_power_law")
    if showers_per_run_power_law is not None:
        showers_per_run_power_law = _parse_power_index_quantity(
            showers_per_run_power_law, "showers_per_run_power_law"
        )
    return (
        showers_per_run,
        showers_per_run_power_law,
        args_dict.get("showers_per_run_scaling", "fixed"),
        args_dict.get(TOTAL_SHOWERS_ARGUMENT),
        args_dict.get("total_showers_scaling", "fixed"),
    )


def _resolve_energy_max_scaling(args_dict):
    """Resolve energy-max zenith scaling from CLI/config."""
    energy_max_scaling = args_dict.get("energy_max_scaling")
    if energy_max_scaling is None:
        return None
    return _parse_power_index_quantity(energy_max_scaling, "energy_max_scaling")


def _scale_total_showers(
    total_showers,
    zenith_angle,
    total_showers_scaling,
    cos_scaling_factor=defaults.ZENITH_ANGLE_SCALING_FACTOR_DEFAULT,
):
    """Return total showers adjusted for the selected scaling mode."""
    if total_showers_scaling == "fixed":
        return int(total_showers)
    if total_showers_scaling == "zenith_scaled":
        cos_zenith = np.cos(zenith_angle.to(u.rad).value)
        scaled_total_showers = total_showers * np.exp(cos_scaling_factor * (cos_zenith - 1))
        return int(np.ceil(scaled_total_showers))
    raise ValueError(f"Unknown total_showers_scaling mode: {total_showers_scaling}")


def _should_log_rounding_warning(rounding_warning_state):
    """Return whether to log a total-showers rounding warning and update its state."""
    if rounding_warning_state is None:
        return True
    if rounding_warning_state["emitted_warnings"] < rounding_warning_state["max_warnings"]:
        rounding_warning_state["emitted_warnings"] += 1
        return True
    if not rounding_warning_state.get("suppression_warning_emitted", False):
        logger.warning(
            "Reached max_total_showers_rounding_warnings=%s; further total_showers "
            "rounding warnings will be suppressed.",
            rounding_warning_state["max_warnings"],
        )
        rounding_warning_state["suppression_warning_emitted"] = True
    return False


def _compute_per_point_runs(run_allocation, zenith_angle, selected_showers_per_run):
    """Compute the number of runs per point considering total-showers constraints."""
    effective_total_showers = _scale_total_showers(
        run_allocation.total_showers,
        zenith_angle,
        run_allocation.total_showers_scaling,
        cos_scaling_factor=run_allocation.zenith_angle_scaling_factor,
    )
    if effective_total_showers <= 0:
        return 0, selected_showers_per_run
    if run_allocation.max_showers_per_run is not None:
        if run_allocation.max_showers_per_run < 1:
            raise ValueError(f"{MAX_SHOWERS_PER_RUN_ARGUMENT} must be a positive integer.")
        showers_per_run_limit = min(selected_showers_per_run, run_allocation.max_showers_per_run)
        per_point_number_of_runs = int(np.ceil(effective_total_showers / showers_per_run_limit))
        selected_showers_per_run = int(np.ceil(effective_total_showers / per_point_number_of_runs))
        return per_point_number_of_runs, selected_showers_per_run
    number_of_full_runs, remainder_showers = divmod(
        effective_total_showers, selected_showers_per_run
    )
    per_point_number_of_runs = number_of_full_runs + int(remainder_showers > 0)
    if remainder_showers > 0 and _should_log_rounding_warning(
        run_allocation.rounding_warning_state
    ):
        logger.warning(
            "total_showers=%s is not divisible by showers_per_run=%s; "
            "adjusting to %s to keep equal showers per run.",
            effective_total_showers,
            selected_showers_per_run,
            per_point_number_of_runs * selected_showers_per_run,
        )
    return per_point_number_of_runs, selected_showers_per_run


def _build_rows_for_point(
    point_base,
    energy_ranges,
    lower_energy_threshold,
    showers_per_run,
    showers_per_run_power_law,
    number_of_runs,
    run_number,
    run_allocation,
    showers_per_run_scaling="fixed",
    energy_max_scaling=None,
):
    """Build all simulation-run rows for a single grid point across all energy ranges."""
    rows = []
    zenith_angle = point_base["zenith_angle"]
    for energy_range_pair in energy_ranges:
        selected_energy_range = scale_energy_max_for_zenith_angle(
            zenith_angle, energy_range_pair, energy_max_scaling
        )
        if selected_energy_range is not None:
            selected_energy_range = _clip_energy_range_from_threshold(
                selected_energy_range, lower_energy_threshold
            )
        if selected_energy_range is not None:
            selected_energy_range = _clip_energy_range_to_configured_bounds(
                selected_energy_range, energy_range_pair
            )
        if selected_energy_range is None:
            continue

        selected_showers_per_run = calculate_scaled_showers_per_run(
            selected_energy_range, showers_per_run, showers_per_run_power_law
        )
        selected_showers_per_run = calculate_zenith_scaled_showers_per_run(
            zenith_angle, selected_showers_per_run, showers_per_run_scaling
        )

        per_point_number_of_runs = number_of_runs
        if run_allocation.total_showers is not None:
            per_point_number_of_runs, selected_showers_per_run = _compute_per_point_runs(
                run_allocation, zenith_angle, selected_showers_per_run
            )

        rows.extend(
            {
                **point_base,
                "configured_energy_min": energy_range_pair[0],
                "configured_energy_max": energy_range_pair[1],
                "energy_min_lookup_limit": lower_energy_threshold,
                "configured_showers_per_run": showers_per_run,
                "energy_min": selected_energy_range[0],
                "energy_max": selected_energy_range[1],
                "showers_per_run": selected_showers_per_run,
                "run_number": run_number + i,
            }
            for i in range(per_point_number_of_runs)
        )
    return rows


def _generate_observation_points_from_axes(
    azimuth_values, zenith_values, corsika_limits, nsb_rate=1.0
):
    """Sample azimuth * zenith grid with optional CORSIKA limits interpolation."""
    points = []
    for azimuth, zenith in itertools.product(azimuth_values, zenith_values):
        point = {"azimuth": azimuth, "zenith_angle": zenith, "nsb_rate": float(nsb_rate)}
        if corsika_limits is not None:
            attach_lookup_limits_to_point(
                point,
                corsika_limits.interpolate_point(zenith, azimuth, nsb=nsb_rate),
                getattr(corsika_limits, "lookup_field_units", None),
            )
        points.append(point)
    return points


def _log_energy_scaling_configuration(energy_max_scaling):
    """Log configured zenith-scaling behavior for energy maxima."""
    if energy_max_scaling is None:
        logger.info("Energy max zenith scaling: disabled.")
        return

    energy_max_scaling_index, energy_max_scaling_reference = energy_max_scaling
    if energy_max_scaling_reference is None:
        logger.info(
            "Energy max zenith scaling: Emax = E_range_max * cos(zenith)^%.3f (legacy mode).",
            energy_max_scaling_index,
        )
        return
    logger.info(
        "Energy max zenith scaling: Emax = %s * cos(zenith)^%.3f.",
        energy_max_scaling_reference,
        energy_max_scaling_index,
    )


def _log_generated_row_summary(rows):
    """Log a compact summary of generated row ranges for user visibility."""
    summary = build_job_grid_summary(rows)
    if summary["simulation_rows"] == 0:
        logger.info("Generated 0 simulation rows after applying all clipping and scaling rules.")
        return

    logger.info("Generated %d simulation rows.", summary["simulation_rows"])
    for log_message, field_name, summary_unit in (
        ("Minimum energy used by jobs %s.", "energy_min", None),
        ("Maximum energy used by jobs %s.", "energy_max", None),
        ("Minimum energy from configuration %s.", "configured_energy_min", None),
        ("Maximum energy from configuration %s.", "configured_energy_max", None),
        ("Minimum energy from lookup tables %s.", "energy_min_lookup_limit", u.GeV),
        ("Core scatter max used by jobs %s.", "core_scatter_max", None),
        ("Core scatter max from configuration %s.", "configured_core_scatter_max", None),
        ("Core scatter max from lookup tables %s.", "lookup_core_scatter_max", None),
        ("Minimum view cone used by jobs %s.", "view_cone_min", None),
        ("Maximum view cone used by jobs %s.", "view_cone_max", None),
        ("Minimum view cone from configuration %s.", "configured_view_cone_min", None),
        ("Maximum view cone from configuration %s.", "configured_view_cone_max", None),
        ("Maximum view cone from lookup tables %s.", "lookup_view_cone_max", None),
    ):
        logger.info(
            log_message, format_quantity_field(rows, field_name, summary_unit, log_wording=True)
        )
    logger.info(
        "Showers per job in generated job grid: minimum %d, maximum %d (configured maximum %d).",
        summary["showers_per_run_min"],
        summary["showers_per_run_max"],
        summary["showers_per_run_configured_max"],
    )
    logger.info("Total showers for the entire production: %d.", summary[TOTAL_SHOWERS_ARGUMENT])


def _resolve_nsb_rates_per_model_version(args_dict, model_versions, context=None):
    """Resolve NSB interpolation rate per model version."""
    return {mv: _resolve_nsb_rate(args_dict, mv, context=context) for mv in model_versions}


def _generate_observation_grids_per_layout(
    args_dict, grid_axes, nsb_rates_per_model_version=None, context=None
):
    """Generate observation grids per array layout.

    Uses either ProductionGridEngine or explicit azimuth/zenith axes.
    """
    resolved_layout_names = {
        model_version: resolve_array_layout_name(args_dict.get("array_layout_name"), model_version)
        for model_version in grid_axes["model_version"]
    }
    use_shared_axes_definition = bool(args_dict.get("axis"))
    corsika_limits_path = args_dict.get("corsika_limits")
    if nsb_rates_per_model_version is None:
        nsb_rates_per_model_version = _resolve_nsb_rates_per_model_version(
            args_dict, grid_axes["model_version"], context=context
        )

    observation_grids_per_model_version = {}
    observation_grid_cache = {}
    for model_version in grid_axes["model_version"]:
        resolved_layout_name = resolved_layout_names[model_version]
        nsb_rate = nsb_rates_per_model_version[model_version]
        cache_key = (resolved_layout_name, nsb_rate, use_shared_axes_definition)
        if cache_key not in observation_grid_cache:
            if use_shared_axes_definition:
                observation_grid_cache[cache_key] = build_production_grid_engine(
                    args_dict,
                    array_layout_name=resolved_layout_name,
                    model_version=model_version,
                    context=context,
                ).generate_simulation_grid()
            else:
                corsika_limits = (
                    CorsikaLimitsLookup(corsika_limits_path, array_layout_name=resolved_layout_name)
                    if corsika_limits_path is not None
                    else None
                )
                observation_grid_cache[cache_key] = _generate_observation_points_from_axes(
                    grid_axes["azimuth_angle"], grid_axes["zenith_angle"], corsika_limits, nsb_rate
                )
        observation_grids_per_model_version[model_version] = observation_grid_cache[cache_key]

    return observation_grids_per_model_version, resolved_layout_names


def _build_run_allocation(args_dict, total_showers, total_showers_scaling):
    """Build the run-count allocation config and validate run-count arguments."""
    if total_showers is not None and args_dict.get(NUMBER_OF_RUNS_ARGUMENT) is not None:
        raise ValueError("total_showers and number_of_runs cannot be configured together.")
    max_showers_per_run = args_dict.get(MAX_SHOWERS_PER_RUN_ARGUMENT)
    if max_showers_per_run is not None and total_showers is None:
        raise ValueError(f"{MAX_SHOWERS_PER_RUN_ARGUMENT} requires total_showers.")
    if max_showers_per_run is not None and int(max_showers_per_run) < 1:
        raise ValueError(f"{MAX_SHOWERS_PER_RUN_ARGUMENT} must be a positive integer.")

    return _RunAllocationConfig(
        total_showers=total_showers,
        total_showers_scaling=total_showers_scaling,
        zenith_angle_scaling_factor=float(
            args_dict.get(
                "zenith_angle_scaling_factor", defaults.ZENITH_ANGLE_SCALING_FACTOR_DEFAULT
            )
        ),
        max_showers_per_run=max_showers_per_run,
        rounding_warning_state={
            "emitted_warnings": 0,
            "max_warnings": max(
                0,
                int(
                    args_dict.get(
                        "max_total_showers_rounding_warnings",
                        TOTAL_SHOWERS_ROUNDING_WARNINGS_MAX_DEFAULT,
                    )
                ),
            ),
            "suppression_warning_emitted": False,
        },
    )


def build_simulation_jobs(args_dict, context=None):
    """
    Expand production config into full simulation job matrix.

    Cartesian product of primaries, model versions, interactions, observation directions,
    energy ranges, and run counts. Energy ranges are clipped by direction-dependent CORSIKA
    limits.

    Returns
    -------
    list[dict]
        Each job: primary, model_version, interactions, directions (Alt/Az), energy_min/max
        (clipped), showers_per_run, run_number, scatter/viewcone values
        (clipped by physics limits).
    """
    grid_axes = normalize_grid_axes(args_dict)
    energy_ranges = normalize_energy_ranges(args_dict["energy_range"])
    (
        showers_per_run,
        showers_per_run_power_law,
        showers_per_run_scaling,
        total_showers,
        total_showers_scaling,
    ) = _resolve_shower_params(args_dict)
    run_allocation = _build_run_allocation(args_dict, total_showers, total_showers_scaling)
    energy_max_scaling = _resolve_energy_max_scaling(args_dict)

    number_of_runs = int(args_dict.get(NUMBER_OF_RUNS_ARGUMENT) or 1)
    run_number = int(args_dict.get("run_number") or 1)
    core_scatter = args_dict["core_scatter"]
    view_cone = args_dict["view_cone"]
    corsika_hadronic_transition_energy = args_dict.get("corsika_hadronic_transition_energy")
    nsb_rates_per_model_version = _resolve_nsb_rates_per_model_version(
        args_dict, grid_axes["model_version"], context=context
    )
    observation_grids_per_model_version, resolved_layout_names = (
        _generate_observation_grids_per_layout(
            args_dict,
            grid_axes,
            nsb_rates_per_model_version=nsb_rates_per_model_version,
            context=context,
        )
    )
    logger.info(
        "Applying job constraints: energy clipped to configured energy_range, "
        "core_scatter_max clipped by configured max and lookup, and view_cone min/max "
        "clipped by configured and lookup limits."
    )
    _log_energy_scaling_configuration(energy_max_scaling)

    rows = []
    for primary, model_version, corsika_le, corsika_he in itertools.product(
        grid_axes["primary"],
        grid_axes["model_version"],
        grid_axes["corsika_le_interaction"],
        grid_axes["corsika_he_interaction"],
    ):
        for point in observation_grids_per_model_version[model_version]:
            lookup_core_scatter_max = point.get("upper_radius_limit")
            lookup_view_cone_max = point.get("viewcone_radius")
            selected_view_cone_max = _clip_max_quantity(view_cone[1], lookup_view_cone_max)
            observation_params = {
                "primary": primary,
                "azimuth_angle": point["azimuth"],
                "zenith_angle": point["zenith_angle"],
                "model_version": model_version,
                "nsb_rate": float(
                    point.get("nsb_rate", nsb_rates_per_model_version[model_version])
                ),
                "array_layout_name": resolved_layout_names[model_version],
                "corsika_le_interaction": corsika_le,
                "corsika_he_interaction": corsika_he,
                "cores_per_shower": int(core_scatter[0]),
                "core_scatter_max": _clip_max_quantity(core_scatter[1], lookup_core_scatter_max),
                "configured_core_scatter_max": core_scatter[1],
                "lookup_core_scatter_max": lookup_core_scatter_max,
                "view_cone_min": _clip_max_quantity(view_cone[0], selected_view_cone_max),
                "view_cone_max": selected_view_cone_max,
                "configured_view_cone_min": view_cone[0],
                "configured_view_cone_max": view_cone[1],
                "lookup_view_cone_max": lookup_view_cone_max,
            }
            for hadec_key in ("ha", "dec"):
                if point.get(hadec_key) is not None:
                    observation_params[hadec_key] = point[hadec_key]
            if corsika_hadronic_transition_energy is not None:
                observation_params["corsika_hadronic_transition_energy"] = (
                    corsika_hadronic_transition_energy
                )
            rows.extend(
                _build_rows_for_point(
                    point_base=observation_params,
                    energy_ranges=energy_ranges,
                    lower_energy_threshold=point.get(
                        "lower_energy_limit", point.get("br_energy_min")
                    ),
                    showers_per_run=showers_per_run,
                    showers_per_run_power_law=showers_per_run_power_law,
                    showers_per_run_scaling=showers_per_run_scaling,
                    number_of_runs=number_of_runs,
                    run_number=run_number,
                    run_allocation=run_allocation,
                    energy_max_scaling=energy_max_scaling,
                )
            )
    _log_generated_row_summary(rows)
    return rows


def renumber_job_rows(job_rows, run_number_offset):
    """
    Set output run numbers continuously.

    Parameters
    ----------
    job_rows : list[dict]
        List of job rows to renumber.
    run_number_offset : int
        Offset to apply to the run numbers (first run number will be offset + 1).

    Returns
    -------
    list[dict]
        List of job rows with updated run numbers.
    """
    for run_number, job_row in enumerate(job_rows, start=run_number_offset + 1):
        job_row["run_number"] = run_number
    return job_rows


def generate_job_grid(args_dict, output_file, metadata=None, context=None):
    """
    Generate and serialize a production job grid.

    Parameters
    ----------
    args_dict : dict
        Production job-grid configuration arguments.
    output_file : str or Path
        Path to the ECSV file to write.
    metadata : dict, optional
        Default metadata to include in the ECSV header.
    context : RunContext, optional
        Run configuration and model reader.
    """
    production_definition_from_arguments(args_dict)
    job_rows = renumber_job_rows(
        build_simulation_jobs(args_dict, context=context),
        run_number_offset=int(args_dict.get("run_number_offset", 0)),
    )
    serialize_job_grid(
        job_rows=job_rows,
        output_file=output_file,
        metadata=(metadata or {}) | build_job_grid_metadata(args_dict),
    )
