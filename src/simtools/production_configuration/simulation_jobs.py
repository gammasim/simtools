"""Expand observation grids into full simulation job matrices.

Combines grids (from ProductionGridEngine or explicit axes) with primaries, interactions,
model versions, energy ranges, and run counts into a complete job parameter set.
"""

import itertools
import logging
import shlex

import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

from simtools.configuration import defaults
from simtools.configuration.commandline_parser import CommandLineParser
from simtools.layout.array_layout_utils import resolve_array_layout_name
from simtools.model.site_model import SiteModel
from simtools.production_configuration.angle_ranges import (
    ceil_with_tolerance,
    directed_circular_span_degrees,
)
from simtools.production_configuration.corsika_limits_lookup import (
    CorsikaLimitsLookup,
    attach_lookup_limits_to_point,
)
from simtools.production_configuration.observation_grid import ProductionGridEngine
from simtools.utils.general import ensure_list

logger = logging.getLogger(__name__)

_GRID_AXES = [
    "primary",
    "azimuth_angle",
    "zenith_angle",
    "model_version",
    "corsika_le_interaction",
    "corsika_he_interaction",
]

_GRID_AXIS_DEFAULTS = {
    "corsika_le_interaction": defaults.CORSIKA_LE_INTERACTION,
    "corsika_he_interaction": defaults.CORSIKA_HE_INTERACTION,
}

GRID_AXIS_ARGUMENTS = {
    "azimuth": {
        "engine_axis": "azimuth",
        "unit": "deg",
        "help": "Azimuth range (deg)",
    },
    "zenith": {
        "engine_axis": "zenith_angle",
        "unit": "deg",
        "help": "Zenith angle range (deg)",
    },
    "ra": {
        "engine_axis": "ra",
        "unit": "deg",
        "help": "Right ascension range (deg)",
    },
    "dec": {
        "engine_axis": "dec",
        "unit": "deg",
        "help": "Declination range (deg)",
    },
    "nsb": {
        "engine_axis": "nsb_level",
        "unit": "MHz",
        "help": "NSB level range (MHz)",
    },
    "offset": {
        "engine_axis": "offset",
        "unit": "deg",
        "help": "Offset range (deg)",
    },
}

_AXIS_SCALING_CHOICES = ("linear", "log", "1/cos")
_HORIZONTAL_AXES = ("azimuth", "zenith")
_RADEC_AXES = ("ra", "dec")
_REQUIRED_AXES = ("nsb", "offset")
_LOCAL_CONSTRAINT_ARGUMENTS = {
    "local_zenith_range": "deg",
    "local_azimuth_range": "deg",
}
_DIRECTION_GRID_DENSITY_UNIT = 1 / u.deg**2


def _parse_axis_range_tokens(range_tokens):
    """Parse a quantity pair from CLI axis range tokens."""
    if len(range_tokens) == 1:
        return CommandLineParser.parse_quantity_pair(range_tokens[0])
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
    """Parse one compact axis definition."""
    tokens = _normalize_axis_spec_tokens(axis_spec)
    if len(tokens) < 3:
        raise ValueError(
            "Axis definitions require at least an axis name, range, and binning value."
        )

    axis_name = tokens[0]
    if axis_name not in GRID_AXIS_ARGUMENTS:
        supported_axes = ", ".join(sorted(GRID_AXIS_ARGUMENTS))
        raise ValueError(f"Unknown axis '{axis_name}'. Supported axes: {supported_axes}.")

    scaling = "linear"
    if tokens[-1] in _AXIS_SCALING_CHOICES:
        scaling = tokens[-1]
        binning_token = tokens[-2]
        range_tokens = tokens[1:-2]
    else:
        binning_token = tokens[-1]
        range_tokens = tokens[1:-1]

    if not range_tokens:
        raise ValueError(f"Axis '{axis_name}' is missing its range definition.")

    try:
        binning = int(binning_token)
    except ValueError as exc:
        raise ValueError(f"Axis '{axis_name}' binning must be an integer.") from exc

    axis_range = _parse_axis_range_tokens(range_tokens)
    axis_args = GRID_AXIS_ARGUMENTS[axis_name]
    return axis_name, {
        "range": [u.Quantity(value).to_value(axis_args["unit"]) for value in axis_range],
        "binning": binning,
        "scaling": scaling,
        "units": axis_args["unit"],
    }


def _iter_compact_axis_specs(args_dict):
    """Iterate over compact axis definitions from CLI or configuration."""
    axis_specs = args_dict.get("axis") or []
    if isinstance(axis_specs, str):
        axis_specs = [axis_specs]

    normalized_specs = []
    for axis_spec in axis_specs:
        if (
            isinstance(axis_spec, (list, tuple))
            and len(axis_spec) > 1
            and all(isinstance(item, str) for item in axis_spec)
            and str(axis_spec[0]).strip() not in GRID_AXIS_ARGUMENTS
        ):
            normalized_specs.extend(axis_spec)
            continue
        normalized_specs.append(axis_spec)
    return normalized_specs


def _resolve_axis_configs(args_dict):
    """Resolve compact axis definitions into one normalized mapping."""
    axis_configs = {}
    for axis_spec in _iter_compact_axis_specs(args_dict):
        axis_name, axis_config = _parse_axis_spec(axis_spec)
        axis_configs[axis_name] = axis_config

    return axis_configs


def _parse_optional_range_argument(range_argument_value, default_unit):
    """Parse optional quantity-pair CLI/config input into float values."""
    if range_argument_value is None:
        return None

    tokens = _normalize_axis_spec_tokens(range_argument_value)
    range_quantities = _parse_axis_range_tokens(tokens)
    return [u.Quantity(value).to_value(default_unit) for value in range_quantities]


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
            quantity = u.Quantity(" ".join(tokens))
            return quantity.to_value(_DIRECTION_GRID_DENSITY_UNIT)
        except (TypeError, ValueError, u.UnitConversionError) as exc:
            raise ValueError(
                "direction_grid_density must be a float or quantity in 1/deg^2."
            ) from exc

    raise TypeError("direction_grid_density must be a number, string, or CLI-style token list.")


def _mean_cosine_over_dec_span(dec_range):
    """Return mean cos(dec) over a declination interval (degrees)."""
    dec_min, dec_max = sorted((float(dec_range[0]), float(dec_range[1])))
    dec_span_rad = np.deg2rad(abs(dec_max - dec_min))
    if np.isclose(dec_span_rad, 0.0):
        return abs(np.cos(np.deg2rad(dec_min)))
    return abs((np.sin(np.deg2rad(dec_max)) - np.sin(np.deg2rad(dec_min))) / dec_span_rad)


def _mean_sine_over_zenith_span(zenith_range):
    """Return mean sin(zenith) over a zenith interval (degrees)."""
    zenith_min, zenith_max = sorted((float(zenith_range[0]), float(zenith_range[1])))
    zenith_span_rad = np.deg2rad(abs(zenith_max - zenith_min))
    if np.isclose(zenith_span_rad, 0.0):
        return abs(np.sin(np.deg2rad(zenith_min)))
    return abs((np.cos(np.deg2rad(zenith_min)) - np.cos(np.deg2rad(zenith_max))) / zenith_span_rad)


def _apply_direction_grid_density(axis_configs, direction_axes, density):
    """Derive direction-axis binning from density (points per deg^2)."""
    if density is None:
        return
    if density <= 0:
        raise ValueError("direction_grid_density must be strictly positive.")

    density_sqrt = np.sqrt(density)
    longitudinal_axis_scale = 1.0
    if tuple(direction_axes) == _RADEC_AXES:
        longitudinal_axis_scale = _mean_cosine_over_dec_span(axis_configs["dec"]["range"])
    elif tuple(direction_axes) == _HORIZONTAL_AXES:
        longitudinal_axis_scale = _mean_sine_over_zenith_span(axis_configs["zenith"]["range"])

    for axis_name in direction_axes:
        axis_range = axis_configs[axis_name]["range"]
        if axis_name == "azimuth":
            span_degrees = directed_circular_span_degrees(axis_range)
        else:
            span_degrees = abs(axis_range[1] - axis_range[0])

        if axis_name in ("azimuth", "ra"):
            span_degrees *= longitudinal_axis_scale

        axis_configs[axis_name]["binning"] = max(
            1,
            ceil_with_tolerance(span_degrees * density_sqrt),
        )


def _resolve_coordinate_system(axis_configs):
    """Resolve the coordinate system from axis definitions."""
    has_horizontal_axes = all(axis_name in axis_configs for axis_name in _HORIZONTAL_AXES)
    has_radec_axes = all(axis_name in axis_configs for axis_name in _RADEC_AXES)

    if has_horizontal_axes and has_radec_axes:
        raise ValueError("Cannot define both azimuth/zenith and ra/dec axes at the same time.")
    if has_radec_axes:
        return "ra_dec"
    if has_horizontal_axes:
        return "horizontal"
    return None


def _resolve_coordinate_system_from_args(args_dict):
    """Resolve the coordinate system from raw CLI arguments."""
    return _resolve_coordinate_system(_resolve_axis_configs(args_dict))


def resolve_time_of_observation(time_of_observation, args_dict):
    """Generate Time object for the observing time. Required if RA/Dec axes are present."""
    coordinate_system = _resolve_coordinate_system_from_args(args_dict)
    if coordinate_system == "ra_dec":
        if not time_of_observation:
            raise ValueError("time_of_observation is required when using RA/Dec axes.")
        return Time(time_of_observation, scale="utc")
    return None


def resolve_single_model_version(model_version):
    """Resolve one model version for helpers that require a scalar version."""
    if isinstance(model_version, list):
        return model_version[0]
    return model_version


def build_observing_location(site, model_version):
    """Build observing location from the site model."""
    site_model = SiteModel(model_version=resolve_single_model_version(model_version), site=site)
    return EarthLocation(
        lat=site_model.get_parameter_value_with_unit("reference_point_latitude"),
        lon=site_model.get_parameter_value_with_unit("reference_point_longitude"),
        height=site_model.get_parameter_value_with_unit("reference_point_altitude"),
    )


def build_axes_dict_from_cli_args(args_dict):
    """Build ProductionGridEngine-compatible axes configuration from CLI arguments."""
    axis_configs = _resolve_axis_configs(args_dict)
    coordinate_system = _resolve_coordinate_system(axis_configs)

    if coordinate_system is None:
        raise ValueError("Must provide either both azimuth/zenith or both ra/dec axis definitions.")

    missing_required_axes = [
        axis_name for axis_name in _REQUIRED_AXES if axis_name not in axis_configs
    ]
    if missing_required_axes:
        missing_axes = ", ".join(missing_required_axes)
        raise ValueError(f"Missing required shared axis definition(s): {missing_axes}.")

    direction_grid_density = _parse_direction_grid_density(args_dict.get("direction_grid_density"))
    direction_axes = _RADEC_AXES if coordinate_system == "ra_dec" else _HORIZONTAL_AXES
    if coordinate_system == "horizontal" and direction_grid_density is not None:
        axis_configs["azimuth"]["direction_grid_density"] = direction_grid_density
    if coordinate_system == "ra_dec" and direction_grid_density is not None:
        axis_configs["ra"]["direction_grid_density"] = direction_grid_density
    _apply_direction_grid_density(
        axis_configs,
        direction_axes,
        direction_grid_density,
    )

    if coordinate_system == "ra_dec":
        local_constraints = {}
        for constraint_argument, default_unit in _LOCAL_CONSTRAINT_ARGUMENTS.items():
            parsed_argument_range = _parse_optional_range_argument(
                args_dict.get(constraint_argument),
                default_unit,
            )
            if parsed_argument_range is not None:
                local_constraints[constraint_argument] = parsed_argument_range

        axis_configs["ra"].update(local_constraints)

    axes_to_export = [*_REQUIRED_AXES, *direction_axes]

    return {
        GRID_AXIS_ARGUMENTS[axis_name]["engine_axis"]: axis_configs[axis_name]
        for axis_name in axes_to_export
    }


def build_production_grid_engine(args_dict, array_layout_name=None):
    """Build a production-grid engine from application arguments."""
    axes = build_axes_dict_from_cli_args(args_dict)
    coordinate_system = _resolve_coordinate_system_from_args(args_dict)
    if coordinate_system == "ra_dec":
        observing_location = build_observing_location(
            site=args_dict["site"],
            model_version=args_dict["model_version"],
        )
    elif coordinate_system == "horizontal":
        coordinate_system = "horizontal"
        observing_location = None
    else:
        raise ValueError("Must provide either both azimuth/zenith or both ra/dec axis definitions.")
    resolved_layout_name = array_layout_name or resolve_array_layout_name(
        args_dict.get("array_layout_name"),
        resolve_single_model_version(args_dict.get("model_version")),
    )
    return ProductionGridEngine(
        axes=axes,
        coordinate_system=coordinate_system,
        observing_location=observing_location,
        time_of_observation=resolve_time_of_observation(
            args_dict.get("time_of_observation"),
            args_dict,
        ),
        lookup_table=args_dict.get("corsika_limits"),
        array_layout_name=resolved_layout_name,
    )


def build_job_grid_metadata(args_dict):
    """Build metadata stored alongside serialized executable job grids."""
    time_of_observation = resolve_time_of_observation(
        args_dict.get("time_of_observation"),
        args_dict,
    )
    coordinate_system = _resolve_coordinate_system_from_args(args_dict)
    direction_grid_density = _parse_direction_grid_density(args_dict.get("direction_grid_density"))
    if coordinate_system == "ra_dec" and not args_dict.get("site"):
        raise ValueError("site is required when using RA/Dec axes.")
    return {
        "site": args_dict.get("site"),
        "simulation_software": args_dict.get("simulation_software"),
        "coordinate_system": coordinate_system,
        "direction_grid_density": direction_grid_density,
        "direction_grid_density_unit": "1/deg^2" if direction_grid_density is not None else None,
        "time_of_observation_utc": time_of_observation.isot if time_of_observation else None,
        "time_of_observation_scale": time_of_observation.scale if time_of_observation else None,
        "corsika_limits": (
            str(args_dict["corsika_limits"]) if args_dict.get("corsika_limits") else None
        ),
    }


def normalize_grid_axes(args_dict):
    """Return normalized grid axes for cartesian product expansion."""
    return {
        axis: (
            ensure_list(args_dict[axis])
            if axis in args_dict and args_dict[axis] is not None
            else [_GRID_AXIS_DEFAULTS[axis]]
            if axis in _GRID_AXIS_DEFAULTS
            else [None]
        )
        for axis in _GRID_AXES
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


def get_energy_range_for_zenith_angle(
    zenith_angle, energy_range_pair, corsika_limits, azimuth_angle=None, nsb_level=1.0
):
    """
    Return a zenith-dependent energy range pair or ``None`` to skip the step.

    The lower energy bound is clipped to the lookup-table threshold for the
    requested direction. If the threshold exceeds the configured upper bound,
    the simulation step is skipped.
    """
    interpolated_limits = _interpolate_corsika_limits(
        corsika_limits, zenith_angle, azimuth_angle, nsb_level
    )
    if interpolated_limits is None:
        return energy_range_pair
    lower_energy_limit = interpolated_limits["lower_energy_limit"]
    if not isinstance(lower_energy_limit, u.Quantity):
        lower_energy_limit = lower_energy_limit * u.TeV
    return _clip_energy_range_from_threshold(
        energy_range_pair,
        lower_energy_limit,
    )


def get_core_scatter_max_for_zenith_angle(
    zenith_angle, core_scatter, corsika_limits, azimuth_angle=None, nsb_level=1.0
):
    """
    Return zenith-dependent max core-scatter value.

    The lookup-table scatter radius is treated as an upper limit and therefore
    clipped against the user-provided maximum value.
    """
    interpolated_limits = _interpolate_corsika_limits(
        corsika_limits, zenith_angle, azimuth_angle, nsb_level
    )
    if interpolated_limits is None:
        return core_scatter[1]
    upper_radius_limit = interpolated_limits["upper_radius_limit"]
    if not isinstance(upper_radius_limit, u.Quantity):
        upper_radius_limit = upper_radius_limit * u.m
    return _clip_max_quantity(core_scatter[1], upper_radius_limit)


def get_viewcone_max_for_zenith_angle(
    zenith_angle, view_cone, corsika_limits, azimuth_angle=None, nsb_level=1.0
):
    """Return zenith-dependent max viewcone value."""
    interpolated_limits = _interpolate_corsika_limits(
        corsika_limits, zenith_angle, azimuth_angle, nsb_level
    )
    if interpolated_limits is None:
        return view_cone[1]
    viewcone_radius = interpolated_limits["viewcone_radius"]
    if not isinstance(viewcone_radius, u.Quantity):
        viewcone_radius = viewcone_radius * u.deg
    return _clip_max_quantity(view_cone[1], viewcone_radius)


def calculate_log_energy_midpoint(energy_range_pair):
    """Return the geometric-mean energy for an energy range pair."""
    energy_min, energy_max = energy_range_pair

    if not isinstance(energy_min, u.Quantity) or not isinstance(energy_max, u.Quantity):
        raise TypeError("energy_range_pair must contain astropy Quantity values.")

    energy_min_tev = energy_min.to(u.TeV)
    energy_max_tev = energy_max.to(u.TeV)

    if energy_min_tev <= 0 * u.TeV or energy_max_tev <= 0 * u.TeV:
        raise ValueError("Energy range values must be strictly positive.")

    mean_log_energy = np.mean(
        [
            np.log10(energy_min_tev.value),
            np.log10(energy_max_tev.value),
        ]
    )
    return 10**mean_log_energy * u.TeV


def calculate_scaled_showers_per_run(
    energy_range_pair,
    baseline_showers_per_run,
    showers_per_run_power_law=None,
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
    zenith_angle,
    baseline_showers_per_run,
    showers_per_run_scaling="fixed",
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

    Raises
    ------
    ValueError
        If baseline showers per run is below 1, the selected scaling mode is unknown,
        or the scaled showers-per-run result is below 1.
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


def scale_energy_max_for_zenith_angle(
    zenith_angle,
    energy_range_pair,
    energy_max_scaling=None,
):
    """Scale energy_max with zenith angle and return an updated energy range pair."""
    if energy_max_scaling is None:
        return energy_range_pair

    energy_max_scaling_index, energy_max_scaling_reference = energy_max_scaling
    energy_min, energy_range_max = energy_range_pair
    cos_zenith = np.cos(zenith_angle.to(u.rad).value)

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
    """Clip selected energy bounds to configured energy-range bounds."""
    energy_min, energy_max = energy_range_pair
    configured_energy_min, configured_energy_max = configured_energy_range_pair

    selected_energy_min = max(
        energy_min.to(configured_energy_min.unit),
        configured_energy_min,
    )
    selected_energy_max = min(
        energy_max.to(configured_energy_max.unit),
        configured_energy_max,
    )

    if selected_energy_max <= selected_energy_min.to(selected_energy_max.unit):
        return None
    return selected_energy_min, selected_energy_max


def _interpolate_corsika_limits(corsika_limits, zenith_angle, azimuth_angle=None, nsb_level=1.0):
    """Return interpolated lookup limits for one pointing, or ``None`` if disabled."""
    if corsika_limits is None:
        return None
    if not isinstance(corsika_limits, CorsikaLimitsLookup):
        corsika_limits = CorsikaLimitsLookup(corsika_limits)
    return corsika_limits.interpolate_point(
        zenith_angle,
        0.0 * u.deg if azimuth_angle is None else azimuth_angle,
        nsb_level,
    )


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
    showers_per_run = args_dict["showers_per_run"]
    showers_per_run_power_law = args_dict.get("showers_per_run_power_law")
    showers_per_run_scaling = args_dict.get("showers_per_run_scaling", "fixed")
    total_showers = args_dict.get("total_showers")
    total_showers_scaling = args_dict.get("total_showers_scaling", "fixed")

    if showers_per_run_power_law is not None:
        showers_per_run_power_law = _parse_power_index_quantity(
            showers_per_run_power_law,
            "showers_per_run_power_law",
        )

    return (
        showers_per_run,
        showers_per_run_power_law,
        showers_per_run_scaling,
        total_showers,
        total_showers_scaling,
    )


def _resolve_energy_max_scaling(args_dict):
    """Resolve energy-max zenith scaling from CLI/config, including legacy options."""
    energy_max_scaling = args_dict.get("energy_max_scaling")
    legacy_energy_max_scaling_index = args_dict.get("energy_max_scaling_index")

    if energy_max_scaling is not None:
        if legacy_energy_max_scaling_index is not None:
            logger.warning(
                "Both energy_max_scaling and legacy energy_max_scaling_index were provided; "
                "energy_max_scaling takes precedence."
            )

        return _parse_power_index_quantity(energy_max_scaling, "energy_max_scaling")

    if legacy_energy_max_scaling_index is not None:
        return (float(legacy_energy_max_scaling_index), None)

    return None


def _scale_total_showers(
    total_showers,
    zenith_angle,
    total_showers_scaling,
    cos_scaling_factor=defaults.ZENITH_ANGLE_SCALING_FACTOR_DEFAULT,
):
    """
    Return total showers adjusted for the selected scaling mode.

    Scaling modes:

    - "fixed": total showers is unchanged.
    - "zenith_scaled": total showers scaled by 'total_showers * exp(factor * (cos(ZD) - 1))'
    """
    if total_showers_scaling == "fixed":
        return int(total_showers)
    if total_showers_scaling == "zenith_scaled":
        cos_zenith = np.cos(zenith_angle.to(u.rad).value)
        scaled_total_showers = total_showers * np.exp(cos_scaling_factor * (cos_zenith - 1))
        return int(np.ceil(scaled_total_showers))
    raise ValueError(f"Unknown total_showers_scaling mode: {total_showers_scaling}")


def _apply_clipping_chain(
    zenith_angle,
    energy_range_pair,
    energy_max_scaling,
    lower_energy_threshold,
):
    """Apply the full clipping chain to an energy range pair."""
    selected_energy_range = scale_energy_max_for_zenith_angle(
        zenith_angle,
        energy_range_pair,
        energy_max_scaling,
    )
    if selected_energy_range is None:
        return None
    selected_energy_range = _clip_energy_range_from_threshold(
        selected_energy_range, lower_energy_threshold
    )
    if selected_energy_range is None:
        return None
    return _clip_energy_range_to_configured_bounds(
        selected_energy_range,
        energy_range_pair,
    )


def _compute_per_point_runs(
    total_showers,
    zenith_angle,
    total_showers_scaling,
    selected_showers_per_run,
    zenith_angle_scaling_factor,
):
    """Compute the number of runs per point considering total-showers constraints."""
    effective_total_showers = _scale_total_showers(
        total_showers,
        zenith_angle,
        total_showers_scaling,
        cos_scaling_factor=zenith_angle_scaling_factor,
    )
    if effective_total_showers <= 0:
        return 0
    number_of_full_runs, remainder_showers = divmod(
        effective_total_showers, selected_showers_per_run
    )
    per_point_number_of_runs = number_of_full_runs + int(remainder_showers > 0)
    if remainder_showers > 0:
        adjusted_total_showers = per_point_number_of_runs * selected_showers_per_run
        logger.warning(
            "total_showers=%s is not divisible by showers_per_run=%s; "
            "adjusting to %s to keep equal showers per run.",
            effective_total_showers,
            selected_showers_per_run,
            adjusted_total_showers,
        )
    return per_point_number_of_runs


def _build_rows_for_point(
    point_base,
    energy_ranges,
    lower_energy_threshold,
    showers_per_run,
    showers_per_run_power_law,
    number_of_runs,
    total_showers,
    total_showers_scaling,
    run_number,
    showers_per_run_scaling="fixed",
    energy_max_scaling=None,
    zenith_angle_scaling_factor=defaults.ZENITH_ANGLE_SCALING_FACTOR_DEFAULT,
):
    """Build all simulation-run rows for a single grid point across all energy ranges."""
    rows = []
    zenith_angle = point_base["zenith_angle"]
    for energy_range_pair in energy_ranges:
        selected_energy_range = _apply_clipping_chain(
            zenith_angle,
            energy_range_pair,
            energy_max_scaling,
            lower_energy_threshold,
        )
        if selected_energy_range is None:
            continue
        selected_showers_per_run = calculate_scaled_showers_per_run(
            selected_energy_range,
            showers_per_run,
            showers_per_run_power_law,
        )
        selected_showers_per_run = calculate_zenith_scaled_showers_per_run(
            zenith_angle,
            selected_showers_per_run,
            showers_per_run_scaling,
        )

        per_point_number_of_runs = number_of_runs
        if total_showers is not None:
            per_point_number_of_runs = _compute_per_point_runs(
                total_showers,
                zenith_angle,
                total_showers_scaling,
                selected_showers_per_run,
                zenith_angle_scaling_factor,
            )

        for i in range(per_point_number_of_runs):
            rows.append(
                {
                    **point_base,
                    "energy_min": selected_energy_range[0],
                    "energy_max": selected_energy_range[1],
                    "showers_per_run": selected_showers_per_run,
                    "run_number": run_number + i,
                }
            )
    return rows


def _generate_observation_points_from_axes(azimuth_values, zenith_values, corsika_limits):
    """Sample azimuth * zenith grid with optional CORSIKA limits interpolation."""
    points = []
    for azimuth, zenith in itertools.product(azimuth_values, zenith_values):
        point = {
            "azimuth": azimuth,
            "zenith_angle": zenith,
        }
        if corsika_limits is not None:
            attach_lookup_limits_to_point(
                point,
                corsika_limits.interpolate_point(zenith, azimuth),
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


def _format_quantity_summary(quantity_values):
    """Format quantity min/max as a single value or range with explicit unit."""
    quantity_min = quantity_values.min()
    quantity_max = quantity_values.max()

    summary_unit = quantity_max.unit
    min_value = quantity_min.to_value(summary_unit)
    max_value = quantity_max.to_value(summary_unit)

    if np.isclose(min_value, max_value):
        return f"{max_value:.6g} {summary_unit}"
    return f"[{min_value:.6g}, {max_value:.6g}] {summary_unit}"


def _log_generated_row_summary(rows):
    """Log a compact summary of generated row ranges for user visibility."""
    if not rows:
        logger.info("Generated 0 simulation rows after applying all clipping and scaling rules.")
        return

    energy_min_values = u.Quantity([row["energy_min"] for row in rows])
    energy_max_values = u.Quantity([row["energy_max"] for row in rows])
    core_scatter_max_values = u.Quantity([row["core_scatter_max"] for row in rows])
    view_cone_min_values = u.Quantity([row["view_cone_min"] for row in rows])
    view_cone_max_values = u.Quantity([row["view_cone_max"] for row in rows])

    logger.info(
        "Generated %d simulation rows.",
        len(rows),
    )
    logger.info(
        "Energy range after clipping/scaling: Emin %s, Emax %s.",
        _format_quantity_summary(energy_min_values),
        _format_quantity_summary(energy_max_values),
    )
    logger.info(
        "Core scatter max range: %s.",
        _format_quantity_summary(core_scatter_max_values),
    )
    logger.info(
        "View cone range: min %s, max %s.",
        _format_quantity_summary(view_cone_min_values),
        _format_quantity_summary(view_cone_max_values),
    )


def _generate_observation_grids_per_layout(args_dict, grid_axes):
    """Generate observation grids per array layout.

    Uses either ProductionGridEngine or explicit azimuth/zenith axes.
    """
    observation_grids_per_layout = {}
    resolved_layout_names = {
        model_version: resolve_array_layout_name(args_dict.get("array_layout_name"), model_version)
        for model_version in grid_axes["model_version"]
    }
    use_shared_axes_definition = bool(args_dict.get("axis"))
    corsika_limits_path = args_dict.get("corsika_limits")

    for model_version in grid_axes["model_version"]:
        resolved_layout_name = resolved_layout_names[model_version]
        if resolved_layout_name in observation_grids_per_layout:
            continue

        if use_shared_axes_definition:
            observation_grids_per_layout[resolved_layout_name] = build_production_grid_engine(
                args_dict,
                array_layout_name=resolved_layout_name,
            ).generate_simulation_grid()
            continue

        corsika_limits = None
        if corsika_limits_path is not None:
            corsika_limits = CorsikaLimitsLookup(
                corsika_limits_path,
                array_layout_name=resolved_layout_name,
            )
        observation_grids_per_layout[resolved_layout_name] = _generate_observation_points_from_axes(
            azimuth_values=grid_axes["azimuth_angle"],
            zenith_values=grid_axes["zenith_angle"],
            corsika_limits=corsika_limits,
        )

    return observation_grids_per_layout, resolved_layout_names


def _build_observation_params_for_point(
    point,
    primary,
    model_version,
    resolved_layout_name,
    corsika_le,
    corsika_he,
    core_scatter,
    core_scatter_number,
    view_cone_min,
    configured_view_cone_max,
):
    """Build observation parameters and derived lookup-limited values for one grid point."""
    lookup_core_scatter_max = point.get("upper_radius_limit")
    lookup_view_cone_max = point.get("viewcone_radius")
    selected_core_scatter_max = _clip_max_quantity(core_scatter[1], lookup_core_scatter_max)
    selected_view_cone_max = _clip_max_quantity(configured_view_cone_max, lookup_view_cone_max)

    return {
        "primary": primary,
        "azimuth_angle": point["azimuth"],
        "zenith_angle": point["zenith_angle"],
        "ra": point.get("ra"),
        "dec": point.get("dec"),
        "model_version": model_version,
        "array_layout_name": resolved_layout_name,
        "corsika_le_interaction": corsika_le,
        "corsika_he_interaction": corsika_he,
        "core_scatter_number": core_scatter_number,
        "core_scatter_max": selected_core_scatter_max,
        "view_cone_min": _clip_max_quantity(view_cone_min, selected_view_cone_max),
        "view_cone_max": selected_view_cone_max,
    }


def build_simulation_jobs(args_dict):
    """
    Expand production config into full simulation job matrix.

    Cartesian product: primaries * model_versions * interactions * observation_directions
    * energy_ranges * run_counts. Energy ranges clipped by direction-dependent CORSIKA
    limits.

    Activates ProductionGridEngine when axis-range CLI arguments are provided;
    otherwise uses explicit azimuth * zenith axes.

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
    zenith_angle_scaling_factor = float(
        args_dict.get(
            "zenith_angle_scaling_factor",
            defaults.ZENITH_ANGLE_SCALING_FACTOR_DEFAULT,
        )
    )
    energy_max_scaling = _resolve_energy_max_scaling(args_dict)

    if total_showers is not None and args_dict.get("number_of_runs") is not None:
        raise ValueError("total_showers and number_of_runs cannot be configured together.")

    number_of_runs = int(args_dict.get("number_of_runs") or 1)
    run_number = int(args_dict.get("run_number") or 1)

    core_scatter = args_dict["core_scatter"]
    view_cone = args_dict["view_cone"]
    core_scatter_number = int(core_scatter[0])
    view_cone_min = view_cone[0]
    configured_view_cone_max = view_cone[1]
    observation_grids_per_layout, resolved_layout_names = _generate_observation_grids_per_layout(
        args_dict, grid_axes
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
        resolved_layout_name = resolved_layout_names[model_version]
        for point in observation_grids_per_layout[resolved_layout_name]:
            observation_params = _build_observation_params_for_point(
                point=point,
                primary=primary,
                model_version=model_version,
                resolved_layout_name=resolved_layout_name,
                corsika_le=corsika_le,
                corsika_he=corsika_he,
                core_scatter=core_scatter,
                core_scatter_number=core_scatter_number,
                view_cone_min=view_cone_min,
                configured_view_cone_max=configured_view_cone_max,
            )
            rows.extend(
                _build_rows_for_point(
                    point_base=observation_params,
                    energy_ranges=energy_ranges,
                    lower_energy_threshold=point.get(
                        "lower_energy_limit",
                        point.get("br_energy_min"),
                    ),
                    showers_per_run=showers_per_run,
                    showers_per_run_power_law=showers_per_run_power_law,
                    showers_per_run_scaling=showers_per_run_scaling,
                    number_of_runs=number_of_runs,
                    total_showers=total_showers,
                    total_showers_scaling=total_showers_scaling,
                    run_number=run_number,
                    energy_max_scaling=energy_max_scaling,
                    zenith_angle_scaling_factor=zenith_angle_scaling_factor,
                )
            )
    _log_generated_row_summary(rows)
    return rows
