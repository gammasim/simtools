"""Expand observation grids into full simulation job matrices.

Combines grids (from ProductionGridEngine or explicit axes) with primaries, interactions,
model versions, energy ranges, and run counts into a complete job parameter set.
"""

import itertools
import shlex

import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

from simtools.configuration import defaults
from simtools.configuration.commandline_parser import CommandLineParser
from simtools.layout.array_layout_utils import resolve_array_layout_name
from simtools.model.site_model import SiteModel
from simtools.production_configuration.corsika_limits_lookup import (
    CorsikaLimitsLookup,
    attach_lookup_limits_to_point,
)
from simtools.production_configuration.observation_grid import ProductionGridEngine
from simtools.utils.general import ensure_list

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
DEFAULT_ZENITH_ANGLE_SCALING_FACTOR = 3.9781  #  derived by the LST team


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

    direction_axes = _RADEC_AXES if coordinate_system == "ra_dec" else _HORIZONTAL_AXES
    return {
        GRID_AXIS_ARGUMENTS[axis_name]["engine_axis"]: axis_configs[axis_name]
        for axis_name in (*_REQUIRED_AXES, *direction_axes)
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
    return {
        "site": args_dict.get("site"),
        "simulation_software": args_dict.get("simulation_software"),
        "coordinate_system": coordinate_system,
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
    return _clip_energy_range_from_threshold(
        energy_range_pair,
        interpolated_limits["lower_energy_threshold"] * u.TeV,
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
    return _clip_max_quantity(core_scatter[1], interpolated_limits["upper_scatter_radius"] * u.m)


def get_viewcone_max_for_zenith_angle(
    zenith_angle, view_cone, corsika_limits, azimuth_angle=None, nsb_level=1.0
):
    """Return zenith-dependent max viewcone value."""
    interpolated_limits = _interpolate_corsika_limits(
        corsika_limits, zenith_angle, azimuth_angle, nsb_level
    )
    if interpolated_limits is None:
        return view_cone[1]
    return _clip_max_quantity(view_cone[1], interpolated_limits["viewcone_radius"] * u.deg)


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
    showers_per_run_power_index=None,
    reference_energy=None,
):
    """Return an energy-dependent showers per run value."""
    if baseline_showers_per_run < 1:
        raise ValueError("baseline_showers_per_run must be a positive integer.")

    if showers_per_run_power_index is None:
        return baseline_showers_per_run

    if reference_energy is None:
        raise ValueError(
            "reference_energy is required when showers_per_run_power_index is configured."
        )

    midpoint_energy = calculate_log_energy_midpoint(energy_range_pair)
    scaling_factor = (midpoint_energy / reference_energy.to(midpoint_energy.unit)).to_value(
        u.dimensionless_unscaled
    ) ** showers_per_run_power_index
    scaled_showers_per_run = int(np.ceil(baseline_showers_per_run * scaling_factor))

    if scaled_showers_per_run < 1:
        raise ValueError("Scaled showers per run must be at least 1.")

    return scaled_showers_per_run


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


def _resolve_shower_params(args_dict):
    """Extract and convert shower-statistics parameters from an args dict."""
    showers_per_run = args_dict["showers_per_run"]
    showers_per_run_power_index = args_dict.get("showers_per_run_power_index")
    reference_energy = args_dict.get("showers_per_run_reference_energy")
    total_showers = args_dict.get("total_showers")
    total_showers_scaling = args_dict.get("total_showers_scaling", "fixed")

    if showers_per_run_power_index is not None and reference_energy is not None:
        reference_energy = u.Quantity(reference_energy)

    return (
        showers_per_run,
        showers_per_run_power_index,
        reference_energy,
        total_showers,
        total_showers_scaling,
    )


def _scale_total_showers(
    total_showers,
    zenith_angle,
    total_showers_scaling,
    cos_scaling_factor=DEFAULT_ZENITH_ANGLE_SCALING_FACTOR,
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


def _build_rows_for_point(
    point_base,
    energy_ranges,
    lower_energy_threshold,
    showers_per_run,
    showers_per_run_power_index,
    reference_energy,
    number_of_runs,
    total_showers,
    total_showers_scaling,
    run_number,
    zenith_angle_scaling_factor=DEFAULT_ZENITH_ANGLE_SCALING_FACTOR,
):
    """Build all simulation-run rows for a single grid point across all energy ranges."""
    rows = []
    for energy_range_pair in energy_ranges:
        selected_energy_range = _clip_energy_range_from_threshold(
            energy_range_pair, lower_energy_threshold
        )
        if selected_energy_range is None:
            continue
        selected_showers_per_run = calculate_scaled_showers_per_run(
            selected_energy_range,
            showers_per_run,
            showers_per_run_power_index,
            reference_energy,
        )

        per_point_number_of_runs = number_of_runs
        if total_showers is not None:
            effective_total_showers = _scale_total_showers(
                total_showers,
                point_base["zenith_angle"],
                total_showers_scaling,
                cos_scaling_factor=zenith_angle_scaling_factor,
            )
            if effective_total_showers <= 0:
                continue
            number_of_full_runs, remainder_showers = divmod(
                effective_total_showers, selected_showers_per_run
            )
            per_point_number_of_runs = number_of_full_runs + int(remainder_showers > 0)

        for i in range(per_point_number_of_runs):
            run_showers_per_run = selected_showers_per_run
            if total_showers is not None and i >= number_of_full_runs:
                run_showers_per_run = remainder_showers
            rows.append(
                {
                    **point_base,
                    "energy_min": selected_energy_range[0],
                    "energy_max": selected_energy_range[1],
                    "showers_per_run": run_showers_per_run,
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
            attach_lookup_limits_to_point(point, corsika_limits.interpolate_point(zenith, azimuth))
        points.append(point)
    return points


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
        showers_per_run_power_index,
        reference_energy,
        total_showers,
        total_showers_scaling,
    ) = _resolve_shower_params(args_dict)
    zenith_angle_scaling_factor = float(
        args_dict.get("zenith_angle_scaling_factor", DEFAULT_ZENITH_ANGLE_SCALING_FACTOR)
    )

    if total_showers is not None and args_dict.get("number_of_runs") is not None:
        raise ValueError("total_showers and number_of_runs cannot be configured together.")

    number_of_runs = int(args_dict.get("number_of_runs") or 1)
    run_number = int(args_dict.get("run_number") or 1)

    core_scatter = args_dict["core_scatter"]
    view_cone = args_dict["view_cone"]
    core_scatter_number = int(core_scatter[0])
    view_cone_min = view_cone[0]
    observation_grids_per_layout, resolved_layout_names = _generate_observation_grids_per_layout(
        args_dict, grid_axes
    )
    rows = []

    for primary, model_version, corsika_le, corsika_he in itertools.product(
        grid_axes["primary"],
        grid_axes["model_version"],
        grid_axes["corsika_le_interaction"],
        grid_axes["corsika_he_interaction"],
    ):
        resolved_layout_name = resolved_layout_names[model_version]
        for point in observation_grids_per_layout[resolved_layout_name]:
            observation_params = {
                "primary": primary,
                "azimuth_angle": point["azimuth"],
                "zenith_angle": point["zenith_angle"],
                "model_version": model_version,
                "array_layout_name": resolved_layout_name,
                "corsika_le_interaction": corsika_le,
                "corsika_he_interaction": corsika_he,
                "core_scatter_number": core_scatter_number,
                "core_scatter_max": _clip_max_quantity(
                    core_scatter[1], point.get("scatter_radius")
                ),
                "view_cone_min": view_cone_min,
                "view_cone_max": _clip_max_quantity(view_cone[1], point.get("viewcone_radius")),
            }
            rows.extend(
                _build_rows_for_point(
                    point_base=observation_params,
                    energy_ranges=energy_ranges,
                    lower_energy_threshold=point.get("lower_energy_threshold"),
                    showers_per_run=showers_per_run,
                    showers_per_run_power_index=showers_per_run_power_index,
                    reference_energy=reference_energy,
                    number_of_runs=number_of_runs,
                    total_showers=total_showers,
                    total_showers_scaling=total_showers_scaling,
                    run_number=run_number,
                    zenith_angle_scaling_factor=zenith_angle_scaling_factor,
                )
            )
    return rows
