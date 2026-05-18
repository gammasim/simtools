"""Build backend-agnostic job specifications for production submissions."""

import ast
import itertools

import numpy as np
from astropy import units as u

import simtools.version as simtools_version
from simtools.configuration import defaults

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


def normalize_to_list(value):
    """Normalize scalar values to lists of length one."""
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def normalize_grid_axes(args_dict):
    """Return normalized grid axes for cartesian product expansion."""
    return {
        axis: (
            normalize_to_list(args_dict[axis])
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


def resolve_array_layout_name(array_layout_name, model_version):
    """Resolve array layout configuration for a specific model version."""
    if isinstance(array_layout_name, list) and len(array_layout_name) == 1:
        array_layout_name = array_layout_name[0]

    if isinstance(array_layout_name, str) and array_layout_name.strip().startswith("{"):
        try:
            parsed_layout = ast.literal_eval(array_layout_name)
            if isinstance(parsed_layout, dict):
                array_layout_name = parsed_layout
        except (SyntaxError, ValueError):
            return array_layout_name

    if not isinstance(array_layout_name, dict) or list(array_layout_name) != ["by_version"]:
        return array_layout_name

    resolved = simtools_version.resolve_by_version(
        {"array_layout_name": array_layout_name}, model_version
    )
    return resolved["array_layout_name"]


def get_energy_range_for_zenith_angle(zenith_angle, energy_range_pair, corsika_limits):
    """Return a zenith-dependent energy range pair or None to skip the simulation step."""
    _ = (zenith_angle, corsika_limits)
    return energy_range_pair


def get_core_scatter_max_for_zenith_angle(zenith_angle, core_scatter, corsika_limits):
    """Return zenith-dependent max core-scatter value."""
    _ = (zenith_angle, corsika_limits)
    return core_scatter[1]


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


def calculate_scaled_nshow(
    energy_range_pair,
    baseline_nshow,
    nshow_power_index=None,
    reference_energy=None,
):
    """Return an energy-dependent nshow value."""
    if baseline_nshow < 1:
        raise ValueError("baseline_nshow must be a positive integer.")

    if nshow_power_index is None:
        return baseline_nshow

    if reference_energy is None:
        raise ValueError("reference_energy is required when nshow_power_index is configured.")

    midpoint_energy = calculate_log_energy_midpoint(energy_range_pair)
    scaling_factor = (midpoint_energy / reference_energy.to(midpoint_energy.unit)).to_value(
        u.dimensionless_unscaled
    ) ** nshow_power_index
    scaled_nshow = int(np.ceil(baseline_nshow * scaling_factor))

    if scaled_nshow < 1:
        raise ValueError("Scaled nshow must be at least 1.")

    return scaled_nshow


def _select_energy_and_core_scatter_for_job(
    zenith, energy_range_pair, core_scatter, corsika_limits
):
    """Return selected energy range and core scatter maximum for a job spec row."""
    selected_energy_range_pair = energy_range_pair
    selected_core_scatter_max = core_scatter[1]

    if corsika_limits is None:
        return selected_energy_range_pair, selected_core_scatter_max

    selected_energy_range_pair = get_energy_range_for_zenith_angle(
        zenith, energy_range_pair, corsika_limits
    )
    if selected_energy_range_pair is None:
        return None, None

    selected_core_scatter_max = get_core_scatter_max_for_zenith_angle(
        zenith, core_scatter, corsika_limits
    )
    return selected_energy_range_pair, selected_core_scatter_max


def build_job_specs(args_dict, image_labels):
    """Build backend-agnostic job specs from comparison and production grids."""
    grid_axes = normalize_grid_axes(args_dict)
    energy_ranges = normalize_energy_ranges(args_dict["energy_range"])
    base_pack_dir = args_dict.get("simulation_output") or "simtools-output"
    corsika_limits = args_dict.get("corsika_limits")
    core_scatter = args_dict["core_scatter"]
    nshow = args_dict["nshow"]
    nshow_power_index = args_dict.get("nshow_power_index")
    reference_energy = args_dict.get("nshow_reference_energy")
    if nshow_power_index is not None and reference_energy is not None:
        reference_energy = u.Quantity(reference_energy)

    combinations = list(
        itertools.product(
            grid_axes["primary"],
            grid_axes["azimuth_angle"],
            grid_axes["zenith_angle"],
            grid_axes["model_version"],
            grid_axes["corsika_le_interaction"],
            grid_axes["corsika_he_interaction"],
            energy_ranges,
        )
    )

    number_of_runs = args_dict.get("number_of_runs", 1)
    run_number = int(args_dict.get("run_number") or 1)

    job_specs = []
    for label in image_labels:
        row_index = 0
        for (
            primary,
            azimuth,
            zenith,
            model_version,
            corsika_le,
            corsika_he,
            energy_range_pair,
        ) in combinations:
            (
                selected_energy_range_pair,
                selected_core_scatter_max,
            ) = _select_energy_and_core_scatter_for_job(
                zenith, energy_range_pair, core_scatter, corsika_limits
            )
            if selected_energy_range_pair is None:
                continue

            selected_nshow = calculate_scaled_nshow(
                selected_energy_range_pair, nshow, nshow_power_index, reference_energy
            )

            for _ in range(number_of_runs):
                job_specs.append(
                    {
                        "image_label": str(label),
                        "primary": primary,
                        "azimuth_angle": azimuth,
                        "zenith_angle": zenith,
                        "model_version": model_version,
                        "array_layout_name": args_dict.get("array_layout_name"),
                        "corsika_le_interaction": corsika_le,
                        "corsika_he_interaction": corsika_he,
                        "energy_min": selected_energy_range_pair[0],
                        "energy_max": selected_energy_range_pair[1],
                        "core_scatter_max": selected_core_scatter_max,
                        "nshow": selected_nshow,
                        "pack_for_grid_register": f"{base_pack_dir}/{label!s}",
                        "run_number": run_number + row_index,
                    }
                )
                row_index += 1
    return job_specs
