"""Build simulation execution grid based on the production configuration and lookup tables."""

import itertools

import numpy as np
from astropy import units as u

from simtools.configuration import defaults
from simtools.production_configuration.corsika_limits_lookup import CorsikaLimitsLookup
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
    if corsika_limits is None:
        return energy_range_pair

    if not isinstance(corsika_limits, CorsikaLimitsLookup):
        corsika_limits = CorsikaLimitsLookup(corsika_limits)

    azimuth_angle = 0.0 * u.deg if azimuth_angle is None else azimuth_angle
    interpolated_limits = corsika_limits.interpolate_point(zenith_angle, azimuth_angle, nsb_level)
    lower_energy_threshold = interpolated_limits["lower_energy_threshold"] * u.TeV

    energy_min, energy_max = energy_range_pair
    if lower_energy_threshold > energy_max.to(lower_energy_threshold.unit):
        return None
    if lower_energy_threshold <= energy_min.to(lower_energy_threshold.unit):
        return energy_range_pair
    return lower_energy_threshold.to(energy_min.unit), energy_max


def get_core_scatter_max_for_zenith_angle(
    zenith_angle, core_scatter, corsika_limits, azimuth_angle=None, nsb_level=1.0
):
    """
    Return zenith-dependent max core-scatter value.

    The lookup-table scatter radius is treated as an upper limit and therefore
    clipped against the user-provided maximum value.
    """
    if corsika_limits is None:
        return core_scatter[1]

    if not isinstance(corsika_limits, CorsikaLimitsLookup):
        corsika_limits = CorsikaLimitsLookup(corsika_limits)

    azimuth_angle = 0.0 * u.deg if azimuth_angle is None else azimuth_angle
    interpolated_limits = corsika_limits.interpolate_point(zenith_angle, azimuth_angle, nsb_level)
    lookup_scatter_max = interpolated_limits["upper_scatter_radius"] * u.m
    configured_scatter_max = core_scatter[1]
    return min(configured_scatter_max, lookup_scatter_max.to(configured_scatter_max.unit))


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


def build_simulation_jobs(args_dict):
    """
    Build simulation job parameters based on the production configuration and lookup tables.

    Each entry corresponds to a single simulation job/run and contains all parameters needed to
    launch that simulation (e.g., primary, zenith, azimuth, energy range, model version, etc.).
    The energy range and core-scatter radius can be adjusted based on the direction-dependent
    lookup-table limits. The number of events (``nshow``) can be scaled with energy according
    to a configurable power law.

    Parameters
    ----------
    args_dict : dict
        Production-job configuration.

    Returns
    -------
    list[dict]
        Simulation job parameter dictionaries for all combinations of configured axes.
    """
    grid_axes = normalize_grid_axes(args_dict)
    energy_ranges = normalize_energy_ranges(args_dict["energy_range"])
    corsika_limits = args_dict.get("corsika_limits")
    if corsika_limits is not None:
        corsika_limits = CorsikaLimitsLookup(
            corsika_limits,
            telescope_ids=args_dict.get("telescope_ids"),
            simtel_file=args_dict.get("simtel_file"),
        )

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

    rows = []
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
        selected_energy_range_pair = get_energy_range_for_zenith_angle(
            zenith,
            energy_range_pair,
            corsika_limits,
            azimuth_angle=azimuth,
        )
        if selected_energy_range_pair is None:
            continue

        selected_core_scatter_max = get_core_scatter_max_for_zenith_angle(
            zenith,
            core_scatter,
            corsika_limits,
            azimuth_angle=azimuth,
        )
        selected_nshow = calculate_scaled_nshow(
            selected_energy_range_pair, nshow, nshow_power_index, reference_energy
        )

        for row_index in range(number_of_runs):
            rows.append(
                {
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
                    "run_number": run_number + row_index,
                }
            )
    return rows
