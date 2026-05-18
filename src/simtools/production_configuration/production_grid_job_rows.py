"""
Builds backend-agnostic job row dictionaries for simulation production grids.

A job row is a configuration dictionary for a single simulation job/run, containing all
parameters needed to launch that simulation (e.g., primary, zenith, azimuth, energy range,
model version, etc.).
"""

import itertools

from astropy import units as u

from simtools.production_configuration.corsika_limits_lookup import CorsikaLimitsLookup
from simtools.production_configuration.production_grid_helpers import (
    normalize_energy_ranges,
    normalize_grid_axes,
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


def build_backend_agnostic_job_rows(
    args_dict,
    calculate_scaled_nshow,
    get_energy_range_for_zenith_angle_function=get_energy_range_for_zenith_angle,
    get_core_scatter_max_for_zenith_angle_function=get_core_scatter_max_for_zenith_angle,
):
    """
    Build normalized production-grid rows for backend consumers.

    Parameters
    ----------
    args_dict : dict
        Production-job configuration.
    calculate_scaled_nshow : callable
        Callback used to compute the per-row ``nshow`` value.
    get_energy_range_for_zenith_angle_function : callable, optional
        Callback used to derive the direction-dependent energy range.
    get_core_scatter_max_for_zenith_angle_function : callable, optional
        Callback used to derive the direction-dependent core-scatter radius.

    Returns
    -------
    list[dict]
        Backend-independent row dictionaries.
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
        selected_energy_range_pair = get_energy_range_for_zenith_angle_function(
            zenith,
            energy_range_pair,
            corsika_limits,
            azimuth_angle=azimuth,
        )
        if selected_energy_range_pair is None:
            continue

        selected_core_scatter_max = get_core_scatter_max_for_zenith_angle_function(
            zenith,
            core_scatter,
            corsika_limits,
            azimuth_angle=azimuth,
        )
        selected_nshow = calculate_scaled_nshow(
            selected_energy_range_pair, nshow, nshow_power_index, reference_energy
        )

        for _ in range(number_of_runs):
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
            row_index += 1
    return rows
