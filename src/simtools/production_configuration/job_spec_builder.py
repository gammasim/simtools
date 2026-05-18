"""Build backend-agnostic job specifications for production submissions."""

import numpy as np
from astropy import units as u

from simtools.production_configuration.production_grid_job_rows import (
    build_backend_agnostic_job_rows,
    get_core_scatter_max_for_zenith_angle,
    get_energy_range_for_zenith_angle,
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


def build_job_specs(args_dict, image_labels):
    """Build backend-agnostic job specs from comparison and production grids."""
    base_pack_dir = args_dict.get("simulation_output") or "simtools-output"
    normalized_rows = build_backend_agnostic_job_rows(
        args_dict,
        calculate_scaled_nshow,
        get_energy_range_for_zenith_angle_function=get_energy_range_for_zenith_angle,
        get_core_scatter_max_for_zenith_angle_function=get_core_scatter_max_for_zenith_angle,
    )

    job_specs = []
    for label in image_labels:
        for row in normalized_rows:
            job_specs.append(
                {
                    "image_label": str(label),
                    **row,
                    "pack_for_grid_register": f"{base_pack_dir}/{label!s}",
                }
            )
    return job_specs
