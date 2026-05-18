"""Helper functions and constants for production-grid generation."""

import numpy as np
from astropy import units as u

from simtools.configuration import defaults
from simtools.utils.general import ensure_list

DEFAULT_SERIALIZATION_ROUND_DECIMALS = 6

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


def normalize_azimuth_angle(azimuth_angle):
    """Return an azimuth angle with degree units and modulo-360 normalization."""
    # TODO - duplication?
    if isinstance(azimuth_angle, u.Quantity):
        return azimuth_angle.to(u.deg) % (360 * u.deg)
    return azimuth_angle % 360.0


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
