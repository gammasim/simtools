#!/usr/bin/python3
"""Helper functions calculations related to model parameters."""

import math

from simtools.utils import names

__all__ = [
    "compute_telescope_transmission",
    "is_two_mirror_telescope",
]


def compute_telescope_transmission(pars: list[float], off_axis: float) -> float:
    """
    Compute telescope transmission (0 < T < 1) for a given off-axis angle.

    The telescope transmission depends on the MC model used.

    Parameters
    ----------
    pars: list of float
        Parameters of the telescope transmission. Len(pars) should be 5 or 6.
    off_axis: float
        Off-axis angle in deg.

    Returns
    -------
    float
        Telescope transmission.
    """
    _deg_to_rad = math.pi / 180.0
    if pars[1] == 0:
        return pars[0]

    t = math.sin(off_axis * _deg_to_rad) / (pars[3] * _deg_to_rad)
    return pars[0] / (1.0 + pars[2] * t ** pars[4])


def is_two_mirror_telescope(telescope_model_name: str) -> bool:
    """
    Determine if the telescope model is a two-mirror telescope.

    Parameters
    ----------
    telescope_model_name: str
        Name of the telescope model.

    Returns
    -------
    bool
        True if it is a two-mirror telescope.
    """
    tel_type = names.get_array_element_type_from_name(telescope_model_name)
    if "SST" in tel_type or "SCT" in tel_type:
        return True
    return False
