#!/usr/bin/python3

import logging
import math

from simtools.utils import names

__all__ = [
    "compute_telescope_transmission",
    "get_telescope_class",
    "get_camera_name",
    "is_two_mirror_telescope",
    "split_simtel_parameter",
]


def split_simtel_parameter(value):
    """
    Some array parameters are stored in sim_telarray model as string separated by comma or spaces.\
    This functions turns this string into a list of floats. The delimiter is identified \
    automatically.

    Parameters
    ----------
    value: str
        String with the array of floats separated by comma or spaces.

    Returns
    -------
    list
        Array of floats.
    """

    delimiter = "," if "," in value else " "
    float_values = [float(v) for v in value.split(delimiter)]
    return float_values


def compute_telescope_transmission(pars, off_axis):
    """
    Compute telescope transmission (0 < T < 1) for a given set of parameters as defined by \
    the MC model and for a given off-axis angle.

    Parameters
    ----------
    pars: list of float
        Parameters of the telescope transmission. Len(pars) should be 4.
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


def get_camera_name(telescope_model_name):
    """
    Get camera name from the telescope name.

    Parameters
    ----------
    telescope_model_name: str
        Telescope model name (e.g., LST-1).

    Returns
    -------
    str
        Camera name (validated by util.names).
    """

    _logger = logging.getLogger(__name__)
    camera_name = ""
    tel_class, tel_type = names.split_telescope_model_name(telescope_model_name)
    if tel_class == "LST":
        camera_name = "LST"
    elif tel_class == "MST":
        if "FlashCam" in tel_type:
            camera_name = "FlashCam"
        elif "NectarCam" in tel_type:
            camera_name = "NectarCam"
        else:
            _logger.error("Camera not found for MST class telescope")
    elif tel_class == "SCT":
        camera_name = "SCT"
    elif tel_class == "SST":
        if "ASTRI" in tel_type:
            camera_name = "ASTRI"
        elif "GCT" in tel_type:
            camera_name = "GCT"
        elif "1M" in tel_type:
            camera_name = "1M"
        else:
            camera_name = "SST"
    else:
        _logger.error("Invalid telescope name - please validate it first")

    camera_name = names.validate_camera_name(camera_name)
    _logger.debug(f"Camera name - {camera_name}")
    return camera_name


def get_telescope_class(telescope_model_name):
    """
    Get telescope class from telescope name.

    Parameters
    ----------
    telescope_model_name: str
        Telescope model name (ex. LST-1).

    Returns
    -------
    str
        Telescope class (SST, MST, ...).
    """

    tel_class, _ = names.split_telescope_model_name(telescope_model_name)
    return tel_class


def is_two_mirror_telescope(telescope_model_name):
    """
    Check if the telescope is a two mirror design.

    Parameters
    ----------
    telescope_model_name: str
        Telescope model name (ex. LST-1).

    Returns
    -------
    bool
        True if the telescope is a two mirror one.
    """
    tel_class, tel_type = names.split_telescope_model_name(telescope_model_name)
    if tel_class == "SST":
        # Only 1M is False
        return "1M" not in tel_type
    if tel_class == "SCT":
        # SCT always two mirrors
        return True

    # All MSTs and LSTs
    return False
