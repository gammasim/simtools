#!/usr/bin/python3
"""Helper functions related to model parameters."""

import math

from simtools.model.calibration_model import CalibrationModel
from simtools.model.site_model import SiteModel
from simtools.model.telescope_model import TelescopeModel
from simtools.utils import names


def initialize_simulation_models(
    label,
    db_config,
    model_version,
    site,
    telescope_name,
    calibration_device_name=None,
    overwrite_model_parameters=None,
):
    """
    Initialize simulation models for a single telescope, site, and calibration device model.

    Parameters
    ----------
    label: str
        Label for the simulation.
    db_config: dict
        Database configuration.
    model_version: str
        Version of the simulation model
    site: str
        Name of the site.
    telescope_name: str
        Name of the telescope.
    calibration_device_name: str, optional
        Name of the calibration device.

    Returns
    -------
    Tuple
        Tuple containing the telescope site, (optional) calibration device model.
    """
    tel_model = TelescopeModel(
        site=site,
        telescope_name=telescope_name,
        db_config=db_config,
        model_version=model_version,
        label=label,
        overwrite_model_parameters=overwrite_model_parameters,
    )
    site_model = SiteModel(
        site=site,
        model_version=model_version,
        db_config=db_config,
        label=label,
        overwrite_model_parameters=overwrite_model_parameters,
    )
    if calibration_device_name is not None:
        calibration_model = CalibrationModel(
            site=site,
            calibration_device_model_name=calibration_device_name,
            db_config=db_config,
            model_version=model_version,
            label=label,
            overwrite_model_parameters=overwrite_model_parameters,
        )
    else:
        calibration_model = None
    for model in tel_model, site_model:
        model.export_model_files()
    return tel_model, site_model, calibration_model


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
