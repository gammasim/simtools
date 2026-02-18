#!/usr/bin/python3
"""Helper functions related to model parameters."""

import math

from simtools import settings
from simtools.data_model import schema
from simtools.io import ascii_handler
from simtools.model.calibration_model import CalibrationModel
from simtools.model.site_model import SiteModel
from simtools.model.telescope_model import TelescopeModel
from simtools.utils import names


def initialize_simulation_models(
    label,
    model_version,
    site,
    telescope_name,
    calibration_device_name=None,
    calibration_device_type=None,
):
    """
    Initialize simulation models for a single telescope, site, and calibration device model.

    Parameters
    ----------
    label: str
        Label for the simulation.
    model_version: str
        Version of the simulation model
    site: str
        Name of the site.
    telescope_name: str
        Name of the telescope.
    calibration_device_name: str, optional
        Name of the calibration device.
    calibration_device_type: str, optional
        Type of the calibration device.

    Returns
    -------
    Tuple
        Tuple containing the telescope site, (optional) calibration device model.
    """
    overwrite_model_parameter_dict = read_overwrite_model_parameter_dict()
    common = {
        "site": site,
        "model_version": model_version,
        "label": label,
        "overwrite_model_parameter_dict": overwrite_model_parameter_dict,
    }

    tel_model = TelescopeModel(telescope_name=telescope_name, **common)
    site_model = SiteModel(**common)

    calibration_device_name = calibration_device_name or tel_model.get_calibration_device_name(
        calibration_device_type
    )
    if calibration_device_name is not None:
        calibration_model = CalibrationModel(
            calibration_device_model_name=calibration_device_name, **common
        )
    else:
        calibration_model = None
    for model in tel_model, site_model:
        model.export_model_files()
    return tel_model, site_model, calibration_model


def read_overwrite_model_parameter_dict(overwrite_model_parameters=None):
    """
    Read overwrite model parameters dictionary from file.

    Parameters
    ----------
    overwrite_model_parameters: str, optional
        File name with overwrite model parameters.

    Returns
    -------
    dict
        Dictionary with model parameters to overwrite.
    """
    overwrite_model_parameter_dict = {}
    overwrite_model_parameters = overwrite_model_parameters or settings.config.args.get(
        "overwrite_model_parameters"
    )
    if overwrite_model_parameters is not None:
        overwrite_model_parameter_dict = schema.validate_dict_using_schema(
            data=ascii_handler.collect_data_from_file(file_name=overwrite_model_parameters),
            schema_file="simulation_models_info.schema.yml",
        ).get("changes", {})

    return overwrite_model_parameter_dict


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


def get_array_elements_for_layout(layout_name, site=None, model_version=None):
    """
    Get array elements for a given array layout.

    Parameters
    ----------
    layout_name: str
        Name of the array layout.
    site: str, optional
        Site name (use central configuration if not provided).
    model_version: str, optional
        Model version (use central configuration if not provided).

    Returns
    -------
    list
        List of array elements for the given array layout.
    """
    if not layout_name or (isinstance(layout_name, list) and len(layout_name) > 1):
        raise ValueError("Single array layout name must be provided.")
    layout_name = layout_name[0] if isinstance(layout_name, list) else layout_name
    site_model = SiteModel(
        site=site or settings.config.args.get("site"),
        model_version=model_version or settings.config.args.get("model_version"),
        label="label",
        overwrite_model_parameter_dict=read_overwrite_model_parameter_dict(),
    )
    return site_model.get_array_elements_for_layout(layout_name)
