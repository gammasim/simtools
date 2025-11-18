#!/usr/bin/python3
"""
Handling of legacy model parameters for backward compatibility.

Collects routines to convert legacy model parameters to more recent
formats. This is a fine-tuned process and requires to hard wire the
changes. All code related to legacy model parameters should go into
this module.

"""

import logging

logger = logging.getLogger(__name__)


def apply_legacy_updates_to_parameters(parameters, _legacy_updates):
    """Apply legacy updates to model parameters.

    Parameters
    ----------
    parameters: dict
        Dictionary of model parameters (all parameters).
    _legacy_updates: dict
        Dictionary of legacy updates to apply.

    Returns
    -------
    dict
        Updated model parameters.
    """
    for par_name, legacy_data in _legacy_updates.items():
        if legacy_data is None or par_name not in parameters:
            continue
        for key in parameters[par_name].keys():
            if key in legacy_data:
                parameters[par_name][key] = legacy_data[key]
        if legacy_data.get("remove_parameter", False):
            parameters.pop(par_name)

    return parameters


def update_parameter(par_name, parameters, schema_version):
    """Update legacy model parameters to recent formats.

    Parameters
    ----------
    par_name: str
        Model parameter name.
    parameters: dict
        Dictionary of model parameters (all parameters).
    schema_version: str
        Target schema version.

    Returns
    -------
    None
    """
    try:
        return globals()[f"_update_{par_name}"](par_name, parameters, schema_version)
    except KeyError as exc:
        raise ValueError(
            _get_unsupported_update_message(parameters[par_name], schema_version)
        ) from exc


def _update_dsum_threshold(par_name, parameters, schema_version):
    """Update legacy dsum_threshold parameter."""
    para_data = parameters[par_name]
    if para_data["model_parameter_schema_version"] == "0.1.0" and schema_version == "0.2.0":
        logger.info(
            f"Updating legacy model parameter {para_data['parameter']} from schema version "
            f"{para_data['model_parameter_schema_version']} to {schema_version}"
        )
        return {
            para_data["parameter"]: {
                "value": int(para_data["value"]),
                "model_parameter_schema_version": schema_version,
            }
        }
    raise ValueError(_get_unsupported_update_message(para_data, schema_version))


def _update_corsika_starting_grammage(par_name, parameters, schema_version):
    """Update legacy corsika_starting_grammage parameter (dummy function until model is updated)."""
    logger.debug("No fix applied to %s to schema version %s", par_name, schema_version)
    return {
        parameters[par_name]["parameter"]: None,
    }


def _update_flasher_pulse_shape(par_name, parameters, schema_version):
    """Update legacy flasher_pulse_shape parameter."""
    para_data = parameters[par_name]
    if para_data["model_parameter_schema_version"] == "0.1.0" and schema_version == "0.2.0":
        logger.info(
            f"Updating legacy model parameter {para_data['parameter']} from schema version "
            f"{para_data['model_parameter_schema_version']} to {schema_version}"
        )
        return {
            para_data["parameter"]: {
                "value": [
                    para_data["value"],
                    parameters.get("flasher_pulse_width", {}).get("value", 0.0),
                    parameters.get("flasher_pulse_exp_decay", {}).get("value", 0.0),
                ],
                "model_parameter_schema_version": schema_version,
                "unit": [None, "ns", "ns"],
                "type": ["string", "float64", "float64"],
            },
            "flasher_pulse_width": {"remove_parameter": True},
            "flasher_pulse_exp_decay": {"remove_parameter": True},
        }

    raise ValueError(_get_unsupported_update_message(para_data, schema_version))


def _get_unsupported_update_message(para_data, schema_version):
    """Get unsupported update message."""
    return (
        f"Unsupported update for legacy parameter {para_data['parameter']}: "
        f"{para_data['model_parameter_schema_version']} to {schema_version}"
    )
