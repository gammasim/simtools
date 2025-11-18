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


UPDATE_HANDLERS = {}


def register_update(name):
    """Register update handler for legacy model parameter."""

    def deco(func):
        UPDATE_HANDLERS[name] = func
        return func

    return deco


def apply_legacy_updates_to_parameters(parameters, legacy_updates):
    """Apply legacy updates to model parameters.

    Parameters
    ----------
    parameters: dict
        Dictionary of model parameters (all parameters).
    legacy_updates: dict
        Dictionary of legacy updates to apply.

    Returns
    -------
    dict
        Updated model parameters.
    """
    for par_name, legacy_data in legacy_updates.items():
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
    dict
        Updated model parameter.
    """
    handler = UPDATE_HANDLERS.get(par_name)
    if handler is None:
        raise ValueError(_get_unsupported_update_message(parameters[par_name], schema_version))
    return handler(parameters, schema_version)


@register_update("dsum_threshold")
def _update_dsum_threshold(parameters, schema_version):
    """Update legacy dsum_threshold parameter."""
    para_data = parameters["dsum_threshold"]
    if para_data["model_parameter_schema_version"] == "0.1.0" and schema_version == "0.2.0":
        logger.info(
            "Updating legacy model parameter dsum_threshold from schema version "
            f"{para_data['model_parameter_schema_version']} to {schema_version}"
        )
        return {
            para_data["parameter"]: {
                "value": int(para_data["value"]),
                "model_parameter_schema_version": schema_version,
            }
        }
    raise ValueError(_get_unsupported_update_message(para_data, schema_version))


@register_update("corsika_starting_grammage")
def _update_corsika_starting_grammage(parameters, schema_version):
    """Update legacy corsika_starting_grammage parameter (dummy function until model is updated)."""
    logger.debug(f"No fix applied to corsika_starting_grammage to schema version {schema_version}")
    return {
        parameters["corsika_starting_grammage"]["parameter"]: None,
    }


@register_update("flasher_pulse_shape")
def _update_flasher_pulse_shape(parameters, schema_version):
    """Update legacy flasher_pulse_shape parameter."""
    para_data = parameters["flasher_pulse_shape"]
    if para_data["model_parameter_schema_version"] == "0.1.0" and schema_version == "0.2.0":
        logger.info(
            f"Updating legacy model parameter flasher_pulse_shape from schema version "
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
