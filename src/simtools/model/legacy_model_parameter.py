#!/usr/bin/python3
"""
Handling of legacy model parameters for backward compatibility.

Collects routines to convert legacy model parameters to more recent
formats. This is a fine-tuned process and requires to hard wire the
changes. All code related to legacy model parameters should go into
this module.

"""

import logging

from simtools.data_model import row_table_utils

logger = logging.getLogger(__name__)


UPDATE_HANDLERS = {}


def _log_schema_update(parameter_name, from_schema_version, to_schema_version):
    """Log schema migration for a legacy model parameter."""
    logger.info(
        f"Updating legacy model parameter {parameter_name} from schema version "
        f"{from_schema_version} to {to_schema_version}"
    )


def register_update(name):
    """Register update handler for legacy model parameter."""

    def deco(func):
        UPDATE_HANDLERS[name] = func
        return func

    return deco


def apply_legacy_updates_to_parameters(parameters, legacy_updates):
    """Apply legacy updates to model parameters.

    Modifies the parameters dictionary in-place.

    Parameters
    ----------
    parameters: dict
        Dictionary of model parameters (all parameters).
    legacy_updates: dict
        Dictionary of legacy updates to apply.
    """
    for par_name, legacy_data in legacy_updates.items():
        if legacy_data is None or par_name not in parameters:
            continue
        for key in parameters[par_name].keys():
            if key in legacy_data:
                parameters[par_name][key] = legacy_data[key]
        if legacy_data.get("remove_parameter", False):
            parameters.pop(par_name)


def update_parameter(par_name, parameters, schema_version, value_resolver=None):
    """Update legacy model parameters to recent formats.

    Parameters
    ----------
    par_name: str
        Model parameter name.
    parameters: dict
        Dictionary of model parameters (all parameters).
    schema_version: str
        Target schema version.
    value_resolver: callable, optional
        Callback used by handlers that need to normalize a stored legacy value
        before it can be embedded in the updated parameter. The callback must
        accept ``(parameter_name, value)`` and return the canonical in-memory
        representation for that parameter value.

    Returns
    -------
    dict
        Updated model parameter.
    """
    handler = UPDATE_HANDLERS.get(par_name)
    if handler is None:
        raise ValueError(_get_unsupported_update_message(parameters[par_name], schema_version))
    return handler(parameters, schema_version, value_resolver=value_resolver)


def _update_file_backed_table_parameter(
    parameter_name,
    parameters,
    schema_version,
    value_resolver=None,
):
    """Update a legacy file-backed table parameter to embedded row data.

    The ``value_resolver`` callback is expected to convert the stored legacy
    value, typically a file name for a GridFS-backed table, into the canonical
    embedded ``{"columns", "rows"}`` representation used in memory.
    """
    para_data = parameters[parameter_name]
    if value_resolver is None:
        raise ValueError(
            f"A value_resolver is required to update legacy file-backed parameter {parameter_name}."
        )

    return {
        para_data["parameter"]: {
            "value": value_resolver(parameter_name, para_data["value"]),
            "model_parameter_schema_version": schema_version,
            "type": "dict",
            "file": False,
        }
    }


@register_update("dsum_threshold")
def _update_dsum_threshold(parameters, schema_version, value_resolver=None):
    """Update legacy dsum_threshold parameter."""
    _ = value_resolver
    para_data = parameters["dsum_threshold"]
    if para_data["model_parameter_schema_version"] == "0.1.0" and schema_version == "0.2.0":
        _log_schema_update(
            para_data["parameter"],
            para_data["model_parameter_schema_version"],
            schema_version,
        )
        return {
            para_data["parameter"]: {
                "value": int(para_data["value"]),
                "model_parameter_schema_version": schema_version,
            }
        }
    raise ValueError(_get_unsupported_update_message(para_data, schema_version))


@register_update("corsika_starting_grammage")
def _update_corsika_starting_grammage(parameters, schema_version, value_resolver=None):  # pylint: disable=unused-argument
    """Update legacy corsika_starting_grammage parameter (dummy function until model is updated)."""
    return {
        parameters["corsika_starting_grammage"]["parameter"]: None,
    }


@register_update("flasher_pulse_shape")
def _update_flasher_pulse_shape(parameters, schema_version, value_resolver=None):
    """Update legacy flasher_pulse_shape parameter."""
    _ = value_resolver
    para_data = parameters["flasher_pulse_shape"]
    if para_data["model_parameter_schema_version"] == "0.1.0" and schema_version == "0.2.0":
        _log_schema_update(
            para_data["parameter"],
            para_data["model_parameter_schema_version"],
            schema_version,
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


@register_update("fadc_pulse_shape")
def _update_fadc_pulse_shape(parameters, schema_version, value_resolver=None):
    """Update legacy fadc_pulse_shape parameter."""
    para_data = parameters["fadc_pulse_shape"]
    current_schema_version = para_data["model_parameter_schema_version"]
    value = para_data.get("value")
    parameter_name = para_data["parameter"]

    # Generic migration for legacy file-backed payloads.
    if para_data.get("file") and isinstance(value, str):
        _log_schema_update(parameter_name, current_schema_version, schema_version)
        return _update_file_backed_table_parameter(
            "fadc_pulse_shape",
            parameters,
            schema_version,
            value_resolver=value_resolver,
        )

    # Already in canonical row-oriented format {columns, rows} - pass through.
    if para_data.get("type") == "dict" and row_table_utils.is_row_table_dict(value):
        _log_schema_update(parameter_name, current_schema_version, schema_version)
        return {
            para_data["parameter"]: {
                "value": value,
                "model_parameter_schema_version": schema_version,
            }
        }

    raise ValueError(_get_unsupported_update_message(para_data, schema_version))


def _get_unsupported_update_message(para_data, schema_version):
    """Get unsupported update message."""
    return (
        f"Unsupported update for legacy parameter {para_data['parameter']}: "
        f"{para_data['model_parameter_schema_version']} to {schema_version}"
    )
