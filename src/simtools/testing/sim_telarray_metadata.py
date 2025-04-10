"""Test consistency of sim_telarray metadata."""

import logging

import numpy as np

from simtools.simtel.simtel_config_reader import SimtelConfigReader
from simtools.simtel.simtel_io_metadata import read_sim_telarray_metadata
from simtools.utils import names

_logger = logging.getLogger(__name__)


def assert_sim_telarray_metadata(file, array_model):
    """
    Assert consistency of sim_telarray metadata with given array model.

    Parameters
    ----------
    file: Path
        Path to the sim_telarray file.
    array_model: ArrayModel
        Array model to compare with.
    """
    global_meta, telescope_meta = read_sim_telarray_metadata(file)
    _logger.info(f"Found metadata in sim_telarray file for {len(telescope_meta)} telescopes")
    site_parameter_mismatch = _assert_model_parameters(global_meta, array_model.site_model)

    if len(telescope_meta) != len(array_model.telescope_model):
        raise ValueError(
            f"Number of telescopes in sim_telarray file ({len(telescope_meta)}) does not match "
            f"number of telescopes in array model ({len(array_model.telescope_model)})"
        )

    telescope_parameter_mismatch = []
    for i, (_, model) in enumerate(array_model.telescope_model.items(), start=1):
        telescope_parameter_mismatch.append(_assert_model_parameters(telescope_meta[i], model))

    if len(site_parameter_mismatch) > 0 or any(len(m) > 0 for m in telescope_parameter_mismatch):
        if len(site_parameter_mismatch) > 0:
            raise ValueError(
                f"Site model parameters do not match sim_telarray metadata: "
                f"{site_parameter_mismatch}"
            )
        if any(len(m) > 0 for m in telescope_parameter_mismatch):
            raise ValueError(
                f"Telescope model parameters do not match sim_telarray metadata: "
                f"{telescope_parameter_mismatch}"
            )


def _assert_model_parameters(global_meta, model):
    """
    Assert that model parameter values matches the values in the sim_telarray metadata.

    Parameters
    ----------
    global_meta: dict
        Metadata dictionary.
    model: SiteModel, TelescopeModel
        Model to compare with.

    Returns
    -------
    invalid_parameter_list: list
        List of parameters that do not match.

    """
    config_reader = SimtelConfigReader()

    global_meta = {k.lower().lstrip("*"): v for k, v in global_meta.items()}
    global_meta = {k: v.strip() if isinstance(v, str) else v for k, v in global_meta.items()}

    invalid_parameter_list = []

    for param in model.parameters:
        sim_telarray_name = _sim_telarray_name_from_parameter_name(param)
        if sim_telarray_name in global_meta.keys():
            parameter_type = model.parameters[param]["type"]
            if parameter_type not in ("string", "dict", "boolean"):
                value = config_reader.extract_value_from_sim_telarray_column(
                    [global_meta[sim_telarray_name]], parameter_type
                )
            else:
                value = global_meta[sim_telarray_name]
            _logger.info(
                f"Parameter {param} in sim_telarray file: {value}, "
                f"in model: {model.parameters[param]['value']}"
            )
            if not is_equal(value, model.parameters[param]["value"], parameter_type):
                invalid_parameter_list.append(
                    f"Parameter {param} mismatch between sim_telarray file: {value}, "
                    f"and model: {model.parameters[param]['value']}"
                )

    return invalid_parameter_list


def _sim_telarray_name_from_parameter_name(parameter_name):
    """Return sim_telarray parameter name. Some specific fine tuning."""
    sim_telarray_name = names.get_simulation_software_name_from_parameter_name(parameter_name)

    # parameters like "reference_point_latitude"
    sim_telarray_name = parameter_name.replace("reference_point_", "")

    if sim_telarray_name == "altitude":
        return "corsika_observation_level"
    if sim_telarray_name == "array_triggers":
        return None

    return sim_telarray_name


def is_equal(value1, value2, value_type):
    """
    Check if two values are equal based on their type.

    Parameters
    ----------
    value1: any
        First value to compare.
    value2: any
        Second value to compare.
    value_type: str
        Type of the values ('string', 'dict', etc.).

    Returns
    -------
    bool
        True if the values are equal, False otherwise.
    """
    print("AAAA", value_type)
    if isinstance(value1, tuple):
        value1 = value1[0]
    if isinstance(value2, tuple):
        value2 = value2[0]
    if value1 is None or value2 is None:
        if value1 in ("none", None) and value2 in ("none", None):
            return True
    if value_type == "string":
        return str(value1).strip() == str(value2).strip()
    if value_type == "dict":
        return value1 == value2
    if value_type == "boolean":
        print("FFFFF", value1, value2, type(value1), type(value2))
        return bool(value1) == bool(value2)
    if isinstance(value1, np.ndarray | list) and isinstance(value2, np.ndarray | list):
        return np.allclose(np.array(value1), np.array(value2), rtol=1e-10)
    return np.isclose(float(value1), float(value2), rtol=1e-10)
