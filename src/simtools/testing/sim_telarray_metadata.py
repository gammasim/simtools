"""Test consistency of sim_telarray metadata."""

import logging

import numpy as np

from simtools.sim_events.file_info import get_corsika_run_number
from simtools.simtel.simtel_config_reader import SimtelConfigReader
from simtools.simtel.simtel_io_metadata import (
    get_sim_telarray_telescope_id,
    read_sim_telarray_metadata,
)
from simtools.utils import random

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
    sim_telarray_seed_mismatch = _assert_sim_telarray_seed(
        global_meta, array_model.instrument_seed, file
    )
    if sim_telarray_seed_mismatch:
        site_parameter_mismatch.append(sim_telarray_seed_mismatch)

    if len(telescope_meta) != len(array_model.telescope_models):
        raise ValueError(
            f"Number of telescopes in sim_telarray file ({len(telescope_meta)}) does not match "
            f"number of telescopes in array model ({len(array_model.telescope_models)})"
        )

    for telescope_name in array_model.telescope_models.keys():
        if not get_sim_telarray_telescope_id(telescope_name, file):
            raise ValueError(f"Telescope {telescope_name} not found in sim_telarray file metadata")

    telescope_parameter_mismatch = [
        _assert_model_parameters(telescope_meta[i], model)
        for i, model in enumerate(array_model.telescope_models.values(), start=1)
    ]

    # ensure printout of all mismatches, not only those found first
    telescope_parameter_mismatch.append(site_parameter_mismatch)
    if any(len(m) > 0 for m in telescope_parameter_mismatch):
        mismatches = [m for m in telescope_parameter_mismatch if len(m) > 0]
        raise ValueError(
            f"Telescope or site model parameters do not match sim_telarray metadata: {mismatches}"
        )


def _assert_model_parameters(metadata, model):
    """
    Assert that model parameter values matches the values in the sim_telarray metadata.

    Parameters
    ----------
    metadata: dict
        Metadata dictionary.
    model: SiteModel, TelescopeModel
        Model to compare with.

    Returns
    -------
    invalid_parameter_list: list
        List of parameters that do not match.

    """
    config_reader = SimtelConfigReader()

    invalid_parameter_list = []

    for param in model.parameters:
        sim_telarray_name = _sim_telarray_name_from_parameter_name(param)
        if sim_telarray_name in metadata.keys():
            parameter_type = model.parameters[param]["type"]
            if parameter_type not in ("string", "dict", "boolean"):
                value, _ = config_reader.extract_value_from_sim_telarray_column(
                    [metadata[sim_telarray_name]], parameter_type
                )
            else:
                value = metadata[sim_telarray_name]
                value = (int)(value) if value.isnumeric() else value

            if not is_equal(value, model.parameters[param]["value"], parameter_type):
                invalid_parameter_list.append(
                    f"Parameter {param} mismatch between sim_telarray file: {value}, "
                    f"and model: {model.parameters[param]['value']}"
                )

    return invalid_parameter_list


def _assert_sim_telarray_seed(metadata, sim_telarray_seed, file=None):
    """
    Assert that sim_telarray seed matches the values in the sim_telarray metadata.

    Regenerate seeds using the sim_telarray_random_seeds function and compare with the metadata.

    Parameters
    ----------
    metadata: dict
        Metadata dictionary.
    sim_telarray_seed: int
        Sim_telarray seed.
    file : Path
        Path to the sim_telarray file.

    Returns
    -------
    invalid_parameter_list: list
        Error message if sim_telarray seeds do not match.

    """
    if "instrument_seed" in metadata.keys() and "instrument_instances" in metadata.keys():
        if str(metadata.get("instrument_seed")) != str(sim_telarray_seed):
            return (
                "Parameter instrument_seed mismatch between sim_telarray file: "
                f"{metadata['instrument_seed']}, and model: {sim_telarray_seed}"
            )
        _logger.info(
            f"sim_telarray_seed in sim_telarray file: {metadata['instrument_seed']}, "
            f"and model: {sim_telarray_seed}"
        )
        if file:
            run_number_modified = get_corsika_run_number(file) - 1
            test_seeds = random.seeds(
                n_seeds=int(metadata["instrument_instances"]),
                max_seed=np.iinfo(np.int32).max,
                fixed_seed=int(metadata["instrument_seed"]),
            )
            # no +1 as in sim_telarray (as we count from 0)
            seed_used = run_number_modified % int(metadata["instrument_instances"])
            if str(metadata.get("rng_select_seed")) != str(test_seeds[seed_used]):
                return (
                    "Parameter rng_select_seed mismatch between sim_telarray file: "
                    f"{metadata['rng_select_seed']}, and model: {test_seeds[seed_used]}"
                )

    return None


def _sim_telarray_name_from_parameter_name(parameter_name):
    """Return sim_telarray parameter name. Some specific fine tuning."""
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

    The complexity of this function reflects the complexity of the sim_telarray
    metadata output.

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
    value1 = value1[0] if isinstance(value1, tuple) else value1
    value2 = value2[0] if isinstance(value2, tuple) else value2
    if value1 is None or value2 is None:
        if value1 in ("none", None) and value2 in ("none", None):
            return True
    if value_type == "string":
        return str(value1).strip() == str(value2).strip()
    if value_type == "dict":
        return value1 == value2
    if value_type == "boolean":
        return bool(value1) == bool(value2)
    return _is_equal_floats_or_ints(value1, value2)


def _is_equal_floats_or_ints(value1, value2):
    """Check if floats and ints are equal."""
    if isinstance(value1, np.ndarray | list) and isinstance(value2, np.ndarray | list):
        return bool(np.allclose(np.array(value1), np.array(value2), rtol=1e-10))
    if isinstance(value1, list) and isinstance(value2, float | int | np.integer | np.floating):
        if all(x == value1[0] for x in value1):
            return bool(np.isclose(float(value1[0]), float(value2), rtol=1e-10))
    if isinstance(value2, list) and isinstance(value1, float | int | np.integer | np.floating):
        if all(x == value2[0] for x in value2):
            return bool(np.isclose(float(value1), float(value2[0]), rtol=1e-10))
    return bool(np.isclose(float(value1), float(value2), rtol=1e-10))
