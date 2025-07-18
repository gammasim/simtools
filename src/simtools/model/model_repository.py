"""Utilities for managing the simulation models repository.

Simulation model parameters and production tables are managed through
a gitlab repository ('SimulationModels'). This module provides service
functions to interact with and verify the repository.
"""

import logging
from pathlib import Path

from simtools.io_operations import ascii_handler
from simtools.utils import names

_logger = logging.getLogger(__name__)


def verify_simulation_model_production_tables(simulation_models_path):
    """
    Verify the simulation model production tables in the specified path.

    Checks that all model parameters defined in the production tables are
    present in the simulation models repository.

    Parameters
    ----------
    simulation_models_path : str
        Path to the simulation models repository.

    Returns
    -------
    bool
        True if all parameters found, False if any missing.
    """
    productions_path = Path(simulation_models_path) / "simulation-models" / "productions"
    production_files = list(productions_path.rglob("*.json"))

    _logger.info(
        f"Verifying {len(production_files)} simulation model production "
        f"tables in {simulation_models_path}"
    )

    missing_files = []
    total_checked = 0

    for production_file in production_files:
        file_missing, file_checked = _verify_model_parameters_for_production(
            simulation_models_path, production_file
        )
        missing_files.extend(file_missing)
        total_checked += file_checked

    _logger.info(f"Checked {total_checked} parameters, {len(missing_files)} missing")

    if missing_files:
        for missing_file in missing_files:
            _logger.error(f"Missing: {missing_file}")
        return False

    _logger.info("Verification passed: All parameters found")
    return True


def _verify_model_parameters_for_production(simulation_models_path, production_file):
    """
    Verify that model parameters defined in the production tables exist.

    Parameters
    ----------
    simulation_models_path : str
        Path to the simulation models repository.
    production_file : Path
        Path to the production file.

    Returns
    -------
    tuple
        (missing_files_list, total_checked_count)
    """
    production_table = ascii_handler.collect_data_from_file(production_file)
    missing_files = []
    total_checked = 0

    parameters = production_table.get("parameters", {})
    for array_element, par_dict in parameters.items():
        if isinstance(par_dict, dict):
            for param_name, param_version in par_dict.items():
                total_checked += 1
                parameter_file = _get_model_parameter_file_path(
                    simulation_models_path, array_element, param_name, param_version
                )
                if parameter_file and not parameter_file.exists():
                    missing_files.append(str(parameter_file))

    return missing_files, total_checked


def _get_model_parameter_file_path(
    simulation_models_path, array_element, parameter_name, parameter_version
):
    """
    Get the file path for a model parameter.

    Take into account path structure based on collections and array elements.

    Parameters
    ----------
    simulation_models_path : str
        Path to the simulation models repository.
    array_element : str
        Name of the array element (e.g., 'telescope').
    parameter_name : str
        Name of the parameter.
    parameter_version : str
        Version of the parameter.

    Returns
    -------
    Path
        The file path to the model parameter JSON file.
    """
    collection = names.get_collection_name_from_parameter_name(parameter_name)
    return (
        Path(simulation_models_path)
        / "simulation-models"
        / "model_parameters"
        / (
            collection
            if collection in ("configuration_sim_telarray", "configuration_corsika")
            else ""
        )
        / (array_element if collection != "configuration_corsika" else "")
        / parameter_name
        / f"{parameter_name}-{parameter_version}.json"
    )
