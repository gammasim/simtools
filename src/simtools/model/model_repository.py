"""Utilities for managing the simulation models repository.

Simulation model parameters and production tables are managed through
a gitlab repository ('SimulationModels'). This module provides service
functions to interact with and verify the repository.
"""

import logging
from pathlib import Path

from simtools.io import ascii_handler
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


def generate_new_production(args_dict):
    """
    Generate a new production definition (production tables and model parameters).

    The following steps are performed:

    - copy of production tables from an existing base model version
    - update production tables with changes defined in a YAML file
    - generate new model parameter entries for changed parameters

    Parameters
    ----------
    args_dict: dict
        Dictionary containing the arguments for copying and updating production tables.
    """
    modifications = ascii_handler.collect_data_from_file(args_dict["modifications"])
    changes = modifications.get("changes", {})
    base_model_version = args_dict["base_model_version"]
    model_version = modifications["model_version"]

    simulation_models_path = Path(args_dict["simulation_models_path"])
    source_path = simulation_models_path / "productions" / base_model_version
    target_path = simulation_models_path / "productions" / model_version
    model_parameters_dir = simulation_models_path / "model_parameters"
    patch_update = args_dict.get("patch_update", False)

    _logger.info(f"Copying production tables from {source_path} to {target_path}")

    _apply_changes_to_production_tables(
        source_path,
        target_path,
        changes,
        model_version,
        patch_update,
        base_model_version,
    )

    _apply_changes_to_model_parameters(changes, model_parameters_dir)


def _apply_changes_to_production_tables(
    source_path, target_path, changes, model_version, patch_update, base_model_version
):
    """
    Apply changes to production tables and write them to target directory.

    Parameters
    ----------
    source_path: Path
        Path to the source production tables.
    target_path: Path
        Path to the target production tables.
    changes: dict
        The changes to be applied.
    model_version: str
        The model version to be set in the JSON data.
    patch_update: bool
        Patch update, copy only tables for changed elements.
    base_model_version: str
        The base model version from which the production tables were copied.
    """
    target_path.mkdir(parents=True, exist_ok=True)
    for file_path in Path(source_path).rglob("*.json"):
        data = ascii_handler.collect_data_from_file(file_path)
        write_to_disk = _apply_changes_to_production_table(
            data, changes, model_version, patch_update, base_model_version
        )
        if write_to_disk:
            ascii_handler.write_data_to_file(data, target_path / file_path.name, sort_keys=True)


def _apply_changes_to_production_table(
    data, changes, model_version, patch_update, base_model_version
):
    """
    Recursively apply changes to the new production tables.

    Parameters
    ----------
    data: dict or list
        The JSON data to be updated.
    changes: dict
        The changes to be applied.
    model_version: str
        The model version to be set in the JSON data.
    patch_update: bool
        Patch update, copy only tables for changed elements.
    base_model_version: str
        The base model version from which the production tables were copied.

    Returns
    -------
    bool
        True if data was modified and should be written to disk (patch updates) and always
        for full updates.
    """
    if isinstance(data, dict):
        table_name = data["production_table_name"]
        data["model_version"] = model_version
        if table_name in changes:
            data["parameters"] = _update_parameters(
                {} if patch_update else data["parameters"].get(table_name, {}), changes, table_name
            )
        elif patch_update:
            return False
        return True
    if isinstance(data, list):
        modified = False
        for item in data:
            modified = _apply_changes_to_production_table(
                item, changes, model_version, patch_update, base_model_version
            )
        return modified

    raise TypeError(f"Unsupported data type {type(data)} in production table update")


def _update_parameters(table_parameters, changes, table_name):
    """
    Create a new parameters dictionary containing only the parameters for the specified table.

    Parameters
    ----------
    table_parameters: dict
        Parameters for the specific table.
    changes: dict
        The changes to be applied, containing table and parameter information.
    table_name: str
        The name of the production table to filter parameters for.

    Returns
    -------
    dict
        Dictionary containing only the new/changed parameters for the specified table.
    """
    updated_parameters_dict = {table_name: table_parameters}
    for param, data in changes[table_name].items():
        version = data["version"]
        _logger.info(f"Setting '{table_name} - {param}' to version {version}")
        updated_parameters_dict[table_name][param] = version
    return updated_parameters_dict


def _apply_changes_to_model_parameters(changes, model_parameters_dir):
    """
    Apply changes to model parameters by creating new parameter entries.

    Parameters
    ----------
    changes: dict
        The changes to be applied.
    model_parameters_dir: str
        Path to the model parameters directory.
    """
    for telescope, parameters in changes.items():
        for param, param_data in parameters.items():
            if param_data.get("value"):
                _create_new_parameter_entry(telescope, param, param_data, model_parameters_dir)


def _create_new_parameter_entry(telescope, param, param_data, model_parameters_dir):
    """
    Create new model parameter JSON file by copying the latest version and updating fields.

    Parameters
    ----------
    telescope: str
        Name of the telescope.
    param: str
        Name of the parameter.
    param_data: dict
        Dictionary containing the parameter data including version and value.
    model_parameters_dir: str
        Path to the model parameters directory.
    """
    telescope_dir = Path(model_parameters_dir) / telescope
    if not telescope_dir.exists():
        raise FileNotFoundError(
            f"Directory for telescope '{telescope}' does not exist in '{model_parameters_dir}'."
        )

    param_dir = telescope_dir / param
    if not param_dir.exists():
        raise FileNotFoundError(
            f"Directory for parameter '{param}' does not exist in '{telescope}'."
        )

    latest_file = _get_latest_model_parameter_file(param_dir, param)
    if not latest_file:
        raise FileNotFoundError(
            f"No files found for parameter '{param}' in directory '{param_dir}'."
        )

    json_data = ascii_handler.collect_data_from_file(latest_file)

    json_data["parameter_version"] = _update_model_parameter_version(
        json_data, param_data, param, telescope
    )
    # important for e.g. nsb_pixel_rate
    if isinstance(json_data["value"], list) and not isinstance(param_data["value"], list):
        json_data["value"] = [param_data["value"]] * len(json_data["value"])
    else:
        json_data["value"] = param_data["value"]

    new_file_name = f"{param}-{param_data['version']}.json"
    new_file_path = param_dir / new_file_name

    ascii_handler.write_data_to_file(json_data, new_file_path, sort_keys=True)
    _logger.info(f"Created new model parameter JSON file: {new_file_path}")


def _get_latest_model_parameter_file(directory, parameter):
    """
    Get the latest model parameter JSON file for a parameter in the given directory.

    Assume files are named in the format 'parameter-version.json'.

    Parameters
    ----------
    directory: str
        Path to the directory containing parameter JSON files.
    parameter: str
        Name of the parameter to find.

    Returns
    -------
    str
        Path to the latest JSON file for the parameter.

    Raises
    ------
    FileNotFoundError
        If no files for the parameter are found in the directory.
    """
    directory_path = Path(directory)
    files = list(directory_path.glob(f"{parameter}-*.json"))
    if not files:
        raise FileNotFoundError(
            f"No JSON files found for parameter '{parameter}' in directory '{directory}'."
        )

    # Sort files by version number (assumes version is part of the filename)
    def safe_parse_version(filename):
        version_str = filename.stem.split("-")[-1]
        parts = version_str.split(".")
        return tuple(part.zfill(8) for part in parts)

    files.sort(key=safe_parse_version)
    return str(files[-1])


def _update_model_parameter_version(json_data, param_data, param, telescope):
    """Check for major version jump and print a warning if necessary."""
    latest_version = int(json_data.get("parameter_version", "0").split(".")[0])
    new_version = int(param_data["version"].split(".")[0])
    if new_version > latest_version + 1:
        _logger.warning(
            f"Major version jump from {latest_version} to {new_version} "
            f"for parameter '{param}' in telescope '{telescope}'."
        )
    return param_data["version"]
