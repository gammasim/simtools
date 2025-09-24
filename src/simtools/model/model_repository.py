"""Utilities for managing the simulation models repository.

Simulation model parameters and production tables are managed through
a gitlab repository ('SimulationModels'). This module provides service
functions to interact with and verify the repository.
"""

import logging
from pathlib import Path

from packaging.version import Version
from packaging.version import parse as parse_version

import simtools.data_model.model_data_writer as writer
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


def generate_new_production(modifications, simulation_models_path):
    """
    Generate a new production definition (production tables and model parameters).

    The following steps are performed:

    - copy of production tables from an existing base model version
    - update production tables with changes defined in a YAML file
    - generate new model parameter entries for changed parameters
    - allows for full or patch updates

    Parameters
    ----------
    modifications: str
        Path to the YAML file defining the changes to be applied.
    simulation_models_path: str
        Path to the simulation models repository.
    """
    modifications = ascii_handler.collect_data_from_file(modifications)
    model_version_history = modifications.get("model_version_history", [])
    try:
        # oldest version is the base version
        base_model_version = min(set(model_version_history), key=Version)
    except ValueError as exc:
        raise ValueError(f"Base model version not found in {modifications}") from exc
    model_version = modifications["model_version"]
    changes = modifications.get("changes", {})

    _apply_changes_to_production_tables(
        changes,
        base_model_version,
        model_version,
        modifications.get("model_update", "full_update"),
        simulation_models_path,
    )

    _apply_changes_to_model_parameters(changes, simulation_models_path)


def _apply_changes_to_production_tables(
    changes, base_model_version, model_version, update_type, simulation_models_path
):
    """
    Apply changes to production tables and write them to target directory.

    Parameters
    ----------
    changes: dict
        The changes to be applied.
    base_model_version: str
        The base model version (source directory for production tables).
    model_version: str
        The model version to be set in the JSON data.
    update_type: str
        Update mode, either 'full_update' or 'patch_update'.
    simulation_models_path: Path
        Path to the simulation models repository.
    """
    source = simulation_models_path / "productions" / base_model_version
    target = simulation_models_path / "productions" / model_version
    _logger.info(f"Production tables {update_type} from {source} to {target}")
    target.mkdir(parents=True, exist_ok=True)

    # load existing tables
    tables = {}
    for file_path in Path(source).rglob("*.json"):
        data = ascii_handler.collect_data_from_file(file_path)
        if not isinstance(data, dict):
            raise TypeError(f"Unsupported data type {type(data)} in {file_path}")
        tables[data["production_table_name"]] = data

    # placeholder for new tables
    for table_name in changes:
        tables.setdefault(table_name, {})

    for table_name, data in tables.items():
        if _apply_changes_to_production_table(
            table_name, data, changes, model_version, update_type == "patch_update"
        ):
            _logger.info(f"Writing updated production table '{table_name}'")
            data["production_table_name"] = table_name
            ascii_handler.write_data_to_file(data, target / f"{table_name}.json", sort_keys=True)


def _apply_changes_to_production_table(table_name, data, changes, model_version, patch_update):
    """
    Apply changes to a single production table.

    Parameters
    ----------
    data: dict
        The data to be updated.
    changes: dict
        The changes to be applied.
    model_version: str
        The model version to be set in the JSON data.
    patch_update: bool
        True if patch update (modify only changed parameters), False for full update.

    Returns
    -------
    bool
        True if data was modified and should be written to disk (patch updates);
        always True for full updates.
    """
    data["model_version"] = model_version
    if table_name in changes:
        table_parameters = {} if patch_update else data.get("parameters", {}).get(table_name, {})
        parameters, deprecated = _update_parameters_dict(table_parameters, changes, table_name)
        data["parameters"] = parameters
        if deprecated:
            data["deprecated_parameters"] = deprecated
    elif patch_update:
        return False

    return True


def _update_parameters_dict(table_parameters, changes, table_name):
    """
    Create a new parameters dictionary for the production tables.

    Include only changes relevant to the specific telescope.
    Do not include parameters if 'deprecated' flag is set to True.

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
    dict, list
        Dictionary containing only the new/changed parameters for the specified table.
        List of deprecated parameters.
    """
    new_params = {table_name: table_parameters}
    deprecated_params = []

    for param, data in changes[table_name].items():
        if data.get("deprecated", False):
            _logger.info(f"Removing model parameter '{table_name} - {param}'")
            deprecated_params.append(param)
        else:
            version = data["version"]
            _logger.info(f"Setting '{table_name} - {param}' to version {version}")
            new_params[table_name][param] = version

    return new_params, deprecated_params


def _apply_changes_to_model_parameters(changes, simulation_models_path):
    """
    Apply changes to model parameters by creating new parameter entries.

    Parameters
    ----------
    changes: dict
        The changes to be applied.
    simulation_models_path: Path
        Path to the simulation models directory.
    """
    for telescope, parameters in changes.items():
        for param, param_data in parameters.items():
            if param_data.get("value") is not None:
                _create_new_model_parameter_entry(
                    telescope, param, param_data, simulation_models_path
                )


def _create_new_model_parameter_entry(telescope, param, param_data, simulation_models_path):
    """
    Create new model parameter entry in the model parameters directory.

    If a model parameter files exists, copy latest version and update the fields.
    Otherwise generate new file using the model parameter schema.

    Parameters
    ----------
    telescope: str
        Name of the telescope.
    param: str
        Name of the parameter.
    param_data: dict
        Dictionary containing the parameter data including version and value.
    simulation_models_path: Path
        Path to the simulation models directory.
    """
    telescope_dir = simulation_models_path / "model_parameters" / telescope
    if not telescope_dir.exists():
        _logger.info(f"Create directory for array element '{telescope}': '{telescope_dir}'.")
        telescope_dir.mkdir(parents=True, exist_ok=True)

    param_dir = telescope_dir / param
    try:
        latest_file = _get_latest_model_parameter_file(param_dir, param)
    except FileNotFoundError:
        latest_file = None

    if latest_file is not None:
        json_data = ascii_handler.collect_data_from_file(latest_file)
        param_data["version"] = _check_for_major_version_jump(
            json_data, param_data, param, telescope
        )
        # important for e.g. nsb_pixel_rate
        if isinstance(json_data["value"], list) and not isinstance(param_data["value"], list):
            param_data["value"] = [param_data["value"]] * len(json_data["value"])
        param_data["meta_parameter"] = json_data.get("meta_parameter", False)

    writer.ModelDataWriter.dump_model_parameter(
        parameter_name=param,
        value=param_data["value"],
        instrument=telescope,
        parameter_version=param_data["version"],
        output_file=f"{param}-{param_data['version']}.json",
        output_path=param_dir,
        unit=param_data.get("unit"),
        meta_parameter=param_data.get("meta_parameter", False),
    )


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

    def extract_version(path: Path):
        # version is part after first '-'
        return parse_version(path.stem.split("-", 1)[1])

    latest_file = max(files, key=extract_version)
    return str(latest_file)


def _check_for_major_version_jump(json_data, param_data, param, telescope):
    """
    Check for major version jump and print a warning if necessary.

    Generally a jump from e.g. '3.1.0' to '5.0.0' should be avoided.
    """
    latest_version = parse_version(json_data.get("parameter_version", "0"))
    new_version = parse_version(param_data["version"])
    if new_version.major > latest_version.major + 1:
        _logger.warning(
            f"Major version jump from {latest_version} to {new_version} "
            f"for parameter '{param}' in telescope '{telescope}'."
        )
    return param_data["version"]
