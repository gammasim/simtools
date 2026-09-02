"""Utilities for managing the simulation models repository.

Simulation model parameters and production tables are managed through
a gitlab repository ('simulation_models'). This module provides service
functions to interact with and verify the repository.

Main functionalities are:

- validation of production tables against model parameters
- generation of new production tables and model parameters based on
  updates defined in a configuration file

"""

import logging
from pathlib import Path

from packaging.version import Version
from packaging.version import parse as parse_version

import simtools.data_model.model_data_writer as writer
from simtools.constants import DEFAULT_SIMULATION_WORKFLOWS
from simtools.io import ascii_handler
from simtools.utils import names, value_conversion

_logger = logging.getLogger(__name__)


def get_production_directory(simulation_models_path, model_version=None):
    """
    Get the production directory for a specific model version.

    Parameters
    ----------
    simulation_models_path : str
        Path to the simulation models repository.
    model_version : str, optional
        Specific model version to get the production directory for.

    Returns
    -------
    Path
        Path to the production directory.
    """
    if model_version:
        return Path(simulation_models_path) / "simulation-models/productions" / str(model_version)
    return Path(simulation_models_path) / "simulation-models/productions"


def get_model_parameter_directory(simulation_models_path):
    """
    Get the model parameters directory.

    Parameters
    ----------
    simulation_models_path : str
        Path to the simulation models repository.

    Returns
    -------
    Path
        Path to the model parameters directory.
    """
    return Path(simulation_models_path) / "simulation-models/model_parameters"


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
    productions_path = get_production_directory(simulation_models_path)
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
                parameter_file = get_model_parameter_file_path(
                    simulation_models_path, array_element, param_name, param_version
                )
                if parameter_file and not parameter_file.exists():
                    missing_files.append(str(parameter_file))

    return missing_files, total_checked


def get_model_parameter_file_path(
    simulation_models_path, array_element, parameter_name, parameter_version
):
    """
    Get the file path for a model parameter.

    Use the instrument scope of the parameter. Null-instrument parameters are
    stored below the explicit ``global`` scope.

    Parameters
    ----------
    simulation_models_path : str
        Path to the simulation models repository.
    array_element : str
        Name of the array element (e.g., 'telescope'), or ``None`` for a
        global parameter.
    parameter_name : str
        Name of the parameter.
    parameter_version : str
        Version of the parameter.

    Returns
    -------
    Path
        The file path to the model parameter JSON file.
    """
    instrument = _get_model_parameter_scope(array_element, parameter_name)
    return (
        get_model_parameter_directory(simulation_models_path)
        / instrument
        / parameter_name
        / f"{parameter_name}-{parameter_version}.json"
    )


def _get_model_parameter_scope(telescope, parameter_name=None):
    """Return the filesystem scope for a production-table key and parameter."""
    if telescope in (None, "global", "configuration_corsika"):
        return "global"
    if names.is_global_sim_telarray_parameter(parameter_name):
        return "global"
    return telescope


def generate_new_production(model_version, simulation_models_path, setting_workflows_git_tag=None):
    """
    Generate a new production definition (production tables and model parameters).

    The following steps are performed:

    - copy of production tables from an existing base model version
    - update production tables with changes defined in a configuration file (expected
      to be called 'info.yml' in the target production directory)
    - generate new model parameter entries for changed parameters
    - allows for full or patch updates

    Parameters
    ----------
    model_version: str
        Model version to be created or updated.
    simulation_models_path: str
        Path to the simulation models repository.
    setting_workflows_git_tag: str, optional
        Branch or tag used to download parameters from the simulation workflow repository.
        If provided, this value overrides ``setting_workflows_git_tag`` from ``info.yml``.
        If None, the value from ``info.yml`` is used, with ``"main"`` as fallback.
    """
    modification_dict = _get_changes_dict(model_version, simulation_models_path)
    update_type = modification_dict.get("model_update", "full_update")
    if setting_workflows_git_tag is None:
        setting_workflows_git_tag = modification_dict.get("setting_workflows_git_tag", "main")
    setting_workflows_git_repository = modification_dict.get(
        "setting_workflows_git_repository", DEFAULT_SIMULATION_WORKFLOWS
    )
    changes, base_model_version = _get_changes_to_production(
        modification_dict, simulation_models_path, update_type
    )
    model_parameter_changes = modification_dict.get("changes", {})

    _apply_changes_to_production_tables(
        changes,
        base_model_version,
        modification_dict["model_version"],
        update_type,
        simulation_models_path,
    )

    _apply_changes_to_model_parameters(
        model_parameter_changes,
        simulation_models_path,
        setting_workflows_git_tag,
        setting_workflows_git_repository,
    )


def _get_production_table_key(table_name):
    """
    Get the production table key for a given table name.

    CORSIKA configuration uses 'global' to indicate parameters that are
    site-wide and independent of specific telescope designs.

    Parameters
    ----------
    table_name : str
        Table name (e.g., 'configuration_corsika', 'LSTN-01').

    Returns
    -------
    str
        Production table key to use in parameter dictionaries.
    """
    return "global" if table_name == "configuration_corsika" else table_name


def _apply_changes_to_production_tables(
    changes, base_model_version, model_version, update_type, simulation_models_path
):
    """
    Apply changes to or generate new production tables and write them to target directory.

    Parameters
    ----------
    changes: dict
        Changes to be applied.
    base_model_version: str
        Base model version (source directory for production tables).
    model_version: str
        Model version of the new production tables.
    update_type: str
        Update type (e.g., 'full_update' or 'patch_update').
    simulation_models_path: Path
        Path to the simulation models repository.
    """
    source = get_production_directory(simulation_models_path, base_model_version)
    target = get_production_directory(simulation_models_path, model_version)
    _logger.info(f"Production tables {update_type} from {source} to {target}")
    target.mkdir(parents=True, exist_ok=True)

    # load existing tables from source
    tables = {}
    for file_path in Path(source).rglob("*.json"):
        data = ascii_handler.collect_data_from_file(file_path)
        if not isinstance(data, dict):
            raise TypeError(f"Unsupported data type {type(data)} in {file_path}")
        tables[data["production_table_name"]] = data

    # placeholder for new tables
    for table_name in changes:
        tables.setdefault(table_name, {})

    tables.setdefault("configuration_sim_telarray", {})

    for table_name, data in tables.items():
        if table_name == "configuration_sim_telarray":
            has_changes = _apply_changes_to_sim_telarray_production_table(
                data,
                changes,
                model_version,
                update_type == "patch_update",
            )
            should_write = update_type != "patch_update" or has_changes
        else:
            should_write = _apply_changes_to_production_table(
                table_name, data, changes, model_version, update_type == "patch_update"
            )
        if should_write:
            target_file = target / f"{table_name}.json"
            _logger.info(f"Writing updated production table '{target_file}'")
            data["production_table_name"] = table_name
            ascii_handler.write_data_to_file(data, target_file, sort_keys=True)


def _apply_changes_to_production_table(table_name, data, changes, model_version, patch_update):
    """
    Apply changes to a single production table.

    Parameters
    ----------
    table_name: str
        Name of the production table.
    data: dict
        Data to be updated.
    changes: dict
        Changes to be applied.
    model_version: str
        Model version of the new production tables.
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
        production_key = _get_production_table_key(table_name)
        table_parameters = (
            {} if patch_update else data.get("parameters", {}).get(production_key, {})
        )
        parameters, deprecated = _update_parameters_dict(table_parameters, changes, table_name)
        if patch_update and not parameters.get(production_key) and not deprecated:
            return False
        data["parameters"] = parameters
        if deprecated and patch_update:
            data["deprecated_parameters"] = deprecated
    elif patch_update:
        return False

    return True


def _get_changes_dict(model_version, simulation_models_path):
    """
    Load the changes dictionary from 'info.yml' files in production directories.

    Parameters
    ----------
    model_version: str
        Model version of the new production tables.
    simulation_models_path: Path
        Path to the simulation models directory.

    Returns
    -------
    dict
        Changes dictionary.
    """
    return ascii_handler.collect_data_from_file(
        get_production_directory(simulation_models_path, model_version) / "info.yml"
    )


def _get_changes_to_production(
    modification_dict, simulation_models_path, update_type="full_update"
):
    """
    Prepare changes applied to production tables.

    For full updates, this includes the combination of changes to be applied
    for all model versions in the history, starting from the base version.

    Parameters
    ----------
    modification_dict: dict
        Modifications dictionary.
    simulation_models_path: Path
        Path to the simulation models directory.
    update_type: str
        Update mode.

    Returns
    -------
    dict, str
        Changes dictionary and base model version.
    """
    model_version_history = modification_dict.get("model_version_history", [])

    try:
        # oldest version is the base version
        base_model_version = min(set(model_version_history), key=Version)
    except ValueError:
        _logger.debug(f"Base model version not found in {model_version_history}")
        return {}, modification_dict.get("model_version")

    changes = modification_dict.get("changes", {})
    if update_type == "patch_update":
        return changes, base_model_version

    for version_mod in reversed(model_version_history):
        _changes_dict = _get_changes_dict(version_mod, simulation_models_path)
        _version_changes, base_model_version = _get_changes_to_production(
            _changes_dict, simulation_models_path, update_type="full_update"
        )
        changes = _update_two_levels_in_changes_dict(_version_changes, changes)
        # stop iterative loop after reaching first full version of production tables
        if _changes_dict.get("model_update", "full_update") == "full_update":
            break

    return changes, base_model_version


def _update_two_levels_in_changes_dict(d, u):
    """Update changes dict, e.g. {"LSTN-design": { "parameter_name: { ... } } }."""
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            d[k].update(v)
        else:
            d[k] = v
    return d


def _apply_sim_telarray_changes_for_telescope(telescope_params, params):
    """Apply configuration_sim_telarray changes for one telescope."""
    has_changes = False
    deprecated = []

    for param_name, param_data in params.items():
        if (
            names.get_collection_name_from_parameter_name(param_name)
            != "configuration_sim_telarray"
        ):
            continue

        has_changes = True
        if param_data.get("deprecated", False):
            if telescope_params is not None:
                telescope_params.pop(param_name, None)
            deprecated.append(param_name)
            continue

        if telescope_params is None:
            telescope_params = {}
        telescope_params[param_name] = param_data["version"]

    return telescope_params, deprecated, has_changes


def _apply_global_sim_telarray_changes(global_params, params):
    """Apply global-scope sim_telarray changes found under one telescope key."""
    deprecated = []
    has_changes = False
    for param_name, param_data in params.items():
        if names.get_collection_name_from_parameter_name(
            param_name
        ) != "configuration_sim_telarray" or not names.is_global_sim_telarray_parameter(param_name):
            continue
        has_changes = True
        if param_data.get("deprecated", False):
            global_params.pop(param_name, None)
            deprecated.append(param_name)
        else:
            global_params[param_name] = param_data["version"]
    return deprecated, has_changes


def _remove_global_sim_telarray_parameters(parameters):
    """Remove migrated global parameters from non-global production scopes."""
    for scope, scope_parameters in parameters.items():
        if scope != "global" and isinstance(scope_parameters, dict):
            for parameter_name in names.SIM_TELARRAY_GLOBAL_PARAMETERS:
                scope_parameters.pop(parameter_name, None)


def _apply_sim_telarray_changes_for_scope(parameters, global_params, telescope, params):
    """Apply global and telescope-scoped sim_telarray changes for one scope."""
    global_deprecated, global_has_changes = _apply_global_sim_telarray_changes(
        global_params, params
    )
    telescope_params = {
        param_name: param_data
        for param_name, param_data in params.items()
        if not names.is_global_sim_telarray_parameter(param_name)
    }
    telescope_params, deprecated, telescope_has_changes = _apply_sim_telarray_changes_for_telescope(
        parameters.get(telescope), telescope_params
    )
    if telescope_params is not None:
        parameters[telescope] = telescope_params
    return (
        global_has_changes or telescope_has_changes,
        global_deprecated,
        deprecated,
    )


def _apply_changes_to_sim_telarray_production_table(data, changes, model_version, patch_update):
    """
    Apply configuration_sim_telarray parameter changes to the production table.

    The configuration_sim_telarray production table stores process-wide
    parameters under ``global`` and telescope-dependent parameters by design.

    Parameters
    ----------
    data : dict
        Production table data to update in place.
    changes : dict
        Full changes dictionary. Only parameters in the configuration_sim_telarray
        collection are applied.
    model_version : str
        Model version of the new production tables.
    patch_update : bool
        True if patch update (only changed parameters), False for full update.

    Returns
    -------
    bool
        True if at least one configuration_sim_telarray parameter was found.
    """
    data["model_version"] = model_version

    has_changes = False
    parameters = data.get("parameters", {})

    _remove_global_sim_telarray_parameters(parameters)
    global_params = parameters.setdefault("global", {})

    global_deprecated = []
    for telescope, params in changes.items():
        if not isinstance(params, dict):
            continue
        changes_found, global_changes, telescope_changes = _apply_sim_telarray_changes_for_scope(
            parameters, global_params, telescope, params
        )
        has_changes = has_changes or changes_found
        global_deprecated.extend(global_changes)
        if telescope_changes and patch_update:
            data.setdefault("deprecated_parameters", []).extend(telescope_changes)
    if global_deprecated and patch_update:
        data.setdefault("deprecated_parameters", []).extend(global_deprecated)
    if not global_params:
        parameters.pop("global", None)
    data["parameters"] = parameters
    return has_changes


def _update_parameters_dict(table_parameters, changes, table_name):
    """
    Create a new parameters dictionary for the production tables.

    Include only changes relevant to the specific telescope.
    Do not include parameters if 'deprecated' flag is set to True.
    Parameters belonging to the configuration_sim_telarray collection are
    skipped here and handled separately.

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
    new_table_name = _get_production_table_key(table_name)
    new_params = {new_table_name: table_parameters}
    deprecated_params = []

    for param, data in changes[table_name].items():
        if names.get_collection_name_from_parameter_name(param) == "configuration_sim_telarray":
            continue
        if data.get("deprecated", False):
            _logger.info(f"Removing model parameter '{table_name} - {param}'")
            deprecated_params.append(param)
            new_params[new_table_name].pop(param, None)
        else:
            version = data["version"]
            _logger.info(f"Setting '{table_name} - {param}' to version {version}")
            new_params[new_table_name][param] = version

    return new_params, deprecated_params


def _apply_model_parameter_change(
    telescope,
    param,
    param_data,
    simulation_models_path,
    setting_workflows_git_tag,
    setting_workflows_git_repository,
):
    """Apply one model parameter change."""
    if param_data.get("activity_id") is not None and param_data.get("value") is not None:
        raise ValueError(
            f"Both activity_id and value are set for '{telescope} - {param}'. "
            "Provide only one source for model parameter content."
        )

    if param_data.get("activity_id") is not None:
        _download_model_parameter_from_workflow(
            telescope,
            param,
            param_data,
            simulation_models_path,
            setting_workflows_git_tag,
            setting_workflows_git_repository,
        )
    elif param_data.get("value") is not None:
        _create_new_model_parameter_entry(telescope, param, param_data, simulation_models_path)


def _format_model_parameter_entry(param_data):
    """Format model parameter data for an error message."""
    if isinstance(param_data, dict):
        return ", ".join(f"{key}={value!r}" for key, value in param_data.items())
    return f"param_data={param_data!r}"


def _process_model_parameter_change(
    telescope,
    param,
    param_data,
    simulation_models_path,
    setting_workflows_git_tag,
    setting_workflows_git_repository,
):
    """Apply one model parameter change and add context to failures."""
    try:
        _apply_model_parameter_change(
            telescope,
            param,
            param_data,
            simulation_models_path,
            setting_workflows_git_tag,
            setting_workflows_git_repository,
        )
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        entry_details = _format_model_parameter_entry(param_data)
        raise type(exc)(
            f"Failed to process info.yml entry '{telescope} -> {param}' ({entry_details}): {exc}"
        ) from exc


def _apply_changes_to_model_parameters(
    changes,
    simulation_models_path,
    setting_workflows_git_tag="main",
    setting_workflows_git_repository=DEFAULT_SIMULATION_WORKFLOWS,
):
    """
    Apply changes to model parameters by creating new parameter entries.

    Parameters
    ----------
    changes: dict
        The changes to be applied.
    simulation_models_path: Path
        Path to the simulation models directory.
    setting_workflows_git_tag: str
        Branch or tag used to download parameters from simulation workflow repository.
    setting_workflows_git_repository: str
        Repository URL used to download parameters from simulation workflow repository.

    Raises
    ------
    ValueError
        If both ``activity_id`` and ``value`` are provided for the same parameter.
    """
    for telescope, parameters in changes.items():
        for param, param_data in parameters.items():
            _process_model_parameter_change(
                telescope,
                param,
                param_data,
                simulation_models_path,
                setting_workflows_git_tag,
                setting_workflows_git_repository,
            )


def _download_model_parameter_from_workflow(
    telescope,
    param,
    param_data,
    simulation_models_path,
    setting_workflows_git_tag="main",
    setting_workflows_git_repository=DEFAULT_SIMULATION_WORKFLOWS,
):
    """
    Download model parameter entry from simulation workflow repository.

    Parameters
    ----------
    telescope: str
        Name of the telescope.
    param: str
        Name of the parameter.
    param_data: dict
        Dictionary containing the parameter data including version and activity_id.
    simulation_models_path: Path
        Path to the simulation models directory.
    setting_workflows_git_tag: str
        Branch or tag used to download parameters from simulation workflow repository.
    setting_workflows_git_repository: str
        Repository URL used to download parameters from simulation workflow repository.

    Raises
    ------
    TypeError
        If downloaded content is not a dictionary.
    ValueError
        If downloaded parameter_version does not match requested version.
    """
    source_file = (
        f"output/{telescope}/{param}/{param_data['activity_id']}/"
        f"{param}/{param}-{param_data['version']}.json"
    )
    _logger.info(f"Downloading model parameter '{telescope} - {param}' from '{source_file}'.")

    downloaded_data = ascii_handler.collect_data_from_git(
        file_name=source_file,
        git_repository=setting_workflows_git_repository,
        git_branch=setting_workflows_git_tag,
    )
    if not isinstance(downloaded_data, dict):
        raise TypeError(
            f"Downloaded model parameter is of type {type(downloaded_data)} "
            f"for '{telescope} - {param}'."
        )

    downloaded_version = downloaded_data.get("parameter_version")
    if downloaded_version != param_data["version"]:
        raise ValueError(
            f"Version mismatch for '{telescope} - {param}': requested "
            f"'{param_data['version']}', downloaded '{downloaded_version}'."
        )

    target_scope = _get_model_parameter_scope(telescope, param)
    target_dir = get_model_parameter_file_path(
        simulation_models_path, target_scope, param, param_data["version"]
    ).parent
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / f"{param}-{param_data['version']}.json"
    if target_scope == "global":
        downloaded_data["instrument"] = None
        downloaded_data["site"] = None
    writer.ModelDataWriter.write_model_parameter_json(downloaded_data, target_file)


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
    target_scope = _get_model_parameter_scope(telescope, param)
    param_dir = get_model_parameter_file_path(
        simulation_models_path, target_scope, param, param_data["version"]
    ).parent
    if not param_dir.exists():
        _logger.info(
            f"Create directory for model parameter '{telescope} - {param}': '{param_dir}'."
        )
        param_dir.mkdir(parents=True, exist_ok=True)
    try:
        latest_file = _get_latest_model_parameter_file(param_dir, param, param_data["version"])
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

    target_file = param_dir / f"{param}-{param_data['version']}.json"
    if target_file.exists():
        _validate_existing_model_parameter_file(
            target_file,
            None if target_scope == "global" else target_scope,
            param,
            param_data,
        )
        _logger.info("Model parameter file already matches requested version: '%s'.", target_file)
        return

    writer.ModelDataWriter.write_model_parameter(
        parameter_name=param,
        value=param_data["value"],
        instrument=None if target_scope == "global" else target_scope,
        parameter_version=param_data["version"],
        output_file=f"{param}-{param_data['version']}.json",
        output_path=param_dir,
        unit=param_data.get("unit"),
        model_parameter_schema_version=param_data.get("model_parameter_schema_version", None),
    )


def _validate_existing_model_parameter_file(target_file, instrument, param, param_data):
    """Validate an existing generated model parameter file before reusing it."""
    existing_data = ascii_handler.collect_data_from_file(target_file)
    expected_data = {
        "parameter": param,
        "instrument": instrument,
        "parameter_version": param_data["version"],
        "value": param_data["value"],
        "unit": _normalize_units_for_comparison(param_data.get("unit")),
    }
    existing_data["unit"] = _normalize_units_for_comparison(existing_data.get("unit"))
    mismatches = {
        key: (existing_data.get(key), value)
        for key, value in expected_data.items()
        if existing_data.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"Existing model parameter file '{target_file}' does not match the requested "
            f"value for '{instrument} - {param}': {mismatches}"
        )


def _normalize_units_for_comparison(unit):
    """Normalize equivalent scalar and per-value unit representations."""
    unit = value_conversion.normalize_dimensionless_unit(unit)
    if isinstance(unit, list):
        unit = [_normalize_units_for_comparison(entry) for entry in unit]
        if unit and all(entry == unit[0] for entry in unit):
            return unit[0]
        return unit
    return value_conversion.normalize_model_parameter_unit(0, unit)


def _get_latest_model_parameter_file(directory, parameter, max_version):
    """
    Get the latest model parameter JSON file for a parameter in the given directory.

    Assume files are named in the format 'parameter-version.json'.

    Parameters
    ----------
    directory: str
        Path to the directory containing parameter JSON files.
    parameter: str
        Name of the parameter to find.
    max_version: str
        Maximum version to consider (inclusive). Files with versions greater than
        this will be excluded.

    Returns
    -------
    str
        Path to the latest JSON file for the parameter with version <= max_version.

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

    max_ver = parse_version(max_version)
    filtered_files = [f for f in files if extract_version(f) <= max_ver]

    if not filtered_files:
        raise FileNotFoundError(
            f"No JSON files found for parameter '{parameter}' with version <= {max_version} "
            f"in directory '{directory}'."
        )

    latest_file = max(filtered_files, key=extract_version)
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
