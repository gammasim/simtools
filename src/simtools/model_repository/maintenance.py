"""Utilities for managing the simulation models repository.

Verify production tables against model parameters, and generate new production
tables and model parameters from the changes defined in ``info.yml``.
"""

import logging
import urllib.error
import urllib.request
from pathlib import Path

from packaging.version import Version
from packaging.version import parse as parse_version

import simtools.data_model.model_data_writer as writer
from simtools.constants import DEFAULT_SIMULATION_WORKFLOWS
from simtools.data_model import parameter_catalog as catalog
from simtools.data_model.table_asset import resolve_asset_path
from simtools.io import ascii_handler
from simtools.utils import value_conversion

_logger = logging.getLogger(__name__)
_INFO_FILE_NAME = "info.yml"
_MODEL_UPDATE = "model_update"
_MODEL_VERSION_HISTORY = "model_version_history"
_PATCH_UPDATE = "patch_update"
_WORKFLOW_ASSET_TIMEOUT_SECONDS = 30


def get_production_directory(simulation_models_path, model_version=None):
    """
    Get the production directory for a model version.

    Parameters
    ----------
    simulation_models_path : str
        Path to the simulation models repository.
    model_version : str, optional
        Model version. If None, the productions root directory is returned.

    Returns
    -------
    Path
        Production directory.
    """
    base = Path(simulation_models_path) / "simulation-models/productions"
    return base / str(model_version) if model_version else base


def initialize_production(
    simulation_models_path, model_version, base_model_version, update_type=_PATCH_UPDATE
):
    """
    Create an initial production info file based on an existing model version.

    Parameters
    ----------
    simulation_models_path : str or Path
        Path to the simulation models repository.
    model_version : str
        Version to initialize.
    base_model_version : str
        Existing production version used as the starting point.
    update_type : str, optional
        Production update type to write to the new info file.

    Raises
    ------
    FileExistsError
        If the target production directory already exists.
    FileNotFoundError
        If the base production info file does not exist.
    ValueError
        If the update type is not supported.
    """
    if update_type not in {_PATCH_UPDATE, "full_update"}:
        raise ValueError(f"Unsupported production update type: {update_type}")

    target_directory = get_production_directory(simulation_models_path, model_version)
    source_info = get_production_directory(simulation_models_path, base_model_version) / (
        _INFO_FILE_NAME
    )
    if target_directory.exists():
        raise FileExistsError(
            f"Cannot initialize production {model_version}: target directory already exists: "
            f"{target_directory}"
        )
    if not source_info.is_file():
        raise FileNotFoundError(
            f"Cannot initialize production {model_version}: base production info file does not "
            f"exist: {source_info}"
        )

    history = [base_model_version]
    if update_type == _PATCH_UPDATE:
        base_info = ascii_handler.collect_data_from_file(source_info)
        if base_info.get(_MODEL_UPDATE) == _PATCH_UPDATE:
            history.extend(base_info.get(_MODEL_VERSION_HISTORY, []))

    info = {
        "model_version": model_version,
        _MODEL_UPDATE: update_type,
        _MODEL_VERSION_HISTORY: history,
        "model_status": "development",
        "description": f"Initial configuration based on production model {base_model_version}",
        "changes": {},
    }
    target_directory.mkdir(parents=True)
    ascii_handler.write_data_to_file(info, target_directory / _INFO_FILE_NAME)


def verify_simulation_model_production_tables(simulation_models_path):
    """
    Verify that all model parameters referenced by the production tables exist.

    Parameters
    ----------
    simulation_models_path : str
        Path to the simulation models repository.

    Returns
    -------
    bool
        True if all parameters found, False if any missing.
    """
    production_files = list(get_production_directory(simulation_models_path).rglob("*.json"))
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
    """Return (missing_files, total_checked_count) for one production table."""
    production_table = ascii_handler.collect_data_from_file(production_file)
    missing_files = []
    total_checked = 0
    for array_element, par_dict in production_table.get("parameters", {}).items():
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

    Null-instrument parameters are stored below the explicit ``global`` scope.

    Parameters
    ----------
    simulation_models_path : str
        Path to the simulation models repository.
    array_element : str
        Name of the array element (e.g., 'telescope'), or ``None`` for a global parameter.
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
        Path(simulation_models_path)
        / "simulation-models/model_parameters"
        / instrument
        / parameter_name
        / f"{parameter_name}-{parameter_version}.json"
    )


def _get_model_parameter_scope(telescope, parameter_name=None):
    """Return the filesystem scope for a production-table key and parameter."""
    if telescope in (None, "global", "configuration_corsika"):
        return "global"
    if catalog.is_global_sim_telarray_parameter(parameter_name):
        return "global"
    return telescope


def generate_new_production(
    model_version, simulation_models_path, setting_workflows_git_tag=None, context=None
):
    """
    Generate a new production definition (production tables and model parameters).

    Production tables are copied from the base model version and updated with the
    changes in 'info.yml' (full or patch update). New model parameter entries are
    generated for the changed parameters.

    Parameters
    ----------
    model_version: str
        Model version to be created or updated.
    simulation_models_path: str
        Path to the simulation models repository.
    setting_workflows_git_tag: str, optional
        Branch or tag used to download parameters from the simulation workflow repository.
        Overrides ``setting_workflows_git_tag`` from ``info.yml`` if given; defaults to
        ``"main"`` if neither is set.
    context : RunContext, optional
        Run configuration and model reader used to write model parameters.
    """
    modification_dict = _get_changes_dict(model_version, simulation_models_path)
    update_type = modification_dict.get(_MODEL_UPDATE, "full_update")
    if setting_workflows_git_tag is None:
        setting_workflows_git_tag = modification_dict.get("setting_workflows_git_tag", "main")
    setting_workflows_git_repository = modification_dict.get(
        "setting_workflows_git_repository", DEFAULT_SIMULATION_WORKFLOWS
    )
    changes, base_model_version = _get_changes_to_production(
        modification_dict, simulation_models_path, update_type
    )

    _apply_changes_to_production_tables(
        changes,
        base_model_version,
        modification_dict["model_version"],
        update_type,
        simulation_models_path,
    )
    _apply_changes_to_model_parameters(
        modification_dict.get("changes", {}),
        simulation_models_path,
        setting_workflows_git_tag,
        setting_workflows_git_repository,
        context=context,
    )


def _apply_changes_to_production_tables(
    changes, base_model_version, model_version, update_type, simulation_models_path
):
    """Apply changes to (or create) production tables and write them to the target directory."""
    source = get_production_directory(simulation_models_path, base_model_version)
    target = get_production_directory(simulation_models_path, model_version)
    _logger.info(f"Production tables {update_type} from {source} to {target}")
    target.mkdir(parents=True, exist_ok=True)

    tables = {}
    for file_path in Path(source).rglob("*.json"):
        data = ascii_handler.collect_data_from_file(file_path)
        if not isinstance(data, dict):
            raise TypeError(f"Unsupported data type {type(data)} in {file_path}")
        tables[data["production_table_name"]] = data

    for table_name in changes:
        tables.setdefault(table_name, {})
    tables.setdefault("configuration_sim_telarray", {})

    for table_name, data in tables.items():
        if table_name == "configuration_sim_telarray":
            has_changes = _apply_changes_to_sim_telarray_production_table(
                data, changes, model_version, update_type == "patch_update"
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

    Returns True if the table should be written (always for full updates).
    """
    data["model_version"] = model_version
    if table_name not in changes:
        return not patch_update

    production_key = "global" if table_name == "configuration_corsika" else table_name
    table_parameters = {} if patch_update else data.get("parameters", {}).get(production_key, {})
    parameters, deprecated = _update_parameters_dict(table_parameters, changes, table_name)
    if patch_update and not parameters.get(production_key) and not deprecated:
        return False
    data["parameters"] = parameters
    if deprecated and patch_update:
        data["deprecated_parameters"] = deprecated
    return True


def _get_changes_dict(model_version, simulation_models_path):
    """Load the changes dictionary from the 'info.yml' of a production directory."""
    return ascii_handler.collect_data_from_file(
        get_production_directory(simulation_models_path, model_version) / _INFO_FILE_NAME
    )


def _get_changes_to_production(
    modification_dict, simulation_models_path, update_type="full_update"
):
    """
    Prepare changes applied to production tables.

    For full updates, the changes of all versions in the history are combined,
    starting from the base version. Returns (changes, base_model_version).
    """
    model_version_history = modification_dict.get(_MODEL_VERSION_HISTORY, [])

    try:
        # oldest version is the base version
        base_model_version = min(set(model_version_history), key=Version)
    except ValueError:
        _logger.debug(f"Base model version not found in {model_version_history}")
        return {}, modification_dict.get("model_version")

    changes = modification_dict.get("changes", {})
    if update_type == _PATCH_UPDATE:
        return changes, base_model_version

    for version_mod in reversed(model_version_history):
        _changes_dict = _get_changes_dict(version_mod, simulation_models_path)
        _version_changes, base_model_version = _get_changes_to_production(
            _changes_dict, simulation_models_path, update_type="full_update"
        )
        changes = _update_two_levels_in_changes_dict(_version_changes, changes)
        # stop iterative loop after reaching first full version of production tables
        if _changes_dict.get(_MODEL_UPDATE, "full_update") == "full_update":
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


def _apply_global_sim_telarray_changes(global_params, sim_telarray):
    """Apply global configuration_sim_telarray changes; returns (has_changes, deprecated)."""
    has_changes = False
    deprecated = []
    for name, data in sim_telarray:
        if not catalog.is_global_sim_telarray_parameter(name):
            continue
        has_changes = True
        if data.get("deprecated", False):
            global_params.pop(name, None)
            deprecated.append(name)
        else:
            global_params[name] = data["version"]
    return has_changes, deprecated


def _apply_telescope_sim_telarray_changes(parameters, telescope, sim_telarray):
    """Apply telescope sim_telarray changes; returns (has_changes, deprecated)."""
    has_changes = False
    deprecated = []
    telescope_params = parameters.get(telescope)
    for name, data in sim_telarray:
        if catalog.is_global_sim_telarray_parameter(name):
            continue
        has_changes = True
        if data.get("deprecated", False):
            if telescope_params is not None:
                telescope_params.pop(name, None)
            deprecated.append(name)
            continue
        if telescope_params is None:
            telescope_params = {}
        telescope_params[name] = data["version"]
    if telescope_params is not None:
        parameters[telescope] = telescope_params
    return has_changes, deprecated


def _apply_sim_telarray_changes_for_scope(parameters, global_params, telescope, params):
    """
    Apply configuration_sim_telarray changes for one scope.

    Global parameters go to ``global_params``; the others to ``parameters[telescope]``.
    Returns (has_changes, global_deprecated, deprecated).
    """
    sim_telarray = [
        (name, data)
        for name, data in params.items()
        if catalog.get_collection_name_from_parameter_name(name) == "configuration_sim_telarray"
    ]
    global_changed, global_deprecated = _apply_global_sim_telarray_changes(
        global_params, sim_telarray
    )
    telescope_changed, deprecated = _apply_telescope_sim_telarray_changes(
        parameters, telescope, sim_telarray
    )
    return global_changed or telescope_changed, global_deprecated, deprecated


def _remove_migrated_global_parameters(parameters):
    """Remove migrated global parameters from all non-global scopes."""
    for scope, scope_parameters in parameters.items():
        if scope != "global" and isinstance(scope_parameters, dict):
            for parameter_name in catalog.SIM_TELARRAY_GLOBAL_PARAMETERS:
                scope_parameters.pop(parameter_name, None)


def _record_deprecated_parameters(data, names, patch_update):
    """Record deprecated parameter names in the production table for patch updates."""
    if names and patch_update:
        data.setdefault("deprecated_parameters", []).extend(names)


def _apply_changes_to_sim_telarray_production_table(data, changes, model_version, patch_update):
    """
    Apply configuration_sim_telarray parameter changes to the production table.

    Process-wide parameters are stored under ``global``; telescope-dependent ones by design.
    Returns True if at least one configuration_sim_telarray parameter was found.
    """
    data["model_version"] = model_version
    has_changes = False
    parameters = data.get("parameters", {})

    # migrated global parameters are removed from non-global scopes
    _remove_migrated_global_parameters(parameters)

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
        _record_deprecated_parameters(data, telescope_changes, patch_update)
    _record_deprecated_parameters(data, global_deprecated, patch_update)
    if not global_params:
        parameters.pop("global", None)
    data["parameters"] = parameters
    return has_changes


def _update_parameters_dict(table_parameters, changes, table_name):
    """
    Create the parameters dictionary for one production table.

    Only changes relevant to the table are included; deprecated parameters are
    removed and listed. configuration_sim_telarray parameters are skipped (handled
    separately). Returns (parameters_dict, deprecated_parameter_names).
    """
    new_table_name = "global" if table_name == "configuration_corsika" else table_name
    new_params = {new_table_name: table_parameters}
    deprecated_params = []

    for param, data in changes[table_name].items():
        if catalog.get_collection_name_from_parameter_name(param) == "configuration_sim_telarray":
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


def _raise_with_entry_details(exc, telescope, param, param_data):
    """Re-raise an info.yml processing error with the offending entry details."""
    if isinstance(param_data, dict):
        entry_details = ", ".join(f"{key}={value!r}" for key, value in param_data.items())
    else:
        entry_details = f"param_data={param_data!r}"
    raise type(exc)(
        f"Failed to process info.yml entry '{telescope} -> {param}' ({entry_details}): {exc}"
    ) from exc


def _apply_model_parameter_entry(
    telescope,
    param,
    param_data,
    simulation_models_path,
    setting_workflows_git_tag,
    setting_workflows_git_repository,
    context,
):
    """Create or download one model parameter entry from an info.yml change."""
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
        _create_new_model_parameter_entry(
            telescope, param, param_data, simulation_models_path, context=context
        )


def _apply_changes_to_model_parameters(
    changes,
    simulation_models_path,
    setting_workflows_git_tag="main",
    setting_workflows_git_repository=DEFAULT_SIMULATION_WORKFLOWS,
    context=None,
):
    """
    Apply changes to model parameters by creating new parameter entries.

    Raises
    ------
    ValueError
        If both ``activity_id`` and ``value`` are provided for the same parameter.
    """
    for telescope, parameters in changes.items():
        for param, param_data in parameters.items():
            try:
                _apply_model_parameter_entry(
                    telescope,
                    param,
                    param_data,
                    simulation_models_path,
                    setting_workflows_git_tag,
                    setting_workflows_git_repository,
                    context,
                )
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                _raise_with_entry_details(exc, telescope, param, param_data)


def _download_model_parameter_from_workflow(
    telescope,
    param,
    param_data,
    simulation_models_path,
    setting_workflows_git_tag="main",
    setting_workflows_git_repository=DEFAULT_SIMULATION_WORKFLOWS,
):
    """
    Download a model parameter entry from the simulation workflow repository.

    Raises
    ------
    TypeError
        If downloaded content is not a dictionary.
    ValueError
        If downloaded parameter_version does not match requested version.
    """
    target_scope = _get_model_parameter_scope(telescope, param)
    target_file = get_model_parameter_file_path(
        simulation_models_path, target_scope, param, param_data["version"]
    )
    source_file = (
        f"output/{telescope}/{param}/{param_data['activity_id']}/"
        f"{param}/{param}-{param_data['version']}.json"
    )
    if target_file.exists():
        _logger.info(
            "Model parameter file already exists for requested version: '%s'.", target_file
        )
        _copy_model_parameter_asset_from_workflow(
            ascii_handler.collect_data_from_file(target_file),
            target_file,
            source_file,
            setting_workflows_git_repository,
            setting_workflows_git_tag,
        )
        return

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

    target_file.parent.mkdir(parents=True, exist_ok=True)
    if target_scope == "global":
        downloaded_data["instrument"] = None
        downloaded_data["site"] = None
    writer.ModelDataWriter.write_model_parameter_json(downloaded_data, target_file)
    _copy_model_parameter_asset_from_workflow(
        downloaded_data,
        target_file,
        source_file,
        setting_workflows_git_repository,
        setting_workflows_git_tag,
    )


def _copy_model_parameter_asset_from_workflow(
    parameter_data, target_file, source_file, git_repository, git_branch
):
    """Copy a file asset referenced by a workflow parameter JSON into the model repository."""
    if not parameter_data.get("file"):
        return

    asset_value = parameter_data.get("value")
    if not isinstance(asset_value, str):
        raise ValueError(f"File model parameter must reference a relative filename: {asset_value}")

    target_asset = resolve_asset_path(asset_value, target_file)
    if target_asset.exists():
        if target_asset.is_file():
            return
        raise IsADirectoryError(f"Model parameter asset path is not a file: {target_asset}")

    source_asset = (Path(source_file).parent / asset_value).as_posix()
    repository_url = str(git_repository).rstrip("/")
    if "/-/raw" in repository_url:
        asset_url = f"{repository_url}/{git_branch}/{source_asset}"
    else:
        asset_url = f"{repository_url.removesuffix('.git')}/-/raw/{git_branch}/{source_asset}"

    try:
        with urllib.request.urlopen(asset_url, timeout=_WORKFLOW_ASSET_TIMEOUT_SECONDS) as response:
            asset_content = response.read()
    except urllib.error.HTTPError as exc:
        exc.close()
        raise FileNotFoundError(
            f"Failed to download model parameter asset from {asset_url}"
        ) from exc

    target_asset.parent.mkdir(parents=True, exist_ok=True)
    target_asset.write_bytes(asset_content)


def _create_new_model_parameter_entry(
    telescope, param, param_data, simulation_models_path, context=None
):
    """
    Create a new model parameter entry in the model parameters directory.

    If a model parameter file exists, the latest version is used as template and
    the fields are updated. Otherwise, a new file is generated from the schema.
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
        context=context,
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
    Get the latest model parameter JSON file with version <= ``max_version``.

    Files are named 'parameter-version.json'.

    Raises
    ------
    FileNotFoundError
        If no matching files are found in the directory.
    """
    files = list(Path(directory).glob(f"{parameter}-*.json"))
    if not files:
        raise FileNotFoundError(
            f"No JSON files found for parameter '{parameter}' in directory '{directory}'."
        )

    def extract_version(path):
        return parse_version(path.stem.split("-", 1)[1])

    max_ver = parse_version(max_version)
    filtered_files = [f for f in files if extract_version(f) <= max_ver]
    if not filtered_files:
        raise FileNotFoundError(
            f"No JSON files found for parameter '{parameter}' with version <= {max_version} "
            f"in directory '{directory}'."
        )
    return str(max(filtered_files, key=extract_version))


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
