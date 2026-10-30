"""Tools for running applications in the simtools framework."""

import glob
import hashlib
import importlib
import json
import logging
import os
import re
import shutil
import subprocess
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import yaml

import simtools.utils.general as gen
from simtools import dependencies
from simtools import version as simtools_version
from simtools.constants import RUN_TIME_ENVIRONMENT_SCHEMA
from simtools.data_model import schema, workflow_metadata
from simtools.io import ascii_handler
from simtools.io.file_type import product_filename
from simtools.job_execution import execution
from simtools.job_execution.execution import JobExecutionError
from simtools.utils import value_conversion

logger = logging.getLogger(__name__)
_SIMULATION_MODELS_PATH = "simulation_models_path"
_SIMULATION_MODELS_GIT_PATH = "simulation_models_git_path"
_IGNORE_EXISTING_PARAMETER_VERSION = "ignore_existing_parameter_version"
_MODEL_SOURCE_OPTIONS = (
    _SIMULATION_MODELS_PATH,
    _SIMULATION_MODELS_GIT_PATH,
    "simulation_models_git_revision",
)
_SCIENCE_PLACEHOLDER = re.compile(r"__[A-Z][A-Z0-9_]+__")


def prepare_workflow(args_dict, replacements=None):
    """Resolve a workflow and its backend configuration without executing it."""
    replacements = dict(replacements or {})
    replacements.setdefault(
        "__CONFIG_DIRECTORY__", str(Path(args_dict["config_file"]).resolve().parent)
    )
    configurations, runtime, log_file, activity, collection = _read_application_configuration(
        args_dict["config_file"],
        args_dict.get("steps"),
        args_dict.get("activity_id"),
        replacements=replacements,
    )
    resolve_nested_configuration_files(configurations, replacements)
    if args_dict.get("runtime_environment") is not None:
        runtime = args_dict["runtime_environment"]
    if args_dict.get("log_file") is not None:
        log_file = args_dict["log_file"]
    _validate_resolved_configuration(configurations, runtime, collection)
    return configurations, runtime, log_file, activity, collection


def run_applications(args_dict, run_time=None, replacements=None, context=None):
    """
    Run simtools applications step-by-step as defined in a configuration file.

    Parameters
    ----------
    args_dict : dict
        Command line arguments. Optional keys: ``overwrite_collection_files`` (allow
        overwriting collection files with the same basename) and ``provenance_path``
        (False omits the resolved workflow file).
    run_time : list or None
        Prepared runtime command, reused instead of preparing the runtime environment.
    replacements : dict or None
        Placeholder replacements applied recursively; ``__CONFIG_DIRECTORY__`` is added
        automatically.
    context : RunContext, optional
        Run context used to report software versions and collect workflow metadata.
    """
    (
        configurations,
        runtime_environment,
        log_file,
        workflow_activity_id,
        collection_config,
    ) = prepare_workflow(args_dict, replacements=replacements)
    workflow_start = datetime.now(UTC)
    associated_activities = []
    runtime_environment_snapshot = deepcopy(runtime_environment)
    model_parameter_metadata_files = []
    source_options = model_source_options(args_dict)

    apply_execution_overrides(
        configurations, args_dict, source_options, get_explicit_env_file(args_dict)
    )
    effective_runtime = None if args_dict["ignore_runtime_environment"] else runtime_environment
    _write_resolved_workflow(
        args_dict["config_file"],
        configurations,
        effective_runtime,
        collection_config,
        provenance_path=args_dict.get("provenance_path"),
    )
    if args_dict["ignore_runtime_environment"]:
        run_time = []
    elif run_time is None:
        run_time = read_runtime_environment(runtime_environment)

    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as file:
        file.write("Running simtools applications\n")
        try:
            versions = dependencies.get_version_string(
                run_time, include_software_versions=True, context=context
            )
        except FileNotFoundError as exc:
            logger.info("Host simulation versions unavailable; using runtime identity: %s", exc)
            versions = dependencies.get_version_string(
                run_time, include_software_versions=False, context=context
            )
        file.write(versions)
        try:
            manifest = dependencies.get_dependency_manifest(run_time or None, context=context)
            file.write(f"Runtime simtools: {manifest.get('simtools', {})}\n")
            file.write(f"Runtime build options: {manifest.get('build_options', {})}\n")
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            file.write(f"Runtime identity unavailable: {exc}\n")
        try:
            for config in configurations:
                app = config.get("application")
                if not config.get("run_application"):
                    logger.info(f"Skipping application: {app}")
                    continue

                app_configuration = config.get("configuration", {})
                app_activity_id = app_configuration.get("activity_id") or gen.get_uuid()
                app_configuration["activity_id"] = app_activity_id
                app_configuration.setdefault("label", app)
                app_configuration["disable_log_file"] = True

                associated_activities.append({"activity_name": app, "activity_id": app_activity_id})

                result = _submit_application_and_collect_metadata(
                    app=app,
                    app_configuration=app_configuration,
                    runtime_environment=run_time,
                    model_parameter_metadata_files=model_parameter_metadata_files,
                )

                file.write("=" * 80 + "\n")
                file.write(
                    f"Application: {app}\nSTDOUT:\n{result.output}\n"
                    f"STDERR:\n{result.error_output}\n"
                )

            _copy_collection_files(
                configurations,
                collection_config,
                overwrite_files=args_dict.get("overwrite_collection_files", False),
            )
        finally:
            _update_workflow_metadata_files(
                args_dict=args_dict,
                workflow_activity_id=workflow_activity_id,
                workflow_start=workflow_start,
                runtime_environment_snapshot=runtime_environment_snapshot,
                configurations=configurations,
                model_parameter_metadata_files=model_parameter_metadata_files,
                associated_activities=associated_activities,
                context=context,
            )


def get_explicit_env_file(args_dict):
    """Return the runner env file when it was explicitly supplied on the CLI."""
    configuration_sources = args_dict.get("_metadata_configuration_sources", {})
    if configuration_sources.get("env_file") != "cli":
        return None
    return args_dict.get("env_file")


def _copy_collection_files(configurations, collection_config, overwrite_files=False):
    """Copy listed files to collection output paths, raising FileExistsError on basename clashes."""
    if not collection_config:
        return
    entries = [collection_config] if isinstance(collection_config, dict) else collection_config

    for entry in entries:
        output_path = entry.get("output_path")
        files = entry.get("files") or []
        if output_path is None or not files:
            continue
        source_directories = _collect_source_directories(
            configurations, source_directory=entry.get("source_directory")
        )
        collection_output_path = Path(output_path)
        collection_output_path.mkdir(parents=True, exist_ok=True)
        inventory = []
        for file_spec in files:
            inventory.extend(
                _copy_pattern_files(
                    file_spec,
                    source_directories,
                    collection_output_path,
                    overwrite_files=overwrite_files,
                    preserve_relative_paths=entry.get("preserve_relative_paths", False),
                )
            )
        if entry.get("write_inventory"):
            inventory_path = Path(
                entry.get("inventory_file", collection_output_path / "inventory.json")
            )
            inventory_path.parent.mkdir(parents=True, exist_ok=True)
            inventory_path.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")


def _copy_pattern_files(
    pattern, source_directories, destination, overwrite_files=False, preserve_relative_paths=False
):
    """Copy all files matching a pattern from the source directories into a destination."""
    destination_name = None
    if isinstance(pattern, dict):
        destination_name = pattern.get("destination")
        pattern = pattern["source"]
    _validate_collection_destination_name(destination_name)

    if pattern == "**/comparison_statistics.meta.yml" and any(
        (source_directory / product_filename("comparison_statistics")).is_file()
        for source_directory in source_directories
    ):
        return []

    source_files = _find_collection_files(pattern, source_directories)
    if destination_name is not None and len(source_files) != 1:
        raise FileExistsError(
            f"Collection destination '{destination_name}' requires exactly one source file, "
            f"but pattern '{pattern}' matched {len(source_files)} files."
        )

    inventory = []
    for source_file in source_files:
        relative_path = _collection_relative_path(source_file, source_directories)
        name = relative_path if preserve_relative_paths else source_file.name
        dest = destination / (destination_name or name)
        if not overwrite_files and dest.exists() and dest.resolve() != source_file.resolve():
            raise FileExistsError(
                f"Filename collision in collection: '{source_file.name}' would be "
                f"overwritten by '{source_file}'. Ensure output files have unique names."
            )
        if dest.resolve() != source_file.resolve():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_file, dest)
        inventory.append(
            {
                "source": str(relative_path),
                "destination": str(dest.relative_to(destination)),
                "sha256": _sha256(source_file),
            }
        )
    return inventory


def _collection_relative_path(source_file, source_directories):
    """Return a portable path within the matching collection source root."""
    for root in source_directories:
        if source_file.resolve().is_relative_to(root.resolve()):
            return source_file.resolve().relative_to(root.resolve())
    raise ValueError(f"Collection source is outside its declared roots: {source_file}")


def _sha256(path):
    """Return the SHA-256 checksum of one collected artifact."""
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _validate_collection_destination_name(destination_name):
    """Validate an optional collection destination as a plain filename."""
    if destination_name is None:
        return
    if (
        not isinstance(destination_name, str)
        or destination_name in {"", ".", ".."}
        or Path(destination_name).name != destination_name
        or "\\" in destination_name
    ):
        raise ValueError(
            "Collection destination must be a non-empty filename without path separators: "
            f"{destination_name!r}"
        )


def _collect_source_directories(configurations, source_directory=None):
    """Return unique source directories from application configurations."""
    if source_directory is None:
        keys = ["output_path"]
    elif isinstance(source_directory, str):
        keys = [source_directory]
    else:
        keys = list(source_directory)
    source_directories = []
    for config in configurations:
        app_configuration = config.get("configuration", {})
        for source_key in keys:
            source_dir = app_configuration.get(source_key)
            if source_dir is not None and Path(source_dir) not in source_directories:
                source_directories.append(Path(source_dir))
    return source_directories


def _find_collection_files(pattern, source_directories):
    """Find files matching a filename or glob pattern in the source directories."""
    if glob.has_magic(pattern):
        matched = sorted(f for d in source_directories for f in d.rglob(pattern) if f.is_file())
        if not matched:
            logger.warning(
                f"No files matched collection pattern '{pattern}' in {source_directories}."
            )
        return matched

    recursive_matches = []
    for source_directory in source_directories:
        candidate = source_directory / pattern
        if candidate.exists():
            return [candidate]
        recursive_matches.extend(
            file_path for file_path in source_directory.rglob(pattern) if file_path.is_file()
        )

    if len(recursive_matches) == 1:
        return recursive_matches
    if len(recursive_matches) > 1:
        raise FileExistsError(
            f"Collection file '{pattern}' matched multiple nested files in {source_directories}: "
            f"{recursive_matches}. Use a more specific path or glob pattern."
        )
    raise FileNotFoundError(f"Could not find collection file '{pattern}' in {source_directories}.")


def model_source_options(args_dict):
    """Return configured model-source options for child applications."""
    return {key: args_dict[key] for key in _MODEL_SOURCE_OPTIONS if args_dict.get(key) is not None}


def _application_argument_names(application):
    """Return the argument names of a simtools application, or None if it is not one."""
    if not application or not application.startswith("simtools-"):
        return None

    module_name = "simtools.applications." + application.removeprefix("simtools-").replace("-", "_")
    try:
        definition = importlib.import_module(module_name).APPLICATION
    except ImportError, AttributeError:
        return None
    return {argument.name for argument in definition.all_arguments}


def _apply_model_source_options(application, configuration, source_options):
    """Apply inherited model-source options to applications that accept them."""
    argument_names = _application_argument_names(application)
    if argument_names is None or not set(source_options) <= argument_names:
        return
    if configuration.get(_SIMULATION_MODELS_PATH):
        return
    for key, value in source_options.items():
        if key == _SIMULATION_MODELS_PATH and configuration.get(_SIMULATION_MODELS_GIT_PATH):
            continue
        if configuration.get(key) is None:
            configuration[key] = value


def _submit_application_and_collect_metadata(
    app,
    app_configuration,
    runtime_environment,
    model_parameter_metadata_files,
):
    """Submit one application and collect metadata file before/after submission."""
    metadata_file = _get_model_parameter_metadata_file(app_configuration)
    if metadata_file is not None:
        model_parameter_metadata_files.append(metadata_file)

    logger.info(f"Running application: {app}")
    result = execution.run_command(
        app,
        stdout=None,
        stderr=None,
        configuration=app_configuration,
        runtime_environment=runtime_environment,
    )

    if metadata_file is None:
        metadata_file = _get_model_parameter_metadata_file(app_configuration)
        if metadata_file is not None:
            model_parameter_metadata_files.append(metadata_file)
    return result


def _update_workflow_metadata_files(
    args_dict,
    workflow_activity_id,
    workflow_start,
    runtime_environment_snapshot,
    configurations,
    model_parameter_metadata_files,
    associated_activities,
    context=None,
):
    """Update collected model-parameter metadata files with workflow metadata."""
    if not model_parameter_metadata_files:
        return

    workflow_activity = workflow_metadata.build_workflow_activity_metadata(
        args_dict=args_dict,
        workflow_activity_id=workflow_activity_id,
        workflow_start=workflow_start,
        workflow_end=max(datetime.now(UTC), workflow_start),
        runtime_environment=(
            runtime_environment_snapshot if not args_dict["ignore_runtime_environment"] else None
        ),
        workflow_context=_get_workflow_context(configurations),
        context=context,
    )
    for metadata_file in model_parameter_metadata_files:
        workflow_metadata.update_model_parameter_metadata_file(
            metadata_file=metadata_file,
            workflow_activity=workflow_activity,
            associated_activities=associated_activities,
        )


def _read_application_configuration(
    configuration_file, steps, workflow_activity_id=None, replacements=None
):
    """Read a workflow file and resolve its placeholders; return configurations and metadata."""
    job_configuration = ascii_handler.collect_data_from_file(configuration_file)
    job_configuration = value_conversion.replace_placeholders_recursively(
        job_configuration, replacements or {}
    )
    workflow_activity_id = (
        gen.extract_uuid7_from_path(configuration_file) or workflow_activity_id or gen.get_uuid()
    )
    derived_output_path, setting_workflow = _set_input_output_directories(configuration_file)
    configurations = job_configuration.get("applications")
    collection_config = value_conversion.replace_placeholders_recursively(
        job_configuration.get("collection"),
        {"__SETTING_WORKFLOW__": setting_workflow},
    )

    output_path_used_as_default = False
    for step_count, config in enumerate(configurations, start=1):
        config["run_application"] = step_count in steps if steps else True
        config = value_conversion.change_dict_keys_case(config, True)
        app_config = config.get("configuration", {})
        if "output_path" not in app_config:
            output_path_used_as_default = True
        config["configuration"] = _prepare_application_configuration(
            app_config,
            derived_output_path,
            setting_workflow,
        )
        if config["configuration"].get("activity_id") is None:
            config["configuration"]["activity_id"] = gen.get_uuid()
        configurations[step_count - 1] = config

    if output_path_used_as_default or not configurations:
        log_path = derived_output_path
    else:
        first_app_output = configurations[0].get("configuration", {}).get("output_path")
        log_path = Path(first_app_output) if first_app_output else derived_output_path
    logger.debug(f"Setting workflow output path to {log_path}")

    return (
        configurations,
        job_configuration.get("runtime_environment"),
        log_path / "simtools.log",
        workflow_activity_id,
        collection_config,
    )


def _validate_resolved_configuration(configurations, runtime_environment, collection_config):
    """Reject unresolved workflow placeholders before starting an application."""
    values = [configurations, runtime_environment, collection_config]
    unresolved = sorted(
        {match.group(0) for value in values for match in _iter_placeholder_matches(value)}
    )
    if unresolved:
        raise ValueError("Unresolved workflow placeholder(s): " + ", ".join(unresolved))


def resolve_nested_configuration_files(configurations, replacements):
    """Resolve placeholders in referenced backend configuration files."""
    for config in configurations:
        app_configuration = config.get("configuration", {})
        backend_config = app_configuration.get("backend_config")
        if not isinstance(backend_config, (str, Path)):
            continue
        backend_path = Path(backend_config)
        if not backend_path.is_absolute():
            backend_path = Path(replacements["__CONFIG_DIRECTORY__"]) / backend_path
        if not backend_path.is_file():
            raise FileNotFoundError(f"Backend configuration not found: {backend_path}")
        backend_values = ascii_handler.collect_data_from_file(backend_path)
        if not isinstance(backend_values, dict):
            raise ValueError(f"Backend configuration must contain a mapping: {backend_path}")
        app_configuration["backend_config"] = value_conversion.replace_placeholders_recursively(
            backend_values,
            replacements,
        )


def _iter_placeholder_matches(value):
    """Yield placeholder matches from nested workflow values."""
    if isinstance(value, str):
        yield from _SCIENCE_PLACEHOLDER.finditer(value)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_placeholder_matches(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_placeholder_matches(item)


def _write_resolved_workflow(
    configuration_file, configurations, runtime_environment, collection, provenance_path=None
):
    """Write execution provenance at an explicit path or below the first output."""
    if provenance_path is False:
        return
    output_path = next(
        (
            config.get("configuration", {}).get("output_path")
            for config in configurations
            if config.get("configuration", {}).get("output_path")
        ),
        None,
    )
    if output_path is None and provenance_path is None:
        return
    provenance_path = (
        Path(provenance_path)
        if provenance_path
        else Path(output_path) / "provenance" / "resolved-workflow.yml"
    )
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "configuration_file": Path(configuration_file).name,
        "applications": configurations,
        "runtime_environment": runtime_environment,
        "collection": collection,
    }
    with provenance_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            ascii_handler.to_builtin(payload), file, sort_keys=False, explicit_start=True
        )


def _get_model_parameter_metadata_file(app_configuration):
    """Return the expected metadata file of a model-parameter submission application."""
    output_path = app_configuration.get("output_path")
    if not output_path:
        return None

    output_path = Path(output_path)
    parameter = app_configuration.get("parameter")
    parameter_version = app_configuration.get("parameter_version")

    if parameter and parameter_version:
        return output_path / parameter / f"{parameter}-{parameter_version}.meta.yml"

    candidates = sorted(output_path.rglob("*.meta.yml"))
    if len(candidates) == 1:
        return candidates[0]
    matching = [
        f
        for f in candidates
        if parameter_version and f.name.endswith(f"-{parameter_version}.meta.yml")
    ]
    return matching[0] if len(matching) == 1 else None


def _get_workflow_context(configurations):
    """Return the site and instrument of the workflow (first non-empty value)."""

    def first_value(key):
        return next(
            (
                value
                for value in (config.get("configuration", {}).get(key) for config in configurations)
                if value is not None
            ),
            None,
        )

    return {
        "site": first_value("site"),
        "instrument": first_value("instrument") or first_value("telescope"),
    }


def _replace_placeholders_in_configuration(
    configuration, output_path, setting_workflow, place_holder="__SETTING_WORKFLOW__"
):
    """Replace the workflow placeholder and default the output path of a configuration."""
    configuration = value_conversion.replace_placeholders_recursively(
        configuration,
        {place_holder: setting_workflow},
    )
    if output_path:
        configuration.setdefault("output_path", str(output_path))
    return configuration


def _prepare_application_configuration(configuration, output_path, setting_workflow):
    """Resolve placeholders and version-dependent values in a workflow configuration."""
    configuration = _replace_placeholders_in_configuration(
        configuration,
        output_path,
        setting_workflow,
    )
    model_version = configuration.get("model_version")
    if model_version is not None:
        configuration = simtools_version.resolve_by_version(configuration, model_version)
    return configuration


def _set_input_output_directories(path):
    """Return the output directory and workflow subdirectory for a configuration file."""
    path = Path(path)
    try:
        setting_workflow = gen.extract_subdirectories_from_path(path, anchor="input")
    except ValueError:
        setting_workflow = str(path.parent) if path.parent != Path() else path.stem

    return Path("simtools-output") / Path(setting_workflow), setting_workflow


def read_runtime_environment(runtime_environment):
    """Read the runtime environment (e.g. docker runtime) and return the required command."""
    if runtime_environment is None:
        return []

    engine = runtime_environment.get("container_engine", "docker")
    if shutil.which(engine) is None:
        raise RuntimeError(f"Container engine '{engine}' not found.")
    if Path(engine).name == "apptainer":
        if runtime_environment.get("network"):
            raise ValueError("Apptainer runtime environments do not support the network option.")
        cmd = [engine, "exec"]
        for option in runtime_environment.get("options", []):
            cmd.extend(option.split())
        if environment_file := runtime_environment.get("environment_file"):
            cmd.extend(["--env-file", environment_file])
        return [*cmd, runtime_environment["image"]]

    cmd = [engine, "run", "--rm"]

    if options := runtime_environment.get("options"):
        for opt in options:
            cmd.extend(os.path.expandvars(opt).split())

    if env := runtime_environment.get("environment_file"):
        cmd += ["--env-file", env]
    if net := runtime_environment.get("network"):
        cmd += ["--network", net]

    cmd.append(runtime_environment["image"])
    _pull_image(engine, runtime_environment["image"])

    return cmd


def prepare_runtime_environment(runtime_environment_file):
    """
    Read and prepare a standalone runtime environment.

    Parameters
    ----------
    runtime_environment_file : str or pathlib.Path
        YAML file with a top-level ``runtime_environment`` mapping.

    Returns
    -------
    tuple[dict, list[str]]
        Runtime-environment configuration and prepared runtime command.

    Raises
    ------
    ValueError
        If the YAML content is not a mapping or lacks ``runtime_environment``.
    jsonschema.ValidationError
        If the configuration does not conform to the runtime-environment schema.
    """
    runtime_config = ascii_handler.collect_data_from_file(runtime_environment_file)
    if not isinstance(runtime_config, dict):
        raise ValueError("Runtime configuration must be a YAML mapping.")
    runtime_environment = runtime_config.get("runtime_environment")
    if runtime_environment is None:
        raise ValueError("Runtime configuration must contain a 'runtime_environment' block.")
    schema.validate_dict_using_schema(
        runtime_config, schema_file=RUN_TIME_ENVIRONMENT_SCHEMA, offline=True
    )
    return runtime_environment, read_runtime_environment(runtime_environment)


def _pull_image(engine, image):
    """Pull an image with the container engine unless it is already available locally."""
    inspect_result = execution.run_command([engine, "image", "inspect", image], check=False)
    if inspect_result and inspect_result.return_code == 0:
        return

    try:
        execution.run_command([engine, "pull", image], capture_output=False)
    except JobExecutionError as exc:
        raise RuntimeError(f"Failed to pull image '{image}' using '{engine}': {exc}") from exc


def apply_execution_overrides(configurations, args, model_source, env_file):
    """Apply inherited settings before recording or running applications."""
    for config in configurations:
        configuration = config.get("configuration", {})
        application = config.get("application")
        _apply_model_source_options(application, configuration, model_source)
        if args.get(_IGNORE_EXISTING_PARAMETER_VERSION):
            argument_names = _application_argument_names(application)
            if argument_names is None or _IGNORE_EXISTING_PARAMETER_VERSION in argument_names:
                configuration[_IGNORE_EXISTING_PARAMETER_VERSION] = True
        if env_file is not None:
            configuration["env_file"] = env_file
