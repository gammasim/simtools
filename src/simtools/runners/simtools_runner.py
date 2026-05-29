"""Tools for running applications in the simtools framework."""

import logging
import shutil
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import simtools.utils.general as gen
from simtools import dependencies
from simtools.data_model import workflow_metadata
from simtools.io import ascii_handler
from simtools.job_execution import job_manager

logger = logging.getLogger(__name__)


def run_applications(args_dict):
    """
    Run simtools applications step-by-step as defined in a configuration file.

    Parameters
    ----------
    args_dict : dict
        Dictionary containing command line arguments.
    """
    (
        configurations,
        runtime_environment,
        log_file,
        workflow_activity_id,
    ) = _read_application_configuration(
        args_dict["config_file"],
        args_dict.get("steps"),
        args_dict.get("activity_id"),
    )

    collection_config = None
    try:
        collection_config = ascii_handler.collect_data_from_file(args_dict["config_file"]).get(
            "collection"
        )
    except (OSError, TypeError):
        logger.debug("Could not read collection configuration from workflow file.")

    workflow_start = datetime.now(UTC)
    associated_activities = []
    runtime_environment_snapshot = deepcopy(runtime_environment)
    model_parameter_metadata_files = []

    run_time = (
        read_runtime_environment(runtime_environment)
        if not args_dict["ignore_runtime_environment"]
        else []
    )

    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as file:
        file.write("Running simtools applications\n")
        file.write(dependencies.get_version_string(run_time, include_software_versions=False))
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
                    f"Application: {app}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\n"
                )

            _copy_collection_files(configurations, collection_config)
        finally:
            _update_workflow_metadata_files(
                args_dict=args_dict,
                workflow_activity_id=workflow_activity_id,
                workflow_start=workflow_start,
                runtime_environment_snapshot=runtime_environment_snapshot,
                configurations=configurations,
                model_parameter_metadata_files=model_parameter_metadata_files,
                associated_activities=associated_activities,
            )


def _copy_collection_files(configurations, collection_config):
    """Copy listed files from application output paths to collection output path."""
    if not collection_config:
        return

    output_path = collection_config.get("output_path")
    files = collection_config.get("files") or []
    if output_path is None or not files:
        return

    source_directories = _collect_source_directories(configurations)
    collection_output_path = Path(output_path)
    collection_output_path.mkdir(parents=True, exist_ok=True)

    for file_name in files:
        source_file = _find_collection_file(file_name, source_directories)
        shutil.copy2(source_file, collection_output_path / file_name)


def _collect_source_directories(configurations):
    """Return unique output directories from application configurations."""
    source_directories = []
    for config in configurations:
        source_dir = config.get("configuration", {}).get("output_path")
        if source_dir is not None:
            source_directory = Path(source_dir)
            if source_directory not in source_directories:
                source_directories.append(source_directory)
    return source_directories


def _find_collection_file(file_name, source_directories):
    """Find a named file in the list of source directories.

    Parameters
    ----------
    file_name : str
        File name to locate.
    source_directories : list
        Directories to search in order.

    Returns
    -------
    Path
        Path to the found file.

    Raises
    ------
    FileNotFoundError
        If the file is not found in any source directory.
    """
    for source_directory in source_directories:
        candidate = source_directory / file_name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find collection file '{file_name}' in {source_directories}."
    )


def _append_metadata_file(model_parameter_metadata_files, metadata_file):
    """Append metadata file to list when available."""
    if metadata_file is not None:
        model_parameter_metadata_files.append(metadata_file)


def _submit_application_and_collect_metadata(
    app,
    app_configuration,
    runtime_environment,
    model_parameter_metadata_files,
):
    """Submit one application and collect metadata file before/after submission."""
    metadata_file = _get_model_parameter_metadata_file(app_configuration)
    _append_metadata_file(model_parameter_metadata_files, metadata_file)

    logger.info(f"Running application: {app}")
    result = job_manager.submit(
        app,
        out_file=None,
        err_file=None,
        configuration=app_configuration,
        runtime_environment=runtime_environment,
    )

    if metadata_file is None:
        _append_metadata_file(
            model_parameter_metadata_files,
            _get_model_parameter_metadata_file(app_configuration),
        )

    return result


def _update_workflow_metadata_files(
    args_dict,
    workflow_activity_id,
    workflow_start,
    runtime_environment_snapshot,
    configurations,
    model_parameter_metadata_files,
    associated_activities,
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
    )
    for metadata_file in model_parameter_metadata_files:
        workflow_metadata.update_model_parameter_metadata_file(
            metadata_file=metadata_file,
            workflow_activity=workflow_activity,
            associated_activities=associated_activities,
        )


def _read_application_configuration(configuration_file, steps, workflow_activity_id=None):
    """
    Read application configuration from file and modify for setting workflows.

    Strong assumptions on the structure of input and output files:

    - configuration file is expected to be in './input/<workflow directory>/<yaml file>'
    - output files will be written out to './output/<workflow directory>/'

    Replaces the placeholders in the configuration file with the actual values.

    Parameters
    ----------
    configuration_file : str
        Configuration file name.
    steps : list
        List of steps to be executed (None: all steps).
    workflow_activity_id : str
        Workflow activity id fallback from command-line context.

    Returns
    -------
    dict
        Application configuration.
    dict:
        Runtime environment configuration.
    Path
        Path to the log file.
    str
        Workflow activity id.

    """
    job_configuration = ascii_handler.collect_data_from_file(configuration_file)
    workflow_activity_id = (
        gen.extract_uuid7_from_path(configuration_file) or workflow_activity_id or gen.get_uuid()
    )
    derived_output_path, setting_workflow = _set_input_output_directories(configuration_file)
    configurations = job_configuration.get("applications")

    output_path_used_as_default = False
    for step_count, config in enumerate(configurations, start=1):
        config["run_application"] = step_count in steps if steps else True
        config = gen.change_dict_keys_case(config, True)
        app_config = config.get("configuration", {})
        if "output_path" not in app_config:
            output_path_used_as_default = True
        config["configuration"] = _replace_placeholders_in_configuration(
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
    logger.info(f"Setting workflow output path to {log_path}")

    return (
        configurations,
        job_configuration.get("runtime_environment"),
        log_path / "simtools.log",
        workflow_activity_id,
    )


def _get_application_log_file(application, app_configuration, counter):
    """Return log file path for an application executed via run_applications."""
    if app_configuration.get("log_file") is not None:
        return app_configuration["log_file"]
    output_path = app_configuration.get("output_path")
    if output_path is None:
        return None
    return Path(output_path) / f"{application}-{counter:02d}.log"


def _get_model_parameter_metadata_file(app_configuration):
    """
    Return expected metadata file for model-parameter submission applications.

    Takes into account differences in how applications generate metadata files
    (e.g., submit-model-parameter style apps vs. more generic applications).
    """
    output_path = app_configuration.get("output_path")
    if not output_path:
        return None

    output_path = Path(output_path)
    parameter = app_configuration.get("parameter")
    parameter_version = app_configuration.get("parameter_version")

    if parameter and parameter_version:
        return output_path / parameter / f"{parameter}-{parameter_version}.meta.yml"

    metadata_candidates = sorted(output_path.rglob("*.meta.yml"))
    if len(metadata_candidates) == 1:
        return metadata_candidates[0]

    if parameter_version:
        matching = [
            file_path
            for file_path in metadata_candidates
            if file_path.name.endswith(f"-{parameter_version}.meta.yml")
        ]
        if len(matching) == 1:
            return matching[0]

    return None


def _get_workflow_configuration_value(configurations, key):
    """Return first non-empty configuration value for a given key."""
    for config in configurations:
        value = config.get("configuration", {}).get(key)
        if value is not None:
            return value
    return None


def _get_workflow_context(configurations):
    """Extract workflow context (site, instrument) from configurations.

    Parameters
    ----------
    configurations : list
        List of application configurations.

    Returns
    -------
    dict
        Context dict with 'site' and 'instrument' keys.
    """
    return {
        "site": _get_workflow_configuration_value(configurations, "site"),
        "instrument": _get_workflow_configuration_value(configurations, "instrument")
        or _get_workflow_configuration_value(configurations, "telescope"),
    }


def _replace_placeholders_in_configuration(
    configuration, output_path, setting_workflow, place_holder="__SETTING_WORKFLOW__"
):
    """
    Replace placeholders in the configuration dictionary.

    Parameters
    ----------
    configuration : dict
        Configuration dictionary.
    output_path : Path
        Path to the output directory.
    setting_workflow : str
        The setting workflow to replace the placeholder with.
    place_holder : str
        Placeholder to be replaced.

    Returns
    -------
    dict
        Configuration dictionary with placeholders replaced.
    """
    configuration = gen.replace_placeholders_recursively(
        configuration,
        {place_holder: setting_workflow},
    )
    if output_path:
        configuration.setdefault("output_path", str(output_path))

    return configuration


def _set_input_output_directories(path):
    """
    Set input and output directories based on the configuration file path.

    Tuned to simulation models setting workflows.

    Parameters
    ----------
    path : str or Path
        Path to the configuration file.

    Returns
    -------
    tuple
        The first part is the output directory, the second part is the subdirectory name.
    """
    path = Path(path)
    try:
        setting_workflow = gen.extract_subdirectories_from_path(path, anchor="input")
    except ValueError:
        if path.parent != Path():
            setting_workflow = str(path.parent)
        else:
            setting_workflow = path.stem

        logger.info(
            "Could not derive setting workflow from 'input' anchor; "
            f"using fallback '{setting_workflow}'"
        )

    output_path = Path("output") / Path(setting_workflow)
    return output_path, setting_workflow


def read_runtime_environment(runtime_environment, workdir="/workdir/external/"):
    """
    Read the runtime environment (e.g. docker runtime) and generate the required command.

    Parameters
    ----------
    runtime_environment : str or None
        Path to the runtime environment configuration file.

    Returns
    -------
    list
        Runtime command.
    """
    if runtime_environment is None:
        return []

    engine = runtime_environment.get("container_engine", "docker")
    if shutil.which(engine) is None:
        raise RuntimeError(f"Container engine '{engine}' not found.")
    cmd = [engine, "run", "--rm", "-v", f"{Path.cwd()}:{workdir}", "-w", workdir]

    if options := runtime_environment.get("options"):
        for opt in options:
            cmd.extend(opt.split())

    if env := runtime_environment.get("env_file"):
        cmd += ["--env-file", env]
    if net := runtime_environment.get("network"):
        cmd += ["--network", net]

    cmd.append(runtime_environment["image"])
    _pull_image(engine, runtime_environment["image"])

    return cmd


def _pull_image(engine, image):
    """
    Pull the specified image using the specified container engine.

    Parameters
    ----------
    engine : str
        Container engine to use (e.g., 'docker' or 'podman').
    image : str
        Image to pull.
    """
    inspect_result = job_manager.submit([engine, "image", "inspect", image], check=False)
    if inspect_result and inspect_result.returncode == 0:
        return

    try:
        job_manager.submit([engine, "pull", image], capture_output=False)
    except job_manager.JobExecutionError as exc:
        raise RuntimeError(f"Failed to pull image '{image}' using '{engine}': {exc}") from exc
