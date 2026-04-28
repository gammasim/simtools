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
    workflow_start = datetime.now(UTC)
    associated_activities = []
    runtime_environment_snapshot = deepcopy(runtime_environment)
    model_parameter_metadata_files = []
    application_counter = 0

    run_time = (
        read_runtime_environment(runtime_environment)
        if not args_dict["ignore_runtime_environment"]
        else []
    )

    with log_file.open("w", encoding="utf-8") as file:
        file.write("Running simtools applications\n")
        file.write(dependencies.get_version_string(run_time, include_software_versions=False))
        try:
            for config in configurations:
                app = config.get("application")
                if not config.get("run_application"):
                    logger.info(f"Skipping application: {app}")
                    continue

                application_counter += 1

                app_configuration = config.get("configuration", {})
                app_activity_id = app_configuration.get("activity_id") or gen.get_uuid()
                app_configuration["activity_id"] = app_activity_id
                app_configuration.setdefault("label", app)
                app_configuration["disable_log_file"] = True

                metadata_file = _get_model_parameter_metadata_file(app_configuration)
                if metadata_file is not None:
                    model_parameter_metadata_files.append(metadata_file)

                associated_activities.append({"activity_name": app, "activity_id": app_activity_id})

                logger.info(f"Running application: {app}")
                result = job_manager.submit(
                    app,
                    out_file=None,
                    err_file=None,
                    configuration=app_configuration,
                    runtime_environment=run_time,
                )
                file.write("=" * 80 + "\n")
                file.write(
                    f"Application: {app}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\n"
                )
        finally:
            if model_parameter_metadata_files:
                workflow_activity = workflow_metadata.build_workflow_activity_metadata(
                    args_dict=args_dict,
                    workflow_activity_id=workflow_activity_id,
                    workflow_start=workflow_start,
                    workflow_end=max(datetime.now(UTC), workflow_start),
                    runtime_environment=(
                        runtime_environment_snapshot
                        if not args_dict["ignore_runtime_environment"]
                        else None
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
    output_path, setting_workflow = _set_input_output_directories(configuration_file)
    configurations = job_configuration.get("applications")
    logger.info(f"Setting workflow output path to {output_path}")
    for step_count, config in enumerate(configurations, start=1):
        config["run_application"] = step_count in steps if steps else True
        config = gen.change_dict_keys_case(config, True)
        config["configuration"] = _replace_placeholders_in_configuration(
            config.get("configuration", {}),
            output_path,
            setting_workflow,
        )
        if config["configuration"].get("activity_id") is None:
            config["configuration"]["activity_id"] = gen.get_uuid()
        configurations[step_count - 1] = config

    return (
        configurations,
        job_configuration.get("runtime_environment"),
        output_path / "simtools.log",
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
        The first part is the 'input' directory, the second part is the subdirectory name
    """
    setting_workflow = gen.extract_subdirectories_from_path(path, anchor="input")
    output_path = Path("output") / Path(setting_workflow)
    output_path.mkdir(parents=True, exist_ok=True)
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
