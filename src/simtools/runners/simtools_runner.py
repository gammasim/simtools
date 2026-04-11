"""Tools for running applications in the simtools framework."""

import shutil
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import simtools.utils.general as gen
from simtools import dependencies
from simtools.data_model.metadata_collector import MetadataCollector
from simtools.io import ascii_handler
from simtools.job_execution import job_manager


def run_applications(args_dict, logger):
    """
    Run simtools applications step-by-step as defined in a configuration file.

    Parameters
    ----------
    args_dict : dict
        Dictionary containing command line arguments.
    logger : logging.Logger
        Logger for logging application output.
    """
    (
        configurations,
        runtime_environment,
        log_file,
        workflow_activity_id,
    ) = _read_application_configuration(
        args_dict["config_file"],
        args_dict.get("steps"),
        logger,
        args_dict.get("activity_id"),
    )
    workflow_start = datetime.now(UTC)
    associated_activities = []
    runtime_environment_snapshot = deepcopy(runtime_environment)
    workflow_site = _get_workflow_configuration_value(configurations, "site")
    workflow_instrument = _get_workflow_configuration_value(configurations, "instrument")
    if workflow_instrument is None:
        workflow_instrument = _get_workflow_configuration_value(configurations, "telescope")
    model_parameter_metadata_files = []

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

                app_configuration = config.get("configuration", {})
                app_activity_id = app_configuration.get("activity_id") or gen.uuid()
                app_configuration["activity_id"] = app_activity_id
                associated_activities.append({"name": app, "activity_id": app_activity_id})
                metadata_file = _get_model_parameter_metadata_file(app, app_configuration)
                if metadata_file is not None:
                    model_parameter_metadata_files.append(metadata_file)

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
            workflow_end = datetime.now(UTC)
            workflow_end = max(workflow_end, workflow_start)
            workflow_metadata = _build_workflow_metadata(
                args_dict=args_dict,
                workflow_activity_id=workflow_activity_id,
                workflow_start=workflow_start,
                workflow_end=workflow_end,
                runtime_environment=runtime_environment_snapshot,
                workflow_site=workflow_site,
                workflow_instrument=workflow_instrument,
            )
            _update_model_parameter_metadata_files(
                model_parameter_metadata_files=model_parameter_metadata_files,
                workflow_metadata=workflow_metadata,
                associated_activities=associated_activities,
                logger=logger,
            )


def _read_application_configuration(configuration_file, steps, logger, workflow_activity_id=None):
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
    logger : Logger
        Logger object.

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
    configurations = job_configuration.get("applications")
    path_activity_id = gen.extract_uuid7_from_path(configuration_file)
    workflow_activity_id = (
        job_configuration.get("activity_id")
        or path_activity_id
        or workflow_activity_id
        or gen.uuid()
    )
    output_path, setting_workflow = _set_input_output_directories(configuration_file)
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
            config["configuration"]["activity_id"] = gen.uuid()
        configurations[step_count - 1] = config

    return (
        configurations,
        job_configuration.get("runtime_environment"),
        output_path / "simtools.log",
        workflow_activity_id,
    )


def _build_workflow_metadata(
    args_dict,
    workflow_activity_id,
    workflow_start,
    workflow_end,
    runtime_environment,
    workflow_site,
    workflow_instrument,
):
    """Build workflow-level metadata dictionary with authoritative lifecycle timestamps."""
    metadata_args = dict(args_dict)
    metadata_args["label"] = "setting_workflow"
    metadata_args["activity_id"] = workflow_activity_id
    metadata_args["activity_start"] = workflow_start.isoformat(timespec="seconds")
    metadata_args["activity_end"] = workflow_end.isoformat(timespec="seconds")
    metadata_args["runtime_environment"] = deepcopy(runtime_environment)
    metadata_args["site"] = workflow_site
    metadata_args["instrument"] = workflow_instrument

    collector = MetadataCollector(metadata_args, clean_meta=False)
    return collector.get_top_level_metadata().get("cta", {})


def _get_model_parameter_metadata_file(application, app_configuration):
    """Return expected metadata file for model-parameter submission applications."""
    if application != "simtools-submit-model-parameter-from-external":
        return None

    parameter = app_configuration.get("parameter")
    parameter_version = app_configuration.get("parameter_version")
    output_path = app_configuration.get("output_path")
    if not parameter or not parameter_version or not output_path:
        return None

    return Path(output_path) / parameter / f"{parameter}-{parameter_version}.meta.yml"


def _update_model_parameter_metadata_files(
    model_parameter_metadata_files,
    workflow_metadata,
    associated_activities,
    logger,
):
    """Inject workflow metadata into model-parameter metadata files."""
    workflow_activity = deepcopy(workflow_metadata.get("activity", {}))

    for metadata_file in model_parameter_metadata_files:
        metadata_path = Path(metadata_file)
        if not metadata_path.exists():
            logger.debug(f"Model-parameter metadata file does not exist: {metadata_path}")
            continue

        metadata = ascii_handler.collect_data_from_file(metadata_path)
        metadata = gen.change_dict_keys_case(metadata, True)
        cta_meta = metadata.get("cta", {})
        cta_meta["activity"] = deepcopy(workflow_activity)

        context = cta_meta.setdefault("context", {})
        context_associated = context.get("associated_activities") or []
        context["associated_activities"] = _merge_associated_activities(
            context_associated,
            associated_activities,
        )

        metadata["cta"] = cta_meta
        ascii_handler.write_data_to_file(metadata, metadata_path)
        logger.info(f"Updated workflow metadata in {metadata_path}")


def _merge_associated_activities(existing_activities, new_activities):
    """Merge associated activities preserving order and uniqueness."""
    merged_activities = []
    seen = set()
    for activity in [*existing_activities, *new_activities]:
        key = (activity.get("name"), activity.get("activity_id"))
        if key in seen:
            continue
        seen.add(key)
        merged_activities.append(activity)
    return merged_activities


def _get_workflow_configuration_value(configurations, key):
    """Return first non-empty configuration value for a given key."""
    for config in configurations:
        value = config.get("configuration", {}).get(key)
        if value is not None:
            return value
    return None


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
    for key, value in configuration.items():
        if isinstance(value, str):
            configuration[key] = value.replace(place_holder, setting_workflow)
        if isinstance(value, list):
            configuration[key] = [
                item.replace(place_holder, setting_workflow) if isinstance(item, str) else item
                for item in value
            ]
    if output_path:
        configuration["output_path"] = str(output_path)

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
    path = Path(path).resolve()
    try:
        input_index = path.parts.index("input")
        # Get all parts after 'input', excluding the filename
        subdirs = path.parts[input_index + 1 : -1]
        setting_workflow = "/".join(subdirs)
        workflow_dir = path.parts[input_index]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Could not find subdirectory under 'input': {exc}") from exc

    output_path = Path(str(workflow_dir).replace("input", "output")) / Path(setting_workflow)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path, "/".join(subdirs)


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
