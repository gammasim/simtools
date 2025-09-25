"""Tools for running applications in the simtools framework."""

import shutil
import subprocess
from pathlib import Path

import simtools.utils.general as gen
from simtools import dependencies
from simtools.io import ascii_handler


def run_applications(args_dict, db_config, logger):
    """
    Run simtools applications step-by-step as defined in a configuration file.

    Parameters
    ----------
    args_dict : dict
        Dictionary containing command line arguments.
    db_config : dict
        Database configuration
    logger : logging.Logger
        Logger for logging application output.
    """
    configurations, runtime_environment, log_file = _read_application_configuration(
        args_dict["configuration_file"], args_dict.get("steps"), logger
    )
    run_time = (
        read_runtime_environment(runtime_environment)
        if not args_dict["ignore_runtime_environment"]
        else []
    )

    with log_file.open("w", encoding="utf-8") as file:
        file.write("Running simtools applications\n")
        file.write(dependencies.get_version_string(db_config, run_time))

        for config in configurations:
            app = config.get("application")
            if not config.get("run_application"):
                logger.info(f"Skipping application: {app}")
                continue
            logger.info(f"Running application: {app}")
            stdout, stderr = run_application(run_time, app, config.get("configuration"), logger)
            file.write("=" * 80 + "\n")
            file.write(f"Application: {app}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}\n")


def run_application(runtime_environment, application, configuration, logger):
    """
    Run a simtools application and return stdout and stderr.

    Allow to specify a runtime environment (e.g., Docker) and a working directory.

    Parameters
    ----------
    runtime_environment : list
        Command to run the application in the specified runtime environment.
    application : str
        Name of the application to run.
    configuration : dict
        Configuration for the application.
    logger : logging.Logger
        Logger for logging application output.

    Returns
    -------
    tuple
        stdout and stderr from the application run.

    """
    command = [application, *_convert_dict_to_args(configuration)]
    if runtime_environment:
        command = runtime_environment + command
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        logger.error(f"Error running application {application}: {exc.stderr}")
        raise exc

    return result.stdout, result.stderr


def _convert_dict_to_args(parameters):
    """
    Convert a dictionary of parameters to a list of command line arguments.

    Parameters
    ----------
    parameters : dict
        Dictionary containing parameters to convert.

    Returns
    -------
    list
        List of command line arguments.
    """
    args = []
    for key, value in parameters.items():
        if isinstance(value, bool):
            if value:
                args.append(f"--{key}")
        elif isinstance(value, list):
            args.extend([f"--{key}", *(str(item) for item in value)])
        else:
            args.extend([f"--{key}", str(value)])
    return args


def _read_application_configuration(configuration_file, steps, logger):
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

    """
    job_configuration = ascii_handler.collect_data_from_file(configuration_file)
    configurations = job_configuration.get("applications")
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
        configurations[step_count - 1] = config

    return (
        configurations,
        job_configuration.get("runtime_environment"),
        output_path / "simtools.log",
    )


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
    return cmd
