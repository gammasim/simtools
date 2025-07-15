"""Tools for running applications in the simtools framework."""

import subprocess
import tempfile
from pathlib import Path

import yaml

import simtools.utils.general as gen
from simtools import dependencies


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
    configurations, log_file = _read_application_configuration(
        args_dict["configuration_file"], args_dict.get("steps"), logger
    )

    with log_file.open("w", encoding="utf-8") as file:
        file.write("Running simtools applications\n")
        file.write(dependencies.get_version_string(db_config))

        for config in configurations:
            app = config.get("application")
            if not config.get("run_application"):
                logger.info(f"Skipping application: {app}")
                continue
            logger.info(f"Running application: {app}")
            stdout, stderr = run_application(app, config.get("configuration"), logger)
            file.write("=" * 80 + "\n")
            file.write(f"Application: {app}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}\n")


def run_application(application, configuration, logger):
    """
    Run a simtools application and return stdout and stderr.

    Parameters
    ----------
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
    with tempfile.NamedTemporaryFile(mode="w", delete=True, suffix=".yml") as temp_config:
        yaml.dump(configuration, temp_config, default_flow_style=False)
        temp_config.flush()
        configuration_file = Path(temp_config.name)
        try:
            result = subprocess.run(
                [application, "--config", configuration_file],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            logger.error(f"Error running application {application}: {exc.stderr}")
            raise exc
        return result.stdout, result.stderr


def _read_application_configuration(configuration_file, steps, logger):
    """
    Read application configuration from file and modify for setting workflows.

    Strong assumptions on the structure of input and output files:

    - configuration file is expected to be in './input/<workflow directory>/<yaml file>'
    - output files will be written out to './output/<workflow directory>/'

    Replaces the placeholders in the configuration file with the actual values.
    Sets 'USE_PLAIN_OUTPUT_PATH' to True for all applications.

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
    Path
        Path to the log file.

    """
    configurations = gen.collect_data_from_file(configuration_file).get("applications")
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

    return configurations, output_path / "simtools.log"


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
        configuration["use_plain_output_path"] = True
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
