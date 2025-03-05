#!/usr/bin/python3

"""
Run simtools applications from configuration files for setting workflows.

Allows to run several simtools applications with a single configuration file, which includes
both the name of the simtools application and the configuration for the application.

Strong assumption on the directory structure for input and output files of applications.

"""

import logging
import subprocess
import tempfile
from pathlib import Path

import yaml

import simtools.utils.general as gen
from simtools import dependencies
from simtools.configuration import configurator


def _parse(label, description, usage):
    """
    Parse command line configuration.

    Parameters
    ----------
    label : str
        Label describing the application.
    description : str
        Description of the application.
    usage : str
        Example on how to use the application.

    Returns
    -------
    CommandLineParser
        Command line parser object.
    """
    config = configurator.Configurator(label=label, description=description, usage=usage)

    config.parser.add_argument(
        "--configuration_file",
        help="Application configuration.",
        type=str,
        required=True,
        default=None,
    )
    return config.initialize(db_config=False)


def run_application(application, configuration, logger):
    """Run a simtools application and return stdout and stderr."""
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


def get_subdirectory_name(path):
    """Get the first subdirectory name under 'input'."""
    path = Path(path).resolve()
    try:
        input_index = path.parts.index("input")
        return path.parts[input_index], path.parts[input_index + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Could not find subdirectory under 'input': {exc}") from exc


def read_application_configuration(configuration_file, logger):
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
    logger : Logger
        Logger object.

    Returns
    -------
    dict
        Application configuration.
    Path
        Path to the log file.

    """
    application_config = gen.collect_data_from_file(configuration_file).get("CTA_SIMPIPE")
    place_holder = "__SETTING_WORKFLOW__"
    workflow_dir, setting_workflow = get_subdirectory_name(configuration_file)
    output_path = str(workflow_dir).replace("input", "output") + setting_workflow
    logger.info(f"Setting workflow output path to {output_path}")
    log_file = (
        Path(application_config.get("LOG_PATH", "./").replace(place_holder, setting_workflow))
        / "simtools.log"
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)
    configurations = application_config.get("APPLICATIONS")
    for config in configurations:
        for key, value in config.get("CONFIGURATION", {}).items():
            if isinstance(value, str):
                config["CONFIGURATION"][key] = value.replace(place_holder, setting_workflow)
            if isinstance(value, list):
                config["CONFIGURATION"][key] = [
                    item.replace(place_holder, setting_workflow) for item in value
                ]
        config["CONFIGURATION"]["USE_PLAIN_OUTPUT_PATH"] = True
        config["OUTPUT_PATH"] = output_path

    return configurations, log_file


def main():  # noqa: D103
    args_dict, _ = _parse(
        Path(__file__).stem,
        description="Run simtools applications from configuration file.",
        usage="simtools-run-application --config_file config_file_name",
    )
    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    configurations, log_file = read_application_configuration(
        args_dict["configuration_file"], logger
    )

    with log_file.open("w", encoding="utf-8") as file:
        file.write("Running simtools applications\n")
        file.write(dependencies.get_version_string())
        for config in configurations:
            logger.info(f"Running application: {config.get('APPLICATION')}")
            config = gen.change_dict_keys_case(config, False)
            stdout, stderr = run_application(
                config.get("APPLICATION"), config.get("CONFIGURATION"), logger
            )
            file.write("=" * 80 + "\n")
            file.write(f"Application: {config.get('APPLICATION')}\n")
            file.write("STDOUT:\n" + stdout)
            file.write("STDERR:\n" + stderr)


if __name__ == "__main__":
    main()
