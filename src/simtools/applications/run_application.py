#!/usr/bin/python3

"""
Run several simtools applications using a configuration file.

Allows to run several simtools applications with a single configuration file, which includes
both the name of the simtools application and the configuration for the application.

This application is used for model parameter setting workflows.
Strong assumptions are applied on the directory structure for input and output files of
applications.

Example
-------

Run the application with the configuration file 'config_file_name':

.. code-block:: console

    simtools-run-application --configuration_file config_file_name

Run the application with the configuration file 'config_file_name', but skipping all steps except
step 2 and 3 (useful for debugging):

.. code-block:: console

    simtools-run-application --configuration_file config_file_name --steps 2 3

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
    config.parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        help="List of steps to be execution (e.g., '--steps 7 8 9'; do not specify to run all).",
    )
    return config.initialize(db_config=True)


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


def read_application_configuration(configuration_file, steps, logger):
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
    application_config = gen.collect_data_from_file(configuration_file)
    place_holder = "__SETTING_WORKFLOW__"
    workflow_dir, setting_workflow = get_subdirectory_name(configuration_file)
    output_path = Path(str(workflow_dir).replace("input", "output")) / Path(setting_workflow)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Setting workflow output path to {output_path}")
    log_file = output_path / "simtools.log"
    configurations = application_config.get("applications")
    for step_count, config in enumerate(configurations, start=1):
        config["run_application"] = step_count in steps if steps else True
        for key, value in config.get("configuration", {}).items():
            if isinstance(value, str):
                config["configuration"][key] = value.replace(place_holder, setting_workflow)
            if isinstance(value, list):
                config["configuration"][key] = [
                    item.replace(place_holder, setting_workflow) if isinstance(item, str) else item
                    for item in value
                ]
        config["configuration"]["USE_PLAIN_OUTPUT_PATH"] = True
        config["configuration"]["OUTPUT_PATH"] = str(output_path)

    return configurations, log_file


def main():  # noqa: D103
    args_dict, db_config = _parse(
        Path(__file__).stem,
        description="Run simtools applications using a configuration file.",
        usage="simtools-run-application --config_file config_file_name",
    )
    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    configurations, log_file = read_application_configuration(
        args_dict["configuration_file"], args_dict["steps"], logger
    )

    with log_file.open("w", encoding="utf-8") as file:
        file.write("Running simtools applications\n")
        file.write(dependencies.get_version_string(db_config))
        for config in configurations:
            if config.get("run_application"):
                logger.info(f"Running application: {config.get('application')}")
            else:
                logger.info(f"Skipping application: {config.get('application')}")
                continue
            config = gen.change_dict_keys_case(config, True)
            stdout, stderr = run_application(
                config.get("application"), config.get("configuration"), logger
            )
            file.write("=" * 80 + "\n")
            file.write(f"Application: {config.get('application')}\n")
            file.write("STDOUT:\n" + stdout)
            file.write("STDERR:\n" + stderr)


if __name__ == "__main__":
    main()
