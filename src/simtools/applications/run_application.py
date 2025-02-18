#!/usr/bin/python3

"""
Run simtools applications from configuration files.

Allows to run several simtools applications with a single configuration file, which includes
both the name of the simtools application and the configuration for the application.

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


def run_application(application, configuration):
    """Run a simtools application and return stdout and stderr."""
    with tempfile.NamedTemporaryFile(mode="w", delete=True, suffix=".yml") as temp_config:
        yaml.dump(configuration, temp_config, default_flow_style=False)
        temp_config.flush()
        configuration_file = Path(temp_config.name)
        result = subprocess.run(
            [application, "--config", configuration_file],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout, result.stderr


def main():  # noqa: D103

    args_dict, _ = _parse(
        Path(__file__).stem,
        description="Run simtools applications from configuration file.",
        usage="simtools-run-application --config_file config_file_name",
    )
    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    application_config = gen.collect_data_from_file(args_dict["configuration_file"]).get(
        "CTA_SIMPIPE"
    )
    log_file = Path(application_config.get("LOG_PATH", "./")) / "simtools.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    configurations = application_config.get("APPLICATIONS")
    with log_file.open("w", encoding="utf-8") as file:
        file.write("Running simtools applications\n")
        file.write(dependencies.get_version_string())
        for config in configurations:
            logger.info(f"Running application: {config.get('APPLICATION')}")
            config = gen.change_dict_keys_case(config, False)
            stdout, stderr = run_application(config.get("APPLICATION"), config.get("CONFIGURATION"))
            file.write("=" * 80 + "\n")
            file.write(f"Application: {config.get('APPLICATION')}\n")
            file.write("STDOUT:\n" + stdout)
            file.write("STDERR:\n" + stderr)


if __name__ == "__main__":
    main()
