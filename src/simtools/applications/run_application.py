#!/usr/bin/python3

"""
Run simtools applications from configuration files.

Allows to run several simtools applications with a single configuration file, which includes
both the name of the simtools application and the configuration for the application.

"""

import logging
from pathlib import Path

import simtools.utils.general as gen
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


def main():  # noqa: D103

    args_dict, _ = _parse(
        Path(__file__).stem,
        description="Run simtools applications from configuration file.",
        usage="simtools-run-application --config_file config_file_name",
    )
    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))


if __name__ == "__main__":
    main()
