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
from pathlib import Path

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.runners import simtools_runner


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
    config.parser.add_argument(
        "--ignore_runtime_environment",
        action="store_true",
        help="Ignore the runtime environment and run the application in the current environment.",
        default=False,
    )
    return config.initialize(db_config=True)


def main():  # noqa: D103
    args_dict, db_config = _parse(
        Path(__file__).stem,
        description="Run simtools applications using a configuration file.",
        usage="simtools-run-application --config_file config_file_name",
    )
    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    simtools_runner.run_applications(args_dict, db_config, logger)


if __name__ == "__main__":
    main()
