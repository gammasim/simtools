#!/usr/bin/python3

"""
Summary
-------
This application calculates the trigger rate from a simtel_array output file or a list of
simtel_array output files.

Command line arguments
----------------------
simtel_array_files (str or list):
    Path to the simtel_array file or a list of simtel_array output files.

Example
-------
.. code-block:: console

    simtools-calculate-trigger-rate --simtel_file_names tests/resources/
    run201_proton_za20deg_azm0deg_North_TestLayout_test-prod.simtel.zst --livetime 100
"""

import logging
from pathlib import Path

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.simtel.simtel_histograms import SimtelHistograms


def _parse(label, description):
    """
    Parse command line configuration

    Parameters
    ----------
    label: str
        Label describing the application.
    description: str
        Description of the application.

    Returns
    -------
    CommandLineParser
        Command line parser object

    """
    config = configurator.Configurator(label=label, description=description)

    config.parser.add_argument(
        "--simtel_file_names",
        help="Name of the simtel_array output files to be calculate the trigger rate from  or the "
        "text file containing the list of simtel_array output files.",
        nargs="+",
        required=True,
        type=str,
    )

    config_parser, _ = config.initialize(db_config=False, paths=True)

    return config_parser


def main():
    label = Path(__file__).stem
    description = (
        "Calculates the simulated and triggered event rate based on simtel_array output files."
    )
    config_parser = _parse(label, description)

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(config_parser["log_level"]))
    logger.info("Starting the application.")

    simtel_array_files = config_parser["simtel_file_names"]

    if isinstance(simtel_array_files, str):
        simtel_array_files = [simtel_array_files]

    histograms = SimtelHistograms(simtel_array_files)

    logger.info("Calculating simulated and triggered event rate")
    histograms.calculate_event_rates()


if __name__ == "__main__":
    main()
