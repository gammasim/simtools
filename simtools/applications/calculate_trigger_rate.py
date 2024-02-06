#!/usr/bin/python3

"""
Summary
-------
This application calculates the trigger rate from a histogram or a list of histograms.

Command line arguments
----------------------
histogram_files (str or list):
    Path to the histogram file or a list of histogram files.
livetime (float):
    Livetime used in the simulation that produced the histograms in seconds.

Example
-------
.. code-block:: console

    simtools-calculate-trigger-rate --hist_file_names tests/resources/run201_proton_za20deg_azm0deg
    _North_TestLayout_test-prod.simtel.zst --livetime 100
"""

import logging
from pathlib import Path

import astropy.units as u

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
        "--hist_file_names",
        help="Name of the histogram files to be calculate the trigger rate from  or the text file "
        "containing the list of histogram files.",
        nargs="+",
        required=True,
        type=str,
    )

    config.parser.add_argument(
        "--livetime",
        help="Livetime used in the simulation that produced the histograms in seconds.",
        type=float,
        required=True,
    )
    config_parser, _ = config.initialize(db_config=False, paths=True)

    return config_parser


def main():
    label = Path(__file__).stem
    description = "Calculates the event rate based on simtel array histograms."
    config_parser = _parse(label, description)

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(config_parser["log_level"]))
    logger.info("Starting the application.")

    histogram_files = config_parser["hist_file_names"]
    livetime = config_parser["livetime"] * u.s

    if isinstance(histogram_files, str):
        histogram_files = [histogram_files]

    histograms = SimtelHistograms(histogram_files)

    logger.info(f"Calculating event rate and trigger rate for livetime: {livetime}")

    # Calculate trigger rate
    obs_time = histograms.estimate_observation_time()
    trigger_rates = histograms.trigger_rate_per_histogram(re_weight=True)
    event_rates = histograms.total_num_simulated_events / obs_time

    # Print the trigger rates
    for i, trigger_rate in enumerate(trigger_rates):
        logger.info(f"Event rate for histogram {i + 1}: {event_rates.value:.4e} Hz")
        logger.info(f"Trigger rate for histogram {i + 1}: {trigger_rate.value:.4e} Hz")

    logger.info("Application completed.")


if __name__ == "__main__":
    main()
