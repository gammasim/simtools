#!/usr/bin/python3

"""
Summary
-------
This application calculates the trigger rate from a histogram or a list of histograms.

Command line arguments
----------------------
histogram_files (str or list):
    Path to the histogram file or a list of histogram files.

Example
-------
.. code-block:: console

    simtools-calculate-trigger-rate --hist_file_names tests/resources/run201_proton_za20deg_azm0deg
    _North_TestLayout_test-prod.simtel.zst --livetime 100
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
        "--hist_file_names",
        help="Name of the histogram files to be calculate the trigger rate from  or the text file "
        "containing the list of histogram files.",
        nargs="+",
        required=True,
        type=str,
    )

    config_parser, _ = config.initialize(db_config=False, paths=True)

    return config_parser


def main():
    label = Path(__file__).stem
    description = (
        "Calculates the simulated and triggered event rate based on simtel array histograms."
    )
    config_parser = _parse(label, description)

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(config_parser["log_level"]))
    logger.info("Starting the application.")

    histogram_files = config_parser["hist_file_names"]

    if isinstance(histogram_files, str):
        histogram_files = [histogram_files]

    histograms = SimtelHistograms(histogram_files)

    logger.info("Calculating simulated and triggered event rate")

    for i in range(histograms.number_of_files):
        histogram_instance = histograms.list_of_hist_instances[i]
        obs_time = histogram_instance.estimate_observation_time()
        event_rate = histogram_instance.total_num_simulated_events / obs_time
        trigger_rate = histogram_instance.trigger_rate_per_histogram(re_weight=True)
        logger.info(f"Histogram {i + 1}:")
        logger.info(
            f"Total number of simulated events: {histogram_instance.total_num_simulated_events} "
            "events"
        )
        logger.info(
            f"Total number of triggered events: {histogram_instance.total_num_triggered_events} "
            "events"
        )
        logger.info(f"Estimated equivalent observation time: {obs_time.value} s")
        logger.info(f"Simulated event rate: {event_rate.value:.4e} Hz")
        logger.info(f"System trigger event rate: {trigger_rate.value:.4e} Hz")


if __name__ == "__main__":
    main()
