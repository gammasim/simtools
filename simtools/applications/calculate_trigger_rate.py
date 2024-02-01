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

    calculate-trigger-rate --histogram_files /path/to/histogram.h5 --livetime 3600
"""

import argparse
import logging

import astropy.units as u

from simtools.simtel.simtel_histograms import SimtelHistograms
from simtools.utils.general import get_log_level_from_user


def _parse():
    """
    Parse command line configuration.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments.

    """
    parser = argparse.ArgumentParser(
        description="Calculate the trigger rate from a histogram or a list of histograms."
    )

    parser.add_argument(
        "--histogram_files",
        help="Path to the histogram file or a list of histogram files.",
        required=True,
    )

    parser.add_argument(
        "--livetime",
        help="Livetime used in the simulation that produced the histograms in seconds.",
        type=float,
        required=True,
    )

    return parser.parse_args()


def main():
    args = _parse()
    logger = logging.getLogger()
    logger.setLevel(get_log_level_from_user(logging.INFO))

    histogram_files = args.histogram_files
    livetime = args.livetime * u.s

    logger.info("Starting the application.")

    # Create an instance of SimtelHistograms
    if isinstance(histogram_files, str):
        histogram_files = [histogram_files]

    histograms = SimtelHistograms(histogram_files)

    logger.info(f"Calculating trigger rate for livetime: {livetime}")

    # Calculate trigger rate
    trigger_rates = histograms.trigger_rate_per_histogram(livetime)

    # Print or save the trigger rates
    for i, trigger_rate in enumerate(trigger_rates):
        logger.info(f"Trigger rate for histogram {i + 1}: {trigger_rate:.4e} Hz")

    logger.info("Application completed.")


if __name__ == "__main__":
    main()
