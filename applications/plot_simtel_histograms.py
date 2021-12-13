#!/usr/bin/python3

'''
    Summary
    -------
    This application perform array simulations.

    The simulations are split into two stages: showers and array.
    Shower simulations are performed with CORSIKA and array simulations \
    with sim_telarray.

    A configuration file is required. See data/test-data/prodConfigTest.yml \
    for an example.

    Command line arguments
    ----------------------
    config (str, required)
        Path to the configuration file.
    primary (str)
        Name of the primary to be selected from the configuration file. In case it \
        is not given, all the primaries listed in the configuration file will be simulated.
    task (str)
        What task to execute. Options:
            simulate (perform simulations),
            lists (print list of output files) [NOT IMPLEMENTED]
            inspect (plot sim_telarray histograms for quick inspection) [NOT IMPLEMENTED]
    array_only (activation mode)
        Simulates only array detector (no showers).
    showers_only (activation mode)
        Simulates only showers (no array detector).
    test (activation mode, optional)
        If activated, no job will be submitted, but all configuration files \
        and run scripts will be created.
    verbosity (str, optional)
        Log level to print (default=INFO).

    Example
    -------
    Testing a mini-prod5 simulation.

    .. code-block:: console

        python applications/production.py -c data/test-data/prodConfigTest.yml --test
'''

import logging
import argparse

from simtools.simtel.simtel_histograms import SimtelHistograms


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=(
            'Simulate showers to be used for trigger rate calculations'
        )
    )
    parser.add_argument(
        '-l',
        '--file_list',
        help='File containing the list of histogram files to be plotted',
        type=str,
        required=True
    )

    parser.add_argument(
        '-o',
        '--output',
        help='File name for the pdf output (without extension)',
        type=str,
        required=False
    )

    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(gen.getLogLevelFromUser(args.logLevel))

    simtelHistograms = SimtelHistograms(args.file_list)
    if args.output is not None:
        simtelHistograms.plotAndSaveFigures(args.output)
    else:
        simtelHistograms.plotFigures(args.output)
