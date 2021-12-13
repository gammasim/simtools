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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import simtools.util.general as gen
from simtools.simtel.simtel_histograms import SimtelHistograms


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=(
            'Simulate showers to be used for trigger rate calculations'
        )
    )
    parser.add_argument(
        '-l',
        '--file_lists',
        help='File containing the list of histogram files to be plotted',
        nargs='+',
        type=str,
        required=True
    )
    parser.add_argument(
        '-o',
        '--output',
        help='File name for the pdf output (without extension)',
        type=str,
        required=True
    )
    parser.add_argument(
        '-v',
        '--verbosity',
        dest='logLevel',
        action='store',
        default='info',
        help='Log level to print (default is INFO)'
    )

    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(gen.getLogLevelFromUser(args.logLevel))

    nLists = len(args.file_lists)
    simtelHistograms = list()
    for thisListOfFiles in args.file_lists:
        # Collecting hist files
        histogramFiles = list()
        with open(thisListOfFiles) as file:
            for line in file:
                histogramFiles.append(line.replace('\n', ''))

        # Building SimtelHistograms
        sh = SimtelHistograms(histogramFiles)
        simtelHistograms.append(sh)


    # Checking if number of histograms is consistent
    numberOfHists = [sh.numberOfHistograms for sh in simtelHistograms]
    # Converting list to set will remove the duplicated entries.
    # If all entries in the list are the same, len(set) will be 1
    if len(set(numberOfHists)) > 1:
        msg = (
            'Number of histograms in different sets of simulations are inconsistent'
            ' - please make sure the simulations sets are consistent'
        )
        logger.error(msg)
        raise IOError(msg)

    # Plotting
    pdfPages = PdfPages(args.output + '.pdf')
    for iHist in range(numberOfHists[0]):

        title = simtelHistograms[0].combinedHists[iHist]['title']

        logger.debug('Processing: {}'.format(title))

        fig, axs = plt.subplots(1, nLists, figsize=(6 * nLists, 6))

        for sh, ax in zip(simtelHistograms, axs):
            sh.plotOneHistogram(iHist, ax)

        plt.tight_layout()
        pdfPages.savefig(fig)
        plt.clf()

    plt.close()
    pdfPages.close()
