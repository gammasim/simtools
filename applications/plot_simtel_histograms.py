#!/usr/bin/python3

'''
    Summary
    -------
    This application prints sim_telarray histograms into pdf file.
    It accepts multiple lists of histograms files that are plot together,
    side-by-side. Each histogram is plotted in a page of the pdf.


    Command line arguments
    ----------------------
    file_lists (str, required)
        Text file containing the list of sim_telarray histogram files to be plotted. \
        Multiple text files can be given.
    output (str, required)
        File name for the pdf output (without extension).
    verbosity (str, optional)
        Log level to print (default=INFO).

    Example
    -------
    .. code-block:: console

        python applications/plot_simtel_histograms.py -l list_test1.txt list_test2.txt -o histograms_comparison
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
            'Plots sim_telarray histograms.'
        )
    )
    parser.add_argument(
        '-l',
        '--file_lists',
        help='File containing the list of histogram files to be plotted.',
        nargs='+',
        type=str,
        required=True
    )
    parser.add_argument(
        '-o',
        '--output',
        help='File name for the pdf output (without extension).',
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
