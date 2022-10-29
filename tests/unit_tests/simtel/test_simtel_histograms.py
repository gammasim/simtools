#!/usr/bin/python3

import logging

from simtools.simtel.simtel_histograms import SimtelHistograms

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_histograms(io_handler):
    histogram_files = list()
    histogram_files.append(
        io_handler.getInputDataFile(
            fileName="run1_gamma_za20deg_azm0deg-North-Prod5_test-production-5.hdata.zst", test=True
        )
    )
    histogram_files.append(
        io_handler.getInputDataFile(
            fileName="run2_gamma_za20deg_azm0deg-North-Prod5_test-production-5.hdata.zst", test=True
        )
    )

    hists = SimtelHistograms(histogramFiles=histogram_files, test=True)

    figName = io_handler.getOutputFile(fileName="simtelHistograms.pdf", dirType="plots", test=True)
    hists.plotAndSaveFigures(figName=figName)

    assert figName.exists()
