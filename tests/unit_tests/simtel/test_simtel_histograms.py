#!/usr/bin/python3

import logging

import simtools.io_handler as io
from simtools.simtel.simtel_histograms import SimtelHistograms

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_histograms(cfg_setup):
    histogram_files = list()
    histogram_files.append(
        io.getDataFile(
            fileName="run1_gamma_za20deg_azm0deg-North-Prod5_test-production-5.hdata.zst", test=True
        )
    )
    histogram_files.append(
        io.getDataFile(
            fileName="run2_gamma_za20deg_azm0deg-North-Prod5_test-production-5.hdata.zst", test=True
        )
    )

    hists = SimtelHistograms(histogramFiles=histogram_files, test=True)

    figName = io.getTestPlotFile("simtelHistograms.pdf")
    hists.plotAndSaveFigures(figName=figName)

    assert io.getTestPlotFile("simtelHistograms.pdf").exists()
