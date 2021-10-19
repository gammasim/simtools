#!/usr/bin/python3

import logging
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from eventio import EventIOFile, Histograms
from eventio.search_utils import yield_toplevel_of_type

import simtools.io_handler as io
from simtools.simtel.simtel_histograms import SimtelHistograms

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_histograms():
    histogram_files = list()
    histogram_files.append(io.getTestDataFile(
        'run1_gamma_za20deg_azm0deg-North-Prod5_test-production-5.hdata.zst')
    )
    histogram_files.append(io.getTestDataFile(
        'run2_gamma_za20deg_azm0deg-North-Prod5_test-production-5.hdata.zst')
    )

    print(histogram_files)

    hists = SimtelHistograms(histogramFiles=histogram_files)

    figName = io.getTestPlotFile('simtelHistograms.pdf')
    hists.plotAndSaveFigures(figName=figName)


if __name__ == '__main__':
    test_histograms()
