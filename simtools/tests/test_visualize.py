#!/usr/bin/python3

import logging

import numpy as np
import astropy.units as u
from astropy.io import ascii

from simtools import visualize
import simtools.io_handler as io

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def test_plot_1D():

    logger.debug("Testing plot1D")

    xTitle = "Wavelength [nm]"
    yTitle = "Mirror reflectivity [%]"
    headersType = {"names": (xTitle, yTitle), "formats": ("f8", "f8")}
    title = "Test 1D plot"

    testDataFile = io.getTestDataFile("ref_200_1100_190211a.dat")
    dataIn = np.loadtxt(testDataFile, usecols=(0, 1), dtype=headersType)

    # Change y-axis to percent
    if "%" in yTitle:
        if np.max(dataIn[yTitle]) <= 1:
            dataIn[yTitle] = 100 * dataIn[yTitle]
    data = dict()
    data["Reflectivity"] = dataIn
    for i in range(5):
        newData = np.copy(dataIn)
        newData[yTitle] = newData[yTitle] * (1 - 0.1 * (i + 1))
        data["{}%% reflectivity".format(100 * (1 - 0.1 * (i + 1)))] = newData

    plt = visualize.plot1D(data, title=title, palette="autumn")

    plotFile = io.getTestPlotFile("plot_1D.pdf")
    if plotFile.exists():
        plotFile.unlink()
    plt.savefig(plotFile)
    if not plotFile.exists():
        raise RuntimeError("Did not create {}!".format(plotFile))

    logger.debug("Produced 1D plot ({}).".format(plotFile))

    return


def test_plot_table():

    logger.debug("Testing plotTable")

    title = "Test plot table"

    tableFile = io.getTestDataFile("Transmission_Spectrum_PlexiGlass.dat")
    table = ascii.read(tableFile)

    plt = visualize.plotTable(table, yTitle="Transmission", title=title, noMarkers=True)

    plotFile = io.getTestPlotFile("plot_table.pdf")
    if plotFile.exists():
        plotFile.unlink()
    plt.savefig(plotFile)
    if not plotFile.exists():
        raise RuntimeError("Did not create {}!".format(plotFile))

    logger.debug("Produced 1D plot ({}).".format(plotFile))

    return


def test_add_unit():

    valueWithUnit = [30, 40] << u.nm
    assert visualize._addUnit("Wavelength", valueWithUnit) == "Wavelength [nm]"
    valueWithoutUnit = [30, 40]
    assert visualize._addUnit("Wavelength", valueWithoutUnit) == "Wavelength"

    return


if __name__ == "__main__":

    test_plot_1D()
    test_plot_table()
