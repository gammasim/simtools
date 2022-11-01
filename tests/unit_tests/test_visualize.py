#!/usr/bin/python3

import logging

import astropy.io.ascii
import astropy.units as u
import numpy as np

import simtools.util.general as gen
from simtools import visualize

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def test_plot_1D(db, io_handler):

    logger.debug("Testing plot_1D")

    xTitle = "Wavelength [nm]"
    yTitle = "Mirror reflectivity [%]"
    headersType = {"names": (xTitle, yTitle), "formats": ("f8", "f8")}
    title = "Test 1D plot"

    testFileName = "ref_200_1100_190211a.dat"
    db.export_file_db(
        dbName=db.DB_CTA_SIMULATION_MODEL,
        dest=io_handler.get_output_directory(dirType="model", test=True),
        fileName=testFileName,
    )
    testDataFile = gen.find_file(
        testFileName, io_handler.get_output_directory(dirType="model", test=True)
    )
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

    plt = visualize.plot_1D(data, title=title, palette="autumn")

    plotFile = io_handler.get_output_file(fileName="plot_1D.pdf", dirType="plots", test=True)
    if plotFile.exists():
        plotFile.unlink()
    plt.savefig(plotFile)

    logger.debug("Produced 1D plot ({}).".format(plotFile))

    assert plotFile.exists()


def test_plot_table(db, io_handler):

    logger.debug("Testing plot_table")

    title = "Test plot table"

    testFileName = "Transmission_Spectrum_PlexiGlass.dat"
    db.export_file_db(
        dbName="test-data",
        dest=io_handler.get_output_directory(dirType="model", test=True),
        fileName=testFileName,
    )
    tableFile = gen.find_file(
        testFileName, io_handler.get_output_directory(dirType="model", test=True)
    )
    table = astropy.io.ascii.read(tableFile)

    plt = visualize.plot_table(table, yTitle="Transmission", title=title, noMarkers=True)

    plotFile = io_handler.get_output_file(fileName="plot_table.pdf", dirType="plots", test=True)
    if plotFile.exists():
        plotFile.unlink()
    plt.savefig(plotFile)

    logger.debug("Produced 1D plot ({}).".format(plotFile))

    assert plotFile.exists()


def test_add_unit():

    valueWithUnit = [30, 40] << u.nm
    assert visualize._add_unit("Wavelength", valueWithUnit) == "Wavelength [nm]"
    valueWithoutUnit = [30, 40]
    assert visualize._add_unit("Wavelength", valueWithoutUnit) == "Wavelength"
