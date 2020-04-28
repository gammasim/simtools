#!/usr/bin/python3

import logging
import os
import sys
from pathlib import Path
import numpy as np
from astropy import units as u
from astropy.io import ascii
from simtools import visualize

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def test_plot_1D():

    logger.debug('Testing plot1D')

    xTitle = 'Wavelength [nm]'
    yTitle = 'Mirror reflectivity [%]'
    headersType = {'names': (xTitle, yTitle),
                   'formats': ('f8', 'f8')}
    title = 'Test 1D plot'

    dataIn = np.loadtxt('../data/test-data/ref_200_1100_190211a.dat',
                        usecols=(0, 1), dtype=headersType)

    # Change y-axis to percent
    if '%' in yTitle:
        if np.max(dataIn[yTitle]) <= 1:
            dataIn[yTitle] = 100*dataIn[yTitle]
    data = dict()
    data['Reflectivity'] = dataIn
    for i in range(5):
        newData = np.copy(dataIn)
        newData[yTitle] = newData[yTitle]*(1 - 0.1*(i + 1))
        data['{}%% reflectivity'.format(100*(1 - 0.1*(i + 1)))] = newData

    plt = visualize.plot1D(data, title=title, palette='autumn')

    plotFile = 'tests/test-plots/plot_1D.pdf'
    if Path(plotFile).exists():
        os.remove(plotFile)
    plt.savefig(plotFile)
    if not Path(plotFile).exists():
        raise RuntimeError('Did not create {}!'.format(plotFile))

    logger.debug('Produced 1D plot ({}).'.format(plotFile))

    return


def test_plot_table():

    logger.debug('Testing plotTable')

    title = 'Test plot table'

    table = ascii.read('../data/test-data/Transmission_Spectrum_PlexiGlass.dat')

    plt = visualize.plotTable(table, yTitle='Transmission', title=title, noMarkers=True)

    plotFile = 'tests/test-plots/plot_table.pdf'
    if Path(plotFile).exists():
        os.remove(plotFile)
    plt.savefig(plotFile)
    if not Path(plotFile).exists():
        raise RuntimeError('Did not create {}!'.format(plotFile))

    logger.debug('Produced 1D plot ({}).'.format(plotFile))

    return


def test_add_unit():

    valueWithUnit = [30, 40] << u.nm
    assert(visualize._addUnit('Wavelength', valueWithUnit) == 'Wavelength [nm]')
    valueWithoutUnit = [30, 40]
    assert(visualize._addUnit('Wavelength', valueWithoutUnit) == 'Wavelength')

    return


if __name__ == '__main__':

    test_plot_1D()
