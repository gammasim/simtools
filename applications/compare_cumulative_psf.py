#!/usr/bin/python3

import logging
import matplotlib.pyplot as plt
from copy import copy
from pathlib import Path
from math import sqrt
from collections import OrderedDict

import numpy as np
import astropy.units as u
from astropy.io import ascii
from astropy.table import Table

import simtools.io_handler as io
from simtools.ray_tracing import RayTracing
from simtools.model.telescope_model import TelescopeModel
from simtools import visualize

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def getData(**kwargs):
    dType = {
        'names': ('Radius [cm]', 'Relative intensity'),
        'formats': ('f8', 'f8')
    }
    testDataFile = io.getTestDataFile('PSFcurve_data_v2.txt')
    data = np.loadtxt(testDataFile, dtype=dType, usecols=(0, 2))
    data['Radius [cm]'] *= 0.1
    data['Relative intensity'] /= np.max(np.abs(data['Relative intensity']))
    return data


if __name__ == '__main__':
    sourceDistance = 12 * u.km
    version = 'prod4'
    label = 'lst_integral'
    zenithAngle = 20 * u.deg
    offAxisAngle = [0 * u.deg]

    tel = TelescopeModel(
        telescopeType='north-lst-1',
        version=version,
        label=label
    )

    # New parameters defined by Konrad
    tel.changeParameters(
        mirror_reflection_random_angle='0.0075,0.15,0.035',
        mirror_align_random_horizontal='0.0040,28.,0.0,0.0',
        mirror_align_random_vertical='0.0040,28.,0.0,0.0'
    )

    ray = RayTracing(
        telescopeModel=tel,
        sourceDistance=sourceDistance,
        zenithAngle=zenithAngle,
        offAxisAngle=offAxisAngle
    )

    ray.simulate(test=True, force=False)
    ray.analyze(force=False)

    # Plotting cumulative PSF
    im = ray.images()[0]

    print('d80 in cm = {}'.format(im.getPSF()))

    dataToPlot = OrderedDict()
    dataToPlot[r'sim$\_$telarray (src dist = 12 km)'] = im.getCumulativeData()
    dataToPlot['measured'] = getData()
    plt = visualize.plot1D(dataToPlot)

    simD80 = im.getPSF(0.8, 'cm') / 2
    plt.plot(
        [simD80, simD80],
        [0, 1.05],
        marker='None',
        color=visualize.getColors()[0],
        linestyle='--'
    )
    measD80 = 3.091 / 2
    plt.plot(
        [measD80, measD80],
        [0, 1.05],
        marker='None',
        color=visualize.getColors()[1],
        linestyle='--'
    )
    plt.gca().set_ylim(0, 1.05)

    plt.savefig('LST_CumulativePSF.pdf', format='pdf', bbox_inches='tight')
    plt.clf()

    dataToPlot = im.getImageData()
    visualize.plotHist2D(dataToPlot, bins=80)
    circle = plt.Circle((0, 0), im.getPSF(0.8) / 2, color='k', fill=False, lw=2, ls='--')
    plt.gca().add_artist(circle)

    plt.savefig('LST_photons.pdf', format='pdf', bbox_inches='tight')

    # plt.show()
