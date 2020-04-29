#!/usr/bin/python3

import logging
import matplotlib.pyplot as plt
import numpy as np
from copy import copy
from pathlib import Path
from astropy.io import ascii
from astropy.table import Table
import astropy.units as u
from math import sqrt
from collections import OrderedDict

from simtools.util import config as cfg
from simtools.ray_tracing import RayTracing
from simtools.telescope_model import TelescopeModel
from simtools import visualize

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

config = cfg.loadConfig()

# plt.rc('font', family='serif', size=20)
# plt.rc('xtick', labelsize=20)
# plt.rc('ytick', labelsize=20)
# plt.rc('text', usetex=True)


def plotData(**kwargs):
    data = np.loadtxt('PSFcurve_data_v2.txt')
    totalIntensity = data[-1][2]
    intensity = list()
    radius = list()
    for row in data:
        radius.append(row[0] * 0.1)  # mm to cm
        intensity.append(row[2] / totalIntensity)
    ax = plt.gca()
    ax.plot(radius, intensity, **kwargs)


def getData(**kwargs):
    dType = {'names': ('Radius [cm]', 'Relative intensity'),
             'formats': ('f8', 'f8')}
    data = np.loadtxt('PSFcurve_data_v2.txt', dtype=dType, usecols=(0, 2))
    data['Radius [cm]'] *= 0.1
    data['Relative intensity'] /= np.max(np.abs(data['Relative intensity']))
    return data


if __name__ == '__main__':
    sourceDistance = 12 * u.km
    site = 'south'
    version = 'prod4'
    label = 'lst_integral'
    zenithAngle = 20 * u.deg
    offAxisAngle = [0 * u.deg]

    tel = TelescopeModel(
        telescopeType='lst',
        site=site,
        version=version,
        label=label
    )

    print('RNDA')
    print(tel.getParameter('mirror_reflection_random_angle'))
    print(tel.getParameter('mirror_align_random_vertical'))
    print(tel.getParameter('mirror_align_random_horizontal'))

    # tel.changeParameters(
    #     mirror_reflection_random_angle='0.0125',
    #     mirror_align_random_horizontal='0.0040,28.,0.0,0.0',
    #     mirror_align_random_vertical='0.0040,28.,0.0,0.0'
    # )

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

    # Plotting PSF images
    allImages = ray.images()
    im = allImages[0]

    print('d80 in cm')
    print(im.getPSF())

    # plt.figure(figsize=(8, 6), tight_layout=True)
    # ax = plt.gca()
    # ax.set_xlabel('radius [cm]')
    # ax.set_ylabel('relative intensity')

    # psf_* for PSF circle
    # image_* for histogram
    dataToPlot = OrderedDict()
    dataToPlot[r'sim$\_$telarray (src dist = 12 km)'] = im.getIntegral()
    dataToPlot['measured'] = getData()
    plt = visualize.plot1D(dataToPlot)

    simD80 = im.getPSF(0.8, 'cm') / 2
    plt.plot([simD80, simD80], [0, 1.05], 
             marker='None', color=visualize.getColors()[0], linestyle='--')
    measD80 = 3.091 / 2
    plt.plot([measD80, measD80], [0, 1.05], 
             marker='None', color=visualize.getColors()[1], linestyle='--')
    plt.gca().set_ylim(0, 1.05)

    # im.plotIntegral(color='r', linestyle=':', label=r'sim$\_$telarray (src dist = 12 km)')
    # plotData(color='b', marker='^', linestyle='--', label='measured')

    # simD80 = im.getPSF(0.8, 'cm') / 2
    # ax.plot([simD80, simD80], [0, 1.05], color='r')
    # measD80 = 3.091 / 2
    # ax.plot([measD80, measD80], [0, 1.05], color='b')
    # ax.set_ylim(0, 1.05)
    # ax.legend(frameon=False)

    plt.savefig('LST_CumulativePSF.pdf', format='pdf', bbox_inches='tight')

    plt.figure(figsize=(8, 6), tight_layout=True)
    ax = plt.gca()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # psf_* for PSF circle
    # image_* for histogram
    im.plotImage(psf_color='b')

    ax.set_aspect('equal', adjustable='datalim')

    # plt.show()
