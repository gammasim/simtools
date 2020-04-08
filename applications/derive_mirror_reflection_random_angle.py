#!/usr/bin/python3

import logging
import matplotlib.pyplot as plt
import numpy as np
from copy import copy
from pathlib import Path
from astropy.io import ascii
from astropy.table import Table
from math import sqrt

from simtools.util import config as cfg
from simtools.ray_tracing import RayTracing
from simtools.telescope_model import TelescopeModel

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.DEBUG)

config = cfg.loadConfig()


def computeRndaRange(tel, n=10):
    rndaStd = tel.getParameter('mirror_reflection_random_angle')
    rndaStd = rndaStd.split()
    rndaStd = float(rndaStd[0])
    rndaRange = np.linspace(rndaStd * 0.8, rndaStd * 1.2, n)
    return np.round(rndaRange, 8)


def loadResults(force=False):
    fileName = 'results.ecsv'
    if Path(fileName).exists() and not force:
        return dict(ascii.read(fileName, format='basic'))
    else:
        res = dict()
        res['rnda'] = list()
        res['mean_d80_cm'] = list()
        res['sig_d80_cm'] = list()
        return res


def sortResults(res):
    res['rnda'], res['mean_d80_cm'], res['sig_d80_cm'] = zip(*sorted(zip(
        res['rnda'],
        res['mean_d80_cm'],
        res['sig_d80_cm']
    )))


if __name__ == '__main__':

    measMean = 1.47
    measSig = 0.24

    site = 'south'
    version = 'prod4'
    label = 'derive_rnda'
    telescopeType = 'mst-flashcam'
    force = True

    tel = TelescopeModel(
        yamlDBPath=config['yamlDBPath'],
        filesLocation=config['outputLocation'],
        telescopeType=telescopeType,
        site=site,
        version=version,
        label=label
    )

    rndaRange = computeRndaRange(tel, n=10)
    results = loadResults(force=force)

    for iRnda, thisRnda in enumerate(rndaRange):
        if not force and thisRnda in results['rnda']:
            continue

        if iRnda < 2:
            numberOfRepetitions = 1
        elif iRnda < 4:
            numberOfRepetitions = 10
        else:
            numberOfRepetitions = 30
        numberOfRepetitions = 1

        tel.changeParameters(mirror_reflection_random_angle=str(thisRnda))
        ray = RayTracing(
            simtelSourcePath=config['simtelPath'],
            filesLocation=config['outputLocation'],
            telescopeModel=tel,
            singleMirrorMode=True,
            numberOfRepetitions=numberOfRepetitions
        )
        ray.simulate(test=True, force=True)
        ray.analyze(force=True)

        results['rnda'].append(thisRnda)
        results['mean_d80_cm'].append(ray.getMean('d80_cm'))
        results['sig_d80_cm'].append(ray.getStdDev('d80_cm') / sqrt(numberOfRepetitions))

    sortResults(results)
    table = Table(results)
    ascii.write(table, 'results.ecsv', format='basic', overwrite=True)

    plt.figure(figsize=(8, 6), tight_layout=True)
    ax = plt.gca()
    ax.set_xlabel('mirror_random_reflection_angle')
    ax.set_ylabel(r'$D_{80}$ [cm]')

    ax.errorbar(
        results['rnda'],
        results['mean_d80_cm'],
        yerr=results['sig_d80_cm'],
        color='r',
        marker='o',
        linestyle='--'
    )

    xlim = ax.get_xlim()
    ax.plot(xlim, [measMean, measMean], color='k', linestyle='-')
    ax.plot(xlim, [measMean + measSig / 2, measMean + measSig / 2], color='k', linestyle=':')
    ax.plot(xlim, [measMean - measSig / 2, measMean - measSig / 2], color='k', linestyle=':')

    plt.show()
