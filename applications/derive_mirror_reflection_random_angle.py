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
logging.getLogger().setLevel(logging.INFO)

config = cfg.loadConfig()


def sortResults(res):
    res['mean'], res['rnda'], res['sig'] = zip(*sorted(zip(
        res['mean'],
        res['rnda'],
        res['sig']
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
    tel.changeParameters(mirror_list='mirror_CTA-Raul.dat')

    def run(rnda, plot=False):
        tel.changeParameters(mirror_reflection_random_angle=str(rnda))
        ray = RayTracing(
            simtelSourcePath=config['simtelPath'],
            filesLocation=config['outputLocation'],
            telescopeModel=tel,
            singleMirrorMode=True,
            mirrorNumbers='all'  # list(range(1, 50))
        )
        ray.simulate(test=False, force=True)
        ray.analyze(force=True)

        if plot:
            # Plotting
            plt.figure(figsize=(8, 6), tight_layout=True)
            ax = plt.gca()
            ax.set_xlabel('d80')

            ray.plotHistogram('d80_cm', color='r', bins=20)

        return ray.getMean('d80_cm'), ray.getStdDev('d80_cm')

    # First - rnda from previous model
    rndaStart = tel.getParameter('mirror_reflection_random_angle')
    if isinstance(rndaStart, str):
        rndaStart = rndaStart.split()
        rndaStart = float(rndaStart[0])

    results = dict()
    results['rnda'] = list()
    results['mean'] = list()
    results['sig'] = list()

    def collectResults(rnda, mean, sig):
        results['rnda'].append(rnda)
        results['mean'].append(mean)
        results['sig'].append(sig)

    stop = False
    meanD80, sigD80 = run(rndaStart)
    rnda = rndaStart
    signDelta = np.sign(meanD80 - measMean)
    collectResults(rnda, meanD80, sigD80)
    while not stop:
        newRnda = rnda - (0.1 * rndaStart * signDelta)
        meanD80, sigD80 = run(newRnda)
        newSignDelta = np.sign(meanD80 - measMean)
        stop = (newSignDelta != signDelta)
        signDelta = newSignDelta
        rnda = newRnda
        collectResults(rnda, meanD80, sigD80)

    # interpolating
    sortResults(results)
    rndaOpt = np.interp(x=measMean, xp=results['mean'], fp=results['rnda'])
    meanD80, sigD80 = run(rndaOpt, plot=True)

    print('--- Measured -----')
    print('Mean = {:.3f}, StdDev = {:.3f}'.format(measMean, measSig))
    print('--- Simulated -----')
    print('Mean = {:.3f}, StdDev = {:.3f}'.format(meanD80, sigD80))
    print('--- mirror_random_reflection_angle ----')
    print('Previous value = {:.6f}'.format(rndaStart))
    print('New value = {:.6f}'.format(rndaOpt))
    print('-------')

    plt.figure(figsize=(8, 6), tight_layout=True)
    ax = plt.gca()
    ax.set_xlabel('mirror_random_reflection_angle')
    ax.set_ylabel(r'$D_{80}$ [cm]')

    ax.errorbar(
        results['rnda'],
        results['mean'],
        yerr=results['sig'],
        color='k',
        marker='o',
        linestyle='none'
    )
    ax.errorbar(
        [rndaOpt],
        [meanD80],
        yerr=[sigD80],
        color='r',
        marker='o',
        linestyle='none',
        label='rnda = {:.6f} (mean = {:.3f}, sig = {:.3f})'.format(rndaOpt, meanD80, sigD80)
    )

    xlim = ax.get_xlim()
    ax.plot(xlim, [measMean, measMean], color='k', linestyle='-')
    ax.plot(xlim, [measMean + measSig, measMean + measSig], color='k', linestyle=':')
    ax.plot(xlim, [measMean - measSig, measMean - measSig], color='k', linestyle=':')

    ax.legend(frameon=False, loc='upper left')

    plt.show()
