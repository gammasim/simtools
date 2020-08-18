#!/usr/bin/python3

'''
    Example: python applications/derive_mirror_rnda.py --tel_type mst-flashcam --mean_d80 1.4
        --no_tunning --mirror_list mirror_MST_focal_lengths.dat --d80_list mirror_MST_D80.dat
'''


import logging
import matplotlib.pyplot as plt
import argparse
from copy import copy
from pathlib import Path

import numpy as np
from astropy.io import ascii
from astropy.table import Table

import simtools.config as cfg
from simtools.util.general import sortArrays
from simtools.ray_tracing import RayTracing
from simtools.model.telescope_model import TelescopeModel

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.DEBUG)

# cfg.setConfigFileName('config.yml')
# config = cfg.loadConfig()

plt.rc('font', family='serif', size=20)
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
plt.rc('text', usetex=True)


def plotMeasuredDistribution(file, **kwargs):
    data = np.loadtxt(file)
    ax = plt.gca()
    ax.hist(data, **kwargs)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--tel_type',
        help='Telescope type (e.g. mst-flashcam, lst)',
        type=str,
        required=True
    )
    parser.add_argument(
        '--model_version',
        help='Model version (default=prod4)',
        type=str,
        default='prod4'
    )
    parser.add_argument(
        '--site',
        help='Site (default=South)',
        type=str,
        default='south'
    )
    parser.add_argument(
        '--mean_d80',
        help='Mean of measured D80 [cm]',
        type=float,
        required=True
    )
    parser.add_argument(
        '--sig_d80',
        help='Std dev of measured D80 [cm]',
        type=float,
        required=False
    )
    parser.add_argument(
        '--d80_list',
        help=(
            'File with single column list of measured D80 [cm].'
            ' If given, the measured distribution will be plotted'
            ' on the top of the simulated one.'
        ),
        type=str,
        required=False
    )
    parser.add_argument(
        '--rnda',
        help='Start value of mirror_reflection_random_angle',
        type=float,
        default=0.
    )
    parser.add_argument(
        '--no_tunning',
        help='Turn off the tunning - A single case will be simulated and plotted',
        action='store_true'
    )
    parser.add_argument(
        '--mirror_list',
        help=(
                'Mirror list file to replace the default one.'
                ' It should be used if measured mirror focal lengths need to be accounted'
        ),
        type=str,
        required=False
    )
    parser.add_argument(
        '--use_random_flen',
        help=(
            'Use random focal lengths. The argument random_flen'
            ' can be used to replace the default random_focal_length'
        ),
        action='store_true'
    )
    parser.add_argument(
        '--random_flen',
        help='Value to replace the default random_focal_length',
        type=float,
        required=False
    )
    parser.add_argument(
        '--test',
        help='Test option will be faster by simulating only few mirrors',
        action='store_true'
    )
    parser.add_argument(
        '-v',
        '--verbosity',
        dest='logLevel',
        action='store',
        default='info',
        help='Log level to print (default is INFO)'
    )

    args = parser.parse_args()

    logger = logging.getLogger('derive_mirror_rnda')
    logger.setLevel(gen.getLogLevelFromUser(args.logLevel))

    label = 'derive_rnda'
    tel = TelescopeModel(
        telescopeType=args.tel_type,
        site=args.site,
        version=args.model_version,
        label=label,
        logger=logger.name
    )
    if args.mirror_list is not None:
        mirrorListFile = cfg.findFile(name=args.mirror_list)
        tel.changeParameters(mirror_list=args.mirror_list)
    if args.random_flen is not None:
        tel.changeParameters(random_focal_length=str(args.random_flen))

    def run(rnda, plot=False):
        tel.changeParameters(mirror_reflection_random_angle=str(rnda))
        ray = RayTracing(
            telescopeModel=tel,
            singleMirrorMode=True,
            mirrorNumbers=list(range(1, 10)) if args.test else 'all',
            useRandomFocalLength=args.use_random_flen,
            logger=logger.name
        )
        ray.simulate(test=False, force=True)
        ray.analyze(force=True)

        if plot:
            # Plotting
            plt.figure(figsize=(8, 6), tight_layout=True)
            ax = plt.gca()
            ax.set_xlabel(r'D$_{80}$ [cm]')

            bins = np.linspace(0.8, 3.5, 27)
            ray.plotHistogram(
                'd80_cm',
                color='r',
                linestyle='-',
                alpha=0.5,
                facecolor='r',
                edgecolor='r',
                bins=bins
            )
            if args.d80_list is not None:
                d80ListFile = cfg.findFile(args.d80_list)
                plotMeasuredDistribution(
                    d80ListFile,
                    color='b',
                    linestyle='-',
                    facecolor='None',
                    edgecolor='b',
                    bins=bins
                )

        return ray.getMean('d80_cm'), ray.getStdDev('d80_cm')

    # First - rnda from previous model
    if args.rnda != 0:
        rndaStart = args.rnda
    else:
        rndaStart = tel.getParameter('mirror_reflection_random_angle')
        if isinstance(rndaStart, str):
            rndaStart = rndaStart.split()
            rndaStart = float(rndaStart[0])

    if not args.no_tunning:
        resultsRnda = list()
        resultsMean = list()
        resultsSig = list()

        def collectResults(rnda, mean, sig):
            resultsRnda.append(rnda)
            resultsMean.append(mean)
            resultsSig.append(sig)

        stop = False
        meanD80, sigD80 = run(rndaStart)
        rnda = rndaStart
        signDelta = np.sign(meanD80 - args.mean_d80)
        collectResults(rnda, meanD80, sigD80)
        while not stop:
            newRnda = rnda - (0.1 * rndaStart * signDelta)
            meanD80, sigD80 = run(newRnda)
            newSignDelta = np.sign(meanD80 - args.mean_d80)
            stop = (newSignDelta != signDelta)
            signDelta = newSignDelta
            rnda = newRnda
            collectResults(rnda, meanD80, sigD80)

        # interpolating
        resultsRnda, resultsMean, resultsSig = sortArrays(resultsRnda, resultsMean, resultsSig)
        rndaOpt = np.interp(x=args.mean_d80, xp=resultsMean, fp=resultsRnda)
    else:
        rndaOpt = rndaStart
    meanD80, sigD80 = run(rndaOpt, plot=True)

    print('--- Measured -----')
    if args.sig_d80 is not None:
        print('Mean = {:.3f}, StdDev = {:.3f}'.format(args.mean_d80, args.sig_d80))
    else:
        print('Mean = {:.3f}'.format(args.mean_d80))
    print('--- Simulated -----')
    print('Mean = {:.3f}, StdDev = {:.3f}'.format(meanD80, sigD80))
    print('--- mirror_random_reflection_angle ----')
    print('Previous value = {:.6f}'.format(rndaStart))
    print('New value = {:.6f}'.format(rndaOpt))
    print('-------')

    plt.figure(figsize=(8, 6), tight_layout=True)
    ax = plt.gca()
    ax.set_xlabel(r'mirror$\_$random$\_$reflection$\_$angle')
    ax.set_ylabel(r'$D_{80}$ [cm]')

    if not args.no_tunning:
        ax.errorbar(
            resultsRnda,
            resultsMean,
            yerr=resultsSig,
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
    ax.plot(xlim, [args.mean_d80, args.mean_d80], color='k', linestyle='-')
    if args.sig_d80 is not None:
        ax.plot(
            xlim,
            [args.mean_d80 + args.sig_d80, args.mean_d80 + args.sig_d80],
            color='k',
            linestyle=':'
        )
        ax.plot(
            xlim,
            [args.mean_d80 - args.sig_d80, args.mean_d80 - args.sig_d80],
            color='k',
            linestyle=':'
        )

    ax.legend(frameon=False, loc='upper left')
    plt.show()
