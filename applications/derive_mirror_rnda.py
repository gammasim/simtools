#!/usr/bin/python3

'''
    Summary:
    --------
    This application derives the parameter mirror_reflection_random_angle (mirror roughness) \
    for a given set of measured D80 of individual mirrors.

    .. _deriva_rnda_plot:
    .. figure::  images/
      :align:   center

    Todo:
    -----

        * Export figures with a proper name
        * Comment code


    Command line arguments:
    -----------------------
    tel_name (str, required)
        Telescope name (e.g. North-LST-1, South-SST-D, ...)
    model_version (str, optional)
        Model version (default=prod4)
    mean_d80 (float, required)
        Mean of measured D80 [cm]
    sig_d80 (float, optional)
        Std dev of measured D80 [cm]
    rnda (float, optional)
        Starting value of mirror_reflection_random_angle. If not given, the value from the default \
        model will be used.
    d80_list (file, optional)
        File with single column list of measured D80 [cm]. It is used only for plotting the D80 \
        distributions. If given, the measured distribution will be plotted on the top of the \
        simulated one.
    mirror_list (file, optional)
        Mirror list file (in sim_telarray format) to replace the default one. It should be used \
        if measured mirror focal lengths need to be taken into account.
    use_random_flen (activation mode, optional)
        Use random focal lengths, instead of the measured ones. The argument random_flen can be \
        used to replace the default random_focal_length from the model.
    random_flen (float, optional)
        Value to replace the default random_focal_length. Only used if use_random_flen is activated.
    test (activation mode, optional)
        If activated, application will be faster by simulating only few mirrors.
    verbosity (str, optional)
        Log level to print (default=INFO).

    Examples:
    ---------
    MST - Prod5 (07.2020)

    .. code-block:: console

        python applications/derive_mirror_rnda.py --tel_name north-mst-flashcam --mean_d80 1.4 --no_tunning --mirror_list mirror_MST_focal_lengths.dat --d80_list mirror_MST_D80.dat
'''


import logging
import matplotlib.pyplot as plt
import argparse
from copy import copy
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.io import ascii
from astropy.table import Table

import simtools.config as cfg
import simtools.util.general as gen
import simtools.io_handler as io
from simtools.util.general import sortArrays
from simtools.ray_tracing import RayTracing
from simtools.model.telescope_model import TelescopeModel
from simtools.visualize import setStyle

setStyle()


def plotMeasuredDistribution(file, **kwargs):
    data = np.loadtxt(file)
    ax = plt.gca()
    ax.hist(data, **kwargs)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--tel_name',
        help='Telescope name (e.g. North-LST-1, South-SST-D, ...)',
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
            'File with single column list of measured D80 [cm]. If given, the measured distribution'
            ' will be plotted on the top of the simulated one.'
        ),
        type=str,
        required=False
    )
    parser.add_argument(
        '--rnda',
        help='Starting value of mirror_reflection_random_angle',
        type=float,
        default=0.
    )
    parser.add_argument(
        '--no_tunning',
        help='Turn off the tunning - A single case will be simulated and plotted.',
        action='store_true'
    )
    parser.add_argument(
        '--mirror_list',
        help=(
            'Mirror list file to replace the default one. It should be used if measured mirror'
            ' focal lengths need to be accounted'
        ),
        type=str,
        required=False
    )
    parser.add_argument(
        '--use_random_flen',
        help=(
            'Use random focal lengths. The argument random_flen can be used to replace the default'
            ' random_focal_length parameter.'
        ),
        action='store_true'
    )
    parser.add_argument(
        '--random_flen',
        help='Value to replace the default random_focal_length.',
        type=float,
        required=False
    )
    parser.add_argument(
        '--test',
        help='Test option will be faster by simulating only 10 mirrors.',
        action='store_true'
    )
    parser.add_argument(
        '-v',
        '--verbosity',
        dest='logLevel',
        action='store',
        default='info',
        help='Log level to print (default is INFO).'
    )

    args = parser.parse_args()
    label = 'derive_mirror_rnda'

    logger = logging.getLogger(label)
    logger.setLevel(gen.getLogLevelFromUser(args.logLevel))

    tel = TelescopeModel(
        telescopeName=args.tel_name,
        version=args.model_version,
        label=label
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
        ray.simulate(test=False, force=False)
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

        return ray.getMean('d80_cm').to(u.cm).value, ray.getStdDev('d80_cm').to(u.cm).value

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

    # Printing results to stdout
    print('\nMeasured D80:')
    if args.sig_d80 is not None:
        print('Mean = {:.3f} cm, StdDev = {:.3f} cm'.format(args.mean_d80, args.sig_d80))
    else:
        print('Mean = {:.3f} cm'.format(args.mean_d80))
    print('\nSimulated D80:')
    print('Mean = {:.3f} cm, StdDev = {:.3f} cm'.format(meanD80, sigD80))
    print('\nmirror_random_reflection_angle')
    print('Previous value = {:.6f}'.format(rndaStart))
    print('New value = {:.6f}\n'.format(rndaOpt))

    # Plotting
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

    outputDir = io.getApplicationOutputDirectory(cfg.get('outputLocation'), label)

    plotFileName = label + '_' + tel.telescopeName
    plotFile = outputDir.joinpath(plotFileName)
    plt.savefig(str(plotFile) + '.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(str(plotFile) + '.png', format='png', bbox_inches='tight')
