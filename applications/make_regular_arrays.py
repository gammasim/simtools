#!/usr/bin/python3

'''
    Summary
    -------
    .. _compare_cumulative_psf_plot:
    .. image::  images/compare_cumulative_psf_North-LST-1_cumulativePSF.png
      :width: 49 %
    .. image::  images/compare_cumulative_psf_North-LST-1_image.png
      :width: 49 %

    Command line arguments
    ----------------------
    tel_name (str, required)
        Telescope name (e.g. North-LST-1, South-SST-D, ...).
    model_version (str, optional)
        Model version (default=prod4).
    src_distance (float, optional)
        Source distance in km (default=10).
    zenith (float, optional)
        Zenith angle in deg (default=20).
    data (str, optional)
        Name of the data file with the measured cumulative PSF.
    pars (str, optional)
        Yaml file with the new model parameters to replace the default ones.
    test (activation mode, optional)
        If activated, application will be faster by simulating fewer photons.
    verbosity (str, optional)
        Log level to print (default=INFO).

    Example
    -------
    LST-1 Prod5

    Runtime < 1 min.

    First, create an yml file named lst_pars.yml with the following content:

    .. code-block:: yaml

        mirror_reflection_random_angle: '0.0075,0.15,0.035'
        mirror_align_random_horizontal: '0.0040,28.,0.0,0.0'
        mirror_align_random_vertical: '0.0040,28.,0.0,0.0'

    And the run:

    .. code-block:: console

        python applications/compare_cumulative_psf.py --tel_name North-LST-1 \
        --model_version prod4 --pars lst_pars.yml --data PSFcurve_data_v2.txt

    .. todo::

        * Change default model to default (after this feature is implemented in db_handler)
'''

import logging
import argparse
# import yaml
from collections import OrderedDict

from astropy.io.misc import yaml

import numpy as np
import astropy.units as u

import simtools.io_handler as io
import simtools.util.general as gen
import simtools.config as cfg
from simtools.layout.layout_array import LayoutArray
from simtools.model.telescope_model import TelescopeModel
from simtools import visualize



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=(
            'Calculate and plot the PSF and eff. mirror area as a function of off-axis angle '
            'of the telescope requested.'
        )
    )
    # parser.add_argument(
    #     '-n',
    #     '--name',
    #     help='Telescope name (e.g. North-MST-FlashCam-D, North-LST-1)',
    #     type=str,
    #     required=True
    # )
    parser.add_argument(
        '-v',
        '--verbosity',
        dest='logLevel',
        action='store',
        default='info',
        help='Log level to print (default is INFO)'
    )

    args = parser.parse_args()
    label = 'make_regular_arrays'

    logger = logging.getLogger(label)
    logger.setLevel(gen.getLogLevelFromUser(args.logLevel))

    # Output directory to save files related directly to this app
    outputDir = io.getApplicationOutputDirectory(cfg.get('outputLocation'), label)

    siteParsFile = io.getDataFile('layout', 'site_parameters.yml')
    with open(siteParsFile, 'r') as f:
        sitePars = yaml.load(f)

    layout = LayoutArray(label=label, name='LST4', **sitePars['South'])

    # Defining the distance between telescopes
    telescopeDistance = {'LST': 57.5 * u.m, 'MST': 70 * u.m, 'SST': 80 * u.m}

    for site in ['South', 'North']:
        for arrayName in ['1SST', '4SST', '1MST', '4MST', '1LST', '4LST']:
            layout = LayoutArray(label=label, name=arrayName, **sitePars[site])

            telNameRoot = arrayName[1]
            telSize = arrayName[1:4]
            # Single telescope at the center of the array
            if arrayName[0] == '1':
                layout.addTelescope(
                    telescopeName=telNameRoot + '-01',
                    posX=0 * u.m,
                    posY=0 * u.m,
                    posZ=0 * u.m
                )
            # 4 telescope on a square grid
            else:  # arrayName[0] == '4'
                layout.addTelescope(
                    telescopeName=telNameRoot + '-01',
                    posX=telescopeDistance[telSize],
                    posY=telescopeDistance[telSize],
                    posZ=0 * u.m
                )
                layout.addTelescope(
                    telescopeName=telNameRoot + '-02',
                    posX=telescopeDistance[telSize],
                    posY=-telescopeDistance[telSize],
                    posZ=0 * u.m
                )
                layout.addTelescope(
                    telescopeName=telNameRoot + '-03',
                    posX=-telescopeDistance[telSize],
                    posY=telescopeDistance[telSize],
                    posZ=0 * u.m
                )
                layout.addTelescope(
                    telescopeName=telNameRoot + '-04',
                    posX=-telescopeDistance[telSize],
                    posY=-telescopeDistance[telSize],
                    posZ=0 * u.m
                )

            layout.convertCoordinates()
            layout.printTelescopeList()


    exit()


    telModel = TelescopeModel(
        telescopeName=args.tel_name,
        version=args.model_version,
        label=label,
        logger=logger.name
    )

    # New parameters
    if args.pars is not None:
        with open(args.pars) as file:
            newPars = yaml.load(file, Loader=yaml.FullLoader)
        telModel.changeParameters(**newPars)

    ray = RayTracing(
        telescopeModel=telModel,
        sourceDistance=args.src_distance * u.km,
        zenithAngle=args.zenith * u.deg,
        offAxisAngle=[0. * u.deg],
        logger=logger.name
    )

    ray.simulate(test=args.test, force=False)
    ray.analyze(force=False)

    # Plotting cumulative PSF
    im = ray.images()[0]

    print('d80 in cm = {}'.format(im.getPSF()))

    # Plotting cumulative PSF
    dataToPlot = OrderedDict()
    dataToPlot[r'sim$\_$telarray'] = im.getCumulativeData()
    if args.data is not None:
        dataFile = cfg.findFile(args.data)
        dataToPlot['measured'] = loadData(dataFile)
    plt = visualize.plot1D(dataToPlot)
    plt.gca().set_ylim(0, 1.05)

    plotFileName = label + '_' + telModel.telescopeName + '_cumulativePSF'
    plotFile = outputDir.joinpath(plotFileName)
    for f in ['pdf', 'png']:
        plt.savefig(str(plotFile) + '.' + f, format=f, bbox_inches='tight')
    plt.clf()

    # Plotting image
    dataToPlot = im.getImageData()
    visualize.plotHist2D(dataToPlot, bins=80)
    circle = plt.Circle((0, 0), im.getPSF(0.8) / 2, color='k', fill=False, lw=2, ls='--')
    plt.gca().add_artist(circle)

    plotFileName = label + '_' + telModel.telescopeName + '_image'
    plotFile = outputDir.joinpath(plotFileName)
    for f in ['pdf', 'png']:
        plt.savefig(str(plotFile) + '.' + f, format=f, bbox_inches='tight')
    plt.clf()
