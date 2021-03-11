#!/usr/bin/python3

'''
    Summary
    -------

    This application creates the layout array files (ECSV) of regular arrays
    with one telescope at the center of the array and with 4 telescopes
    in a square grid. These arrays are used for trigger rate simulations.

    The array layout files created should be available at the data/layout directory.

    Command line arguments
    ----------------------
    site_pars (str, optional)
        Site parameters file in yaml format. If not given, the default one
        from data/layout will be used.
    verbosity (str, optional)
        Log level to print (default=INFO).

    Example
    -------
    Runtime < 1 min.

    python applications/make_regular_arrays.py
'''

import logging
import argparse

import astropy.units as u
from astropy.io.misc import yaml

import simtools.io_handler as io
import simtools.util.general as gen
from simtools.layout.layout_array import LayoutArray


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=(
            'Calculate and plot the PSF and eff. mirror area as a function of off-axis angle '
            'of the telescope requested.'
        )
    )
    parser.add_argument(
        '--site_pars',
        type=str,
        help=(
            'Site parameters file in yaml format. If not given, the '
            'default one from data/layout will be used.'
        )
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
    label = 'make_regular_arrays'

    logger = logging.getLogger(label)
    logger.setLevel(gen.getLogLevelFromUser(args.logLevel))

    # Getting site pars file
    if args.site_pars is not None:
        siteParsFile = args.site_pars
        logger.debug('Reading site parameters from {}'.format(siteParsFile))
    else:
        siteParsFile = io.getDataFile('layout', 'site_parameters.yml')
        logger.debug('Reading default site parameters from {}'.format(siteParsFile))

    try:
        with open(siteParsFile, 'r') as f:
            sitePars = yaml.load(f)
    except FileNotFoundError:
        msg = 'Site parameter file ({}) could not be opened/read'.format(siteParsFile)
        logger.error(msg)
        raise FileNotFoundError(msg)

    # Telescope distances for 4 tel square arrays
    # !HARDCODED
    telescopeDistance = {'LST': 57.5 * u.m, 'MST': 70 * u.m, 'SST': 80 * u.m}

    for site in ['South', 'North']:
        for arrayName in ['1SST', '4SST', '1MST', '4MST', '1LST', '4LST']:
            logger.info('Processing array {}'.format(arrayName))
            layout = LayoutArray(
                label=label,
                name=site + '-' + arrayName,
                logger=logger.name,
                **sitePars[site]
            )

            telNameRoot = arrayName[1]
            telSize = arrayName[1:4]

            # Single telescope at the center
            if arrayName[0] == '1':
                layout.addTelescope(
                    telescopeName=telNameRoot + '-01',
                    posX=0 * u.m,
                    posY=0 * u.m,
                    posZ=0 * u.m
                )
            # 4 telescopes in a regular square grid
            else:
                layout.addTelescope(
                    telescopeName=telNameRoot + '-01',
                    posX=telescopeDistance[telSize],
                    posY=telescopeDistance[telSize],
                    posZ=0 * u.m
                )
                layout.addTelescope(
                    telescopeName=telNameRoot + '-02',
                    posX=-telescopeDistance[telSize],
                    posY=telescopeDistance[telSize],
                    posZ=0 * u.m
                )
                layout.addTelescope(
                    telescopeName=telNameRoot + '-03',
                    posX=telescopeDistance[telSize],
                    posY=-telescopeDistance[telSize],
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
            layout.exportTelescopeList()
