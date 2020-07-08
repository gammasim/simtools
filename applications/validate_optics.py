#!/usr/bin/python3

import logging
import matplotlib.pyplot as plt
import argparse
import numpy as np

import astropy.units as u

import simtools.config as cfg
import simtools.util.general as gen
from simtools.model.telescope_model import TelescopeModel
from simtools.ray_tracing import RayTracing


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=(
            'Calculate the camera FoV of the telescope requested. '
            'Plot the camera as well, as seen for an observer facing the camera.'
        )
    )
    parser.add_argument(
        '-t',
        '--tel_type',
        help='Telescope type (e.g. mst-flashcam, lst)',
        type=str,
        required=True
    )
    parser.add_argument(
        '-m',
        '--model_version',
        help='Model version (default=prod4)',
        type=str,
        default='prod4'
    )
    parser.add_argument(
        '-l',
        '--label',
        help='Label (default=validate-optics)',
        type=str,
        default='validate-optics'
    )
    parser.add_argument(
        '-s',
        '--site',
        help='Site (default=South)',
        type=str,
        default='South'
    )
    parser.add_argument(
        '-d',
        '--src_distance',
        help='Source distance in km (default=10)',
        type=float,
        default=10
    )
    parser.add_argument(
        '-z',
        '--zenith',
        help='Zenith angle in deg (default=20)',
        type=float,
        default=20
    )
    parser.add_argument(
        '--max_offset',
        help='Maximum offset angle in deg (default=4)',
        type=float,
        default=4
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

    logger = logging.getLogger('validate_optics')
    logger.setLevel(gen.getLogLevelFromUser(args.logLevel))

    telModel = TelescopeModel(
        telescopeType=args.tel_type,
        site=args.site,
        version=args.model_version,
        label=args.label
    )

    print(
        '\nValidating telescope optics with ray tracing simulations'
        'for {}\n'.format(telModel.telescopeType)
    )

    ray = RayTracing(
        telescopeModel=telModel,
        sourceDistance=args.src_distance * u.km,
        zenithAngle=args.zenith * u.deg,
        offAxisAngle=np.linspace(0, args.max_offset, int(args.max_offset / 0.2) + 1) * u.deg,
        logger=logger.name
    )
    ray.simulate(test=False, force=False)
    ray.analyze(force=False)

    # Plotting
    ray.plotAndSave('d80_deg', marker='o', linestyle=':', color='k')
    ray.plotAndSave('d80_cm', marker='o', linestyle=':', color='k')
    ray.plotAndSave('eff_area', marker='o', linestyle=':', color='k')
    ray.plotAndSave('eff_flen', marker='o', linestyle=':', color='k')
