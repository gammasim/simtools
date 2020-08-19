#!/usr/bin/python3

import logging
import matplotlib.pyplot as plt
import argparse

import simtools.config as cfg
import simtools.util.general as gen
from simtools.model.telescope_model import TelescopeModel
from simtools.model.camera import Camera


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=(
            'Calculate the camera FoV of the telescope requested. '
            'Plot the camera as well, as seen for an observer facing the camera.'
        )
    )
    parser.add_argument(
        '--tel_type',
        help='Telescope type (e.g. north-lst-1, south-sst-d)',
        type=str,
        required=True
    )
    parser.add_argument(
        '-l',
        '--label',
        help='Label (default=validate-FoV)',
        type=str,
        default='validate-FoV'
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
        '-v',
        '--verbosity',
        dest='logLevel',
        action='store',
        default='info',
        help='Log level to print (default is INFO)'
    )

    args = parser.parse_args()

    logger = logging.getLogger('validate_camera_fov')
    logger.setLevel(gen.getLogLevelFromUser(args.logLevel))

    telModel = TelescopeModel(
        telescopeType=args.tel_type,
        version=args.model_version,
    )

    print('\nValidating the camera FoV of {}\n'.format(telModel.telescopeType))

    cameraConfigFile = telModel.getParameter('camera_config_file')
    focalLength = float(telModel.getParameter('effective_focal_length'))
    camera = Camera(
        telescopeType=telModel.telescopeType,
        cameraConfigFile=cfg.findFile(cameraConfigFile),
        focalLength=focalLength,
        logger=logger.name
    )

    fov, rEdgeAvg = camera.calcFOV()

    print('\nEffective focal length = ' + '{0:.3f} cm'.format(focalLength))
    print('{0} FoV = {1:.3f} deg'.format(telModel.telescopeType, fov))
    print('Avg. edge radius = {0:.3f} cm\n'.format(rEdgeAvg))

    # Now plot the camera as well
    plt = camera.plotPixelLayout()
    cameraPlotFile = 'pixelLayout-{}.pdf'.format(telModel.telescopeType)
    plt.savefig(cameraPlotFile, bbox_inches='tight')
    print('\nPlotted camera in {}\n'.format(cameraPlotFile))
    plt.clf()
