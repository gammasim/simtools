#!/usr/bin/python3

import logging
import matplotlib.pyplot as plt

import simtools.config as cfg
from simtools.model.telescope_model import TelescopeModel
from simtools.model.camera import Camera

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == '__main__':
    '''
    This is an example application to calculate the FoV of a camera, in this case LST is used.
    The application prints out the FoV and plots the camera.
    '''

    site = 'south'
    telescope = 'lst'
    version = 'prod4'
    label = 'lst-test'

    telModel = TelescopeModel(
        telescopeType=telescope,
        site=site,
        version=version,
        label=label
    )

    focalLength = float(telModel.getParameter('effective_focal_length'))
    camera = Camera(
        telescopeType=telModel.telescopeType,
        cameraConfigFile=telModel.getParameter('camera_config_file'),
        focalLength=focalLength
    )

    fov, rEdgeAvg = camera.calcFOV()

    print('\nEffective focal length = ' + '{0:.3f} cm'.format(focalLength))
    print('{0} FoV = {1:.3f} deg'.format(telModel.telescopeType, fov))
    print('Avg. edge radius = {0:.3f} cm\n'.format(rEdgeAvg))

    # Now plot the camera as well
    plt = camera.plotPixelLayout()
    plt.savefig('pixelLayout-LST.pdf', format='pdf', bbox_inches='tight')
    plt.clf()
