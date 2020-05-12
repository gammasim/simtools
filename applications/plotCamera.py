#!/usr/bin/python3

import logging
import matplotlib.pyplot as plt
from simtools.model.telescope_model import TelescopeModel
import simtools.config as cfg
from simtools.camera import Camera

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == '__main__':
    '''
    This is an example application to calculate the FoV of a camera, in this case LST is used.
    The application prints out the FoV and plots the camera.
    '''

    site = 'south'
    version = 'prod4'
    label = 'lst-test'

    telModel = TelescopeModel(
        telescopeType='lst',
        site=site,
        version=version,
        label=label
    )

    camera = Camera()

    focalLength = float(telModel.getParameter('effective_focal_length'))
    pixelLayoutFile = telModel.getParameter('camera_config_file')

    pixels = camera.readPixelList(cfg.findFile(pixelLayoutFile))
    neighbours = camera.getNeighbourPixels(pixels)
    edgePixelIndices = camera.getEdgePixels(pixels, neighbours)
    fov, rEdgeAvg = camera.calcFOV(pixels['x'], pixels['y'], edgePixelIndices, focalLength)

    print('\nEffective focal length = ' + '{0:.3f} cm'.format(focalLength))
    print('{0} FoV = {1:.3f} deg'.format(telModel.telescopeType, fov))
    print('Avg. edge radius = {0:.3f} cm\n'.format(rEdgeAvg))

    # Now plot the camera as well
    plt = camera.plotPixelLayout(telModel.telescopeType, pixels, focalLength)
    plt.savefig('pixelLayout-LST.pdf', format='pdf', bbox_inches='tight')
    plt.clf()
