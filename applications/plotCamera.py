#!/usr/bin/python3

import logging
import matplotlib.pyplot as plt
from simtools.model.telescope_model import TelescopeModel
import simtools.config as cfg
from simtools.camera import Camera

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == '__main__':

    site = 'south'
    version = 'prod4'
    label = 'lst-test'

    tel = TelescopeModel(
        telescopeType='lst',
        site=site,
        version=version,
        label=label
    )

    camera = Camera()

    pixelLayoutFile = tel.getParameter('camera_config_file')
    plt = camera.readPixelList(cfg.findFile(pixelLayoutFile))
    plt.savefig('pixelLayout-LST.pdf', format='pdf', bbox_inches='tight')
    plt.clf()
