#!/usr/bin/python3

import logging

import yaml

from simtools.psf_analysis import PSFImage
import simtools.io_handler as io

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def test_reading_simtel_file():
    testFile = io.getTestDataFile('photons-LST-d12.0-za20.0-off0.000_lst_integral.lis')
    image = PSFImage(focalLength=2800.)
    image.readSimtelFile(testFile)
    logger.info(image.getPSF(0.8, 'cm'))


if __name__ == '__main__':

    test_reading_simtel_file()
