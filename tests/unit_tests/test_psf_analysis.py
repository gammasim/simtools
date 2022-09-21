#!/usr/bin/python3

import logging

import pytest

import simtools.io_handler as io
from simtools.psf_analysis import PSFImage

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_reading_simtel_file(cfg_setup):

    testFile = io.getDataFile(
        fileName="photons-LST-d12.0-za20.0-off0.000_lst_integral.lis", test=True
    )
    image = PSFImage(focalLength=2800.0)
    image.readPhotonListFromSimtelFile(testFile)
    logger.info(image.getPSF(0.8, "cm"))

    assert 3.2248259134010397 == pytest.approx(image.getPSF(0.8, "cm"))
