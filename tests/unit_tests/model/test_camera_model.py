#!/usr/bin/python3

import logging

import pytest

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_get_number_of_pixels(telescope_model_lst):

    telescope_model_lst.exportModelFiles()
    assert telescope_model_lst.camera.getNumberOfPixels() == 1855  # Value for LST


def test_pixel_solid_angle(telescope_model_lst):

    telModel = telescope_model_lst
    telescope_model_lst.exportModelFiles()
    pixSolidAngle = telModel.camera.getPixelActiveSolidAngle()
    logger.debug(f"Pixel solid angle is {pixSolidAngle}")

    assert pixSolidAngle * 1e6 == pytest.approx(2.43, 0.01)  # Value for LST
