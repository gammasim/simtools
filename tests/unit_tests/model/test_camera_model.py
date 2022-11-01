#!/usr/bin/python3

import logging

import pytest

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_get_number_of_pixels(telescope_model_lst):

    telescope_model_lst.export_model_files()
    assert telescope_model_lst.camera.get_number_of_pixels() == 1855  # Value for LST


def test_pixel_solid_angle(telescope_model_lst):

    telModel = telescope_model_lst
    telescope_model_lst.export_model_files()
    pixSolidAngle = telModel.camera.get_pixel_active_solid_angle()
    logger.debug(f"Pixel solid angle is {pixSolidAngle}")

    assert pixSolidAngle * 1e6 == pytest.approx(2.43, 0.01)  # Value for LST
