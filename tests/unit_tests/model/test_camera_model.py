#!/usr/bin/python3

import logging

import pytest

from simtools.model.telescope_model import TelescopeModel

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def telescope_model(set_db):
    telescopeModel = TelescopeModel(
        site="North",
        telescopeModelName="LST-1",
        modelVersion="Prod5",
        label="test-telescope-model",
    )
    telescopeModel.exportModelFiles()
    return telescopeModel


def test_pixel_solid_angle(telescope_model):

    telModel = telescope_model
    pixSolidAngle = telModel.camera.getPixelActiveSolidAngle()
    logger.debug(f"Pixel solid angle is {pixSolidAngle}")

    assert pixSolidAngle * 1e6 == pytest.approx(2.43, 0.01)  # Value for LST
