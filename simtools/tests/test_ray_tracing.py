#!/usr/bin/python3

import pytest
import logging

import astropy.units as u

import simtools.io_handler as io
from simtools.ray_tracing import RayTracing
from simtools.model.telescope_model import TelescopeModel
from simtools.util.tests import (
    has_db_connection,
    DB_CONNECTION_MSG,
)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.mark.skipif(not has_db_connection(), reason=DB_CONNECTION_MSG)
def test_config_data():

    label = "test-config-data"
    version = "prod4"

    configData = {
        "sourceDistance": 10 * u.km,
        "zenithAngle": 30 * u.deg,
        "offAxisAngle": [0, 2] * u.deg,
    }

    tel = TelescopeModel(
        site="north",
        telescopeModelName="mst-FlashCam-D",
        modelVersion=version,
        label=label,
    )

    ray = RayTracing(telescopeModel=tel, configData=configData)

    assert ray.config.zenithAngle == 30
    assert len(ray.config.offAxisAngle) == 2


def test_from_kwargs():

    label = "test-from-kwargs"

    sourceDistance = 10 * u.km
    zenithAngle = 30 * u.deg
    offAxisAngle = [0, 2] * u.deg

    cfgFile = io.getTestDataFile("CTA-North-LST-1-Current_test-telescope-model.cfg")

    tel = TelescopeModel.fromConfigFile(
        site="north",
        telescopeModelName="lst-1",
        configFileName=cfgFile,
        label=label,
    )

    ray = RayTracing.fromKwargs(
        telescopeModel=tel,
        sourceDistance=sourceDistance,
        zenithAngle=zenithAngle,
        offAxisAngle=offAxisAngle,
    )

    assert ray.config.zenithAngle == 30
    assert len(ray.config.offAxisAngle) == 2
