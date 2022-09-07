#!/usr/bin/python3

import astropy.units as u
import pytest

import simtools.config as cfg
import simtools.io_handler as io
from simtools import db_handler
from simtools.model.telescope_model import TelescopeModel
from simtools.ray_tracing import RayTracing


@pytest.fixture
def db(set_db):
    db = db_handler.DatabaseHandler()
    return db


def test_config_data_from_dict(set_db):

    label = "test-config-data"
    version = "prod5"

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


def test_from_kwargs(db):

    label = "test-from-kwargs"

    sourceDistance = 10 * u.km
    zenithAngle = 30 * u.deg
    offAxisAngle = [0, 2] * u.deg

    testFileName = "CTA-North-LST-1-Current_test-telescope-model.cfg"
    db.exportFileDB(dbName="test-data", dest=io.getTestModelDirectory(), fileName=testFileName)

    cfgFile = cfg.findFile(testFileName, io.getTestModelDirectory())

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
