#!/usr/bin/python3

import astropy.units as u

import simtools.util.general as gen
from simtools.model.telescope_model import TelescopeModel
from simtools.ray_tracing import RayTracing


def test_config_data_from_dict(db_config, simtelpath, io_handler):

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
        mongoDBConfig=db_config,
    )

    ray = RayTracing(
        telescopeModel=tel,
        simtelSourcePath=simtelpath,
        configData=configData,
    )

    assert ray.config.zenithAngle == 30
    assert len(ray.config.offAxisAngle) == 2


def test_from_kwargs(db, io_handler, simtelpath):

    label = "test-from-kwargs"

    sourceDistance = 10 * u.km
    zenithAngle = 30 * u.deg
    offAxisAngle = [0, 2] * u.deg

    testFileName = "CTA-North-LST-1-Current_test-telescope-model.cfg"
    db.exportFileDB(
        dbName="test-data",
        dest=io_handler.getOutputDirectory(dirType="model", test=True),
        fileName=testFileName,
    )

    cfgFile = gen.findFile(testFileName, io_handler.getOutputDirectory(dirType="model", test=True))

    tel = TelescopeModel.fromConfigFile(
        site="north",
        telescopeModelName="lst-1",
        configFileName=cfgFile,
        label=label,
    )

    ray = RayTracing.fromKwargs(
        telescopeModel=tel,
        simtelSourcePath=simtelpath,
        sourceDistance=sourceDistance,
        zenithAngle=zenithAngle,
        offAxisAngle=offAxisAngle,
    )

    assert ray.config.zenithAngle == 30
    assert len(ray.config.offAxisAngle) == 2
