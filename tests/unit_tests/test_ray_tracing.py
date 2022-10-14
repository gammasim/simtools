#!/usr/bin/python3

import astropy.units as u
import pytest

import simtools.io_handler as io
import simtools.util.general as gen
from simtools import db_handler
from simtools.model.telescope_model import TelescopeModel
from simtools.ray_tracing import RayTracing


@pytest.fixture
def db(db_connection):
    db = db_handler.DatabaseHandler(mongoDBConfigFile=str(db_connection))
    return db


def test_config_data_from_dict(args_dict, db_connection):

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
        modelFilesLocations=args_dict["model_path"],
        filesLocation=args_dict["output_path"],
        modelVersion=version,
        label=label,
        mongoDBConfigFile=str(db_connection),
    )

    ray = RayTracing(
        telescopeModel=tel,
        simtelSourcePath=args_dict["simtelpath"],
        filesLocation=args_dict["output_path"],
        dataLocation=args_dict["data_path"],
        configData=configData,
    )

    assert ray.config.zenithAngle == 30
    assert len(ray.config.offAxisAngle) == 2


def test_from_kwargs(args_dict, db):

    label = "test-from-kwargs"

    sourceDistance = 10 * u.km
    zenithAngle = 30 * u.deg
    offAxisAngle = [0, 2] * u.deg

    testFileName = "CTA-North-LST-1-Current_test-telescope-model.cfg"
    db.exportFileDB(
        dbName="test-data",
        dest=io.getOutputDirectory(
            filesLocation=args_dict["output_path"], dirType="model", test=True
        ),
        fileName=testFileName,
    )

    cfgFile = gen.findFile(
        testFileName, io.getOutputDirectory(args_dict["output_path"], dirType="model", test=True)
    )

    tel = TelescopeModel.fromConfigFile(
        site="north",
        telescopeModelName="lst-1",
        modelFilesLocations=args_dict["model_path"],
        filesLocation=args_dict["output_path"],
        configFileName=cfgFile,
        label=label,
    )

    ray = RayTracing.fromKwargs(
        telescopeModel=tel,
        simtelSourcePath=args_dict["simtelpath"],
        filesLocation=args_dict["output_path"],
        dataLocation=args_dict["data_path"],
        sourceDistance=sourceDistance,
        zenithAngle=zenithAngle,
        offAxisAngle=offAxisAngle,
    )

    assert ray.config.zenithAngle == 30
    assert len(ray.config.offAxisAngle) == 2
