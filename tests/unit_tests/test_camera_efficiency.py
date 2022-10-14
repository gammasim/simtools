#!/usr/bin/python3

import logging

import astropy.units as u
import pytest
from astropy.table import Table

import simtools.io_handler as io
from simtools import db_handler
from simtools.camera_efficiency import CameraEfficiency
from simtools.model.telescope_model import TelescopeModel

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def db(db_connection):
    db = db_handler.DatabaseHandler(mongoDBConfigFile=str(db_connection))
    return db


@pytest.fixture
def telescope_model(args_dict, db_connection):
    telescopeModel = TelescopeModel(
        site="North",
        telescopeModelName="LST-1",
        modelFilesLocations=args_dict["model_path"],
        filesLocation=args_dict["output_path"],
        modelVersion="Prod5",
        label="validate_camera_efficiency",
        mongoDBConfigFile=str(db_connection),
    )
    return telescopeModel


@pytest.fixture
def camera_efficiency(telescope_model, args_dict):
    camera_efficiency = CameraEfficiency(
        telescopeModel=telescope_model,
        simtelSourcePath=args_dict["simtelpath"],
        filesLocation=args_dict["output_path"],
        dataLocation=args_dict["data_path"],
        test=True,
    )
    return camera_efficiency


@pytest.fixture
def results_file(db, args_dict):
    testFileName = "camera-efficiency-North-LST-1-za20.0_validate_camera_efficiency.ecsv"
    db.exportFileDB(
        dbName="test-data",
        dest=io.getOutputDirectory(
            filesLocation=args_dict["output_path"],
            label="validate_camera_efficiency",
            dirType="camera-efficiency",
            test=True,
        ),
        fileName=testFileName,
    )

    return io.getOutputDirectory(
        filesLocation=args_dict["output_path"],
        label="validate_camera_efficiency",
        dirType="camera-efficiency",
        test=True,
    ).joinpath("camera-efficiency-North-LST-1-za20.0_validate_camera_efficiency.ecsv")


def test_from_kwargs(telescope_model, args_dict):

    telModel = telescope_model
    label = "test-from-kwargs"
    zenithAngle = 30 * u.deg
    ce = CameraEfficiency.fromKwargs(
        telescopeModel=telModel,
        simtelSourcePath=args_dict["simtelpath"],
        filesLocation=args_dict["output_path"],
        dataLocation=args_dict["data_path"],
        label=label,
        zenithAngle=zenithAngle,
        test=True,
    )
    assert ce.config.zenithAngle == 30


def test_validate_telescope_model(args_dict):

    with pytest.raises(ValueError):
        CameraEfficiency(
            telescopeModel="bla_bla",
            simtelSourcePath=args_dict["simtelpath"],
            filesLocation=args_dict["output_path"],
            dataLocation=args_dict["data_path"],
        )


def test_load_files(camera_efficiency):
    assert (
        camera_efficiency._fileResults.name
        == "camera-efficiency-North-LST-1-za20.0_validate_camera_efficiency.ecsv"
    )
    assert (
        camera_efficiency._fileSimtel.name
        == "camera-efficiency-North-LST-1-za20.0_validate_camera_efficiency.dat"
    )
    assert (
        camera_efficiency._fileLog.name
        == "camera-efficiency-North-LST-1-za20.0_validate_camera_efficiency.log"
    )


def test_read_results(camera_efficiency, results_file):
    camera_efficiency._readResults()
    assert isinstance(camera_efficiency._results, Table)
    assert camera_efficiency._hasResults is True


def test_calc_camera_efficiency(telescope_model, camera_efficiency, results_file):
    camera_efficiency._readResults()
    telescope_model.exportModelFiles()
    assert camera_efficiency.calcCameraEfficiency() == pytest.approx(
        0.24468117923810984
    )  # Value for Prod5 LST-1


def test_calc_tel_efficiency(telescope_model, camera_efficiency, results_file):
    camera_efficiency._readResults()
    telescope_model.exportModelFiles()
    assert camera_efficiency.calcTelEfficiency() == pytest.approx(
        0.23988884493787524
    )  # Value for Prod5 LST-1


def test_calc_tot_efficiency(telescope_model, camera_efficiency, results_file):
    camera_efficiency._readResults()
    telescope_model.exportModelFiles()
    assert camera_efficiency.calcTotEfficiency(
        camera_efficiency.calcTelEfficiency()
    ) == pytest.approx(
        0.48018680628175714
    )  # Value for Prod5 LST-1


def test_calc_reflectivity(camera_efficiency, results_file):
    camera_efficiency._readResults()
    assert camera_efficiency.calcReflectivity() == pytest.approx(
        0.9167918392938349
    )  # Value for Prod5 LST-1


def test_calc_nsb_rate(telescope_model, camera_efficiency, results_file):
    camera_efficiency._readResults()
    telescope_model.exportModelFiles()
    assert camera_efficiency.calcNsbRate()[0] == pytest.approx(
        0.24421390533203186
    )  # Value for Prod5 LST-1
