#!/usr/bin/python3

import logging

import astropy.units as u
import pytest
from astropy.table import Table

import simtools.config as cfg
import simtools.io_handler as io
from simtools import db_handler
from simtools.camera_efficiency import CameraEfficiency

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def db(set_db):
    db = db_handler.DatabaseHandler()
    return db


@pytest.fixture
def camera_efficiency_lst(telescope_model_lst):
    camera_efficiency_lst = CameraEfficiency(telescopeModel=telescope_model_lst, test=True)
    return camera_efficiency_lst


@pytest.fixture
def camera_efficiency_sst(telescope_model_sst):
    camera_efficiency_sst = CameraEfficiency(telescopeModel=telescope_model_sst, test=True)
    return camera_efficiency_sst


@pytest.fixture
def results_file(db):
    testFileName = "camera-efficiency-North-LST-1-za20.0_validate_camera_efficiency.ecsv"
    db.exportFileDB(
        dbName="test-data",
        dest=io.getOutputDirectory(
            filesLocation=cfg.get("outputLocation"),
            label="validate_camera_efficiency",
            dirType="camera-efficiency",
            test=True,
        ),
        fileName=testFileName,
    )

    return io.getOutputDirectory(
        filesLocation=cfg.get("outputLocation"),
        label="validate_camera_efficiency",
        dirType="camera-efficiency",
        test=True,
    ).joinpath("camera-efficiency-North-LST-1-za20.0_validate_camera_efficiency.ecsv")


def test_from_kwargs(telescope_model_lst):

    telModel = telescope_model_lst
    label = "test-from-kwargs"
    zenithAngle = 30 * u.deg
    ce = CameraEfficiency.fromKwargs(
        telescopeModel=telModel, label=label, zenithAngle=zenithAngle, test=True
    )
    assert ce.config.zenithAngle == 30


def test_validate_telescope_model(cfg_setup):

    with pytest.raises(ValueError):
        CameraEfficiency(
            telescopeModel="bla_bla",
        )


def test_load_files(camera_efficiency_lst):
    assert (
        camera_efficiency_lst._fileResults.name
        == "camera-efficiency-North-LST-1-za20.0_validate_camera_efficiency.ecsv"
    )
    assert (
        camera_efficiency_lst._fileSimtel.name
        == "camera-efficiency-North-LST-1-za20.0_validate_camera_efficiency.dat"
    )
    assert (
        camera_efficiency_lst._fileLog.name
        == "camera-efficiency-North-LST-1-za20.0_validate_camera_efficiency.log"
    )


def test_read_results(camera_efficiency_lst, results_file):
    camera_efficiency_lst._readResults()
    assert isinstance(camera_efficiency_lst._results, Table)
    assert camera_efficiency_lst._hasResults is True


def test_calc_camera_efficiency(telescope_model_lst, camera_efficiency_lst, results_file):
    camera_efficiency_lst._readResults()
    telescope_model_lst.exportModelFiles()
    assert camera_efficiency_lst.calcCameraEfficiency() == pytest.approx(
        0.24468117923810984
    )  # Value for Prod5 LST-1


def test_calc_tel_efficiency(telescope_model_lst, camera_efficiency_lst, results_file):
    camera_efficiency_lst._readResults()
    telescope_model_lst.exportModelFiles()
    assert camera_efficiency_lst.calcTelEfficiency() == pytest.approx(
        0.23988884493787524
    )  # Value for Prod5 LST-1


def test_calc_tot_efficiency(telescope_model_lst, camera_efficiency_lst, results_file):
    camera_efficiency_lst._readResults()
    telescope_model_lst.exportModelFiles()
    assert camera_efficiency_lst.calcTotEfficiency(
        camera_efficiency_lst.calcTelEfficiency()
    ) == pytest.approx(
        0.48018680628175714
    )  # Value for Prod5 LST-1


def test_calc_reflectivity(camera_efficiency_lst, results_file):
    camera_efficiency_lst._readResults()
    assert camera_efficiency_lst.calcReflectivity() == pytest.approx(
        0.9167918392938349
    )  # Value for Prod5 LST-1


def test_calc_nsb_rate(telescope_model_lst, camera_efficiency_lst, results_file):
    camera_efficiency_lst._readResults()
    telescope_model_lst.exportModelFiles()
    assert camera_efficiency_lst.calcNsbRate()[0] == pytest.approx(
        0.24421390533203186
    )  # Value for Prod5 LST-1


def test_get_one_dim_distribution(telescope_model_sst, camera_efficiency_sst):

    telescope_model_sst.exportModelFiles()
    cameraFilterFile = camera_efficiency_sst._getOneDimDistribution(
        "camera_filter", "camera_filter_incidence_angle"
    )
    assert cameraFilterFile.exists()
