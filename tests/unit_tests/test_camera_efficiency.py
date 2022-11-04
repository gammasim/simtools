#!/usr/bin/python3

import logging

import astropy.units as u
import pytest
from astropy.table import Table

from simtools.camera_efficiency import CameraEfficiency

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def camera_efficiency_lst(telescope_model_lst, simtelpath):
    camera_efficiency_lst = CameraEfficiency(
        telescope_model=telescope_model_lst, simtel_source_path=simtelpath, test=True
    )
    return camera_efficiency_lst


@pytest.fixture
def camera_efficiency_sst(telescope_model_sst, simtelpath):
    camera_efficiency_sst = CameraEfficiency(
        telescope_model=telescope_model_sst, simtel_source_path=simtelpath, test=True
    )
    return camera_efficiency_sst


@pytest.fixture
def results_file(db, io_handler):
    test_file_name = "camera-efficiency-North-LST-1-za20.0_validate_camera_efficiency.ecsv"
    db.export_file_db(
        db_name="test-data",
        dest=io_handler.get_output_directory(
            label="validate_camera_efficiency",
            dir_type="camera-efficiency",
            test=True,
        ),
        file_name=test_file_name,
    )

    return io_handler.get_output_directory(
        label="validate_camera_efficiency",
        dir_type="camera-efficiency",
        test=True,
    ).joinpath("camera-efficiency-North-LST-1-za20.0_validate_camera_efficiency.ecsv")


def test_from_kwargs(telescope_model_lst, simtelpath):

    tel_model = telescope_model_lst
    label = "test-from-kwargs"
    zenith_angle = 30 * u.deg
    ce = CameraEfficiency.from_kwargs(
        telescope_model=tel_model,
        simtel_source_path=simtelpath,
        label=label,
        zenith_angle=zenith_angle,
        test=True,
    )
    assert ce.config.zenith_angle == 30


def test_validate_telescope_model(simtelpath):

    with pytest.raises(ValueError):
        CameraEfficiency(telescope_model="bla_bla", simtel_source_path=simtelpath)


def test_load_files(camera_efficiency_lst):
    assert (
        camera_efficiency_lst._file_results.name
        == "camera-efficiency-North-LST-1-za20.0_validate_camera_efficiency.ecsv"
    )
    assert (
        camera_efficiency_lst._file_simtel.name
        == "camera-efficiency-North-LST-1-za20.0_validate_camera_efficiency.dat"
    )
    assert (
        camera_efficiency_lst._file_log.name
        == "camera-efficiency-North-LST-1-za20.0_validate_camera_efficiency.log"
    )


def test_read_results(camera_efficiency_lst, results_file):
    camera_efficiency_lst._read_results()
    assert isinstance(camera_efficiency_lst._results, Table)
    assert camera_efficiency_lst._has_results is True


def test_calc_camera_efficiency(telescope_model_lst, camera_efficiency_lst, results_file):
    camera_efficiency_lst._read_results()
    telescope_model_lst.export_model_files()
    assert camera_efficiency_lst.calc_camera_efficiency() == pytest.approx(
        0.24468117923810984
    )  # Value for Prod5 LST-1


def test_calc_tel_efficiency(telescope_model_lst, camera_efficiency_lst, results_file):
    camera_efficiency_lst._read_results()
    telescope_model_lst.export_model_files()
    assert camera_efficiency_lst.calc_tel_efficiency() == pytest.approx(
        0.23988884493787524
    )  # Value for Prod5 LST-1


def test_calc_tot_efficiency(telescope_model_lst, camera_efficiency_lst, results_file):
    camera_efficiency_lst._read_results()
    telescope_model_lst.export_model_files()
    assert camera_efficiency_lst.calc_tot_efficiency(
        camera_efficiency_lst.calc_tel_efficiency()
    ) == pytest.approx(
        0.48018680628175714
    )  # Value for Prod5 LST-1


def test_calc_reflectivity(camera_efficiency_lst, results_file):
    camera_efficiency_lst._read_results()
    assert camera_efficiency_lst.calc_reflectivity() == pytest.approx(
        0.9167918392938349
    )  # Value for Prod5 LST-1


def test_calc_nsb_rate(telescope_model_lst, camera_efficiency_lst, results_file):
    camera_efficiency_lst._read_results()
    telescope_model_lst.export_model_files()
    assert camera_efficiency_lst.calc_nsb_rate()[0] == pytest.approx(
        0.24421390533203186
    )  # Value for Prod5 LST-1


def test_get_one_dim_distribution(telescope_model_sst, camera_efficiency_sst):

    telescope_model_sst.export_model_files()
    camera_filter_file = camera_efficiency_sst._get_one_dim_distribution(
        "camera_filter", "camera_filter_incidence_angle"
    )
    assert camera_filter_file.exists()
