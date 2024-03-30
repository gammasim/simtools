#!/usr/bin/python3

import logging
import shutil

import astropy.units as u
import pytest
from astropy.table import Table

from simtools.camera_efficiency import CameraEfficiency

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def camera_efficiency_lst(telescope_model_lst, simtel_path):
    camera_efficiency_lst = CameraEfficiency(
        telescope_model=telescope_model_lst,
        label="validate_camera_efficiency",
        simtel_source_path=simtel_path,
        test=True,
    )
    return camera_efficiency_lst


@pytest.fixture
def camera_efficiency_sst(telescope_model_sst, simtel_path):
    camera_efficiency_sst = CameraEfficiency(
        telescope_model=telescope_model_sst,
        label="validate_camera_efficiency",
        simtel_source_path=simtel_path,
        test=True,
    )
    return camera_efficiency_sst


@pytest.fixture
def results_file(io_handler):
    test_file_name = (
        "tests/resources/"
        "camera-efficiency-table-North-LSTN-01-za20.0deg_azm000deg_validate_camera_efficiency.ecsv"
    )
    output_directory = io_handler.get_output_directory(
        label="validate_camera_efficiency",
        sub_dir="camera-efficiency",
        dir_type="test",
    )
    shutil.copy(test_file_name, output_directory)
    return output_directory.joinpath(test_file_name)


def test_from_kwargs(telescope_model_lst, simtel_path):
    tel_model = telescope_model_lst
    label = "test-from-kwargs"
    zenith_angle = 30 * u.deg
    ce = CameraEfficiency.from_kwargs(
        telescope_model=tel_model,
        simtel_source_path=simtel_path,
        label=label,
        zenith_angle=zenith_angle,
        test=True,
    )
    assert ce.config.zenith_angle == 30


def test_validate_telescope_model(simtel_path):
    with pytest.raises(ValueError):
        CameraEfficiency(telescope_model="bla_bla", simtel_source_path=simtel_path)


def test_load_files(camera_efficiency_lst):
    _name = "camera-efficiency-table-North-LSTN-01-za20.0deg_azm000deg_validate_camera_efficiency"
    assert camera_efficiency_lst._file["results"].name == _name + ".ecsv"
    _name = "camera-efficiency-North-LSTN-01-za20.0deg_azm000deg_validate_camera_efficiency"
    assert camera_efficiency_lst._file["simtel"].name == _name + ".dat"
    assert camera_efficiency_lst._file["log"].name == _name + ".log"


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


@pytest.mark.xfail(reason="Missing ray_tracing for prod6 in Derived-DB")
def test_calc_nsb_rate(telescope_model_lst, camera_efficiency_lst, results_file):
    camera_efficiency_lst._read_results()
    telescope_model_lst.export_model_files()
    _, nsb_rate_ref_conditions = camera_efficiency_lst.calc_nsb_rate()
    assert nsb_rate_ref_conditions == pytest.approx(0.24421390533203186)  # Value for Prod5 LST-1


def test_export_results(simtel_path, telescope_model_lst, caplog):
    config_data = {
        "zenith_angle": 20 * u.deg,
    }
    camera_efficiency = CameraEfficiency(
        telescope_model=telescope_model_lst,
        simtel_source_path=simtel_path,
        config_data=config_data,
        label="export_results",
    )
    camera_efficiency.export_results()
    assert "Cannot export results because they do not exist" in caplog.text


@pytest.mark.xfail(reason="Missing ray_tracing for prod6 in Derived-DB")
def test_results_summary(telescope_model_lst, camera_efficiency_lst, results_file):
    camera_efficiency_lst._read_results()
    telescope_model_lst.export_model_files()
    summary = camera_efficiency_lst.results_summary()
    assert "Results summary for LSTN-01" in summary
