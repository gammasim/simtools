#!/usr/bin/python3

import logging
from pathlib import Path

import astropy.units as u
import pytest

from simtools.corsika_simtel.corsika_simtel_runner import CorsikaSimtelRunner
from simtools.model.array_model import ArrayModel

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def common_args(simtel_path_no_mock):
    return {
        "label": "test-corsika-simtel-runner",
        "simtel_source_path": simtel_path_no_mock,
    }


@pytest.fixture
def array_config_data():
    return {
        "site": "North",
        "layout_name": "test-layout",
        "model_version": "Prod5",
        "default": {"LST": "D234", "MST": "NectarCam-D"},
        "LST-01": "1",
    }


@pytest.fixture
def corsika_config_data(tmp_test_directory):
    return {
        "data_directory": str(tmp_test_directory) + "/test-output",
        "nshow": 10,
        "primary": "gamma",
        "erange": [100 * u.GeV, 1 * u.TeV],
        "eslope": -2,
        "zenith": 20 * u.deg,
        "azimuth": 0 * u.deg,
        "viewcone": 0 * u.deg,
        "cscat": [10, 1500 * u.m, 0],
    }


@pytest.fixture
def corsika_args(corsika_config_data, db_config, array_config_data):
    return {
        "mongo_db_config": db_config,
        "site": "North",
        "layout_name": array_config_data["layout_name"],
        "corsika_config_data": corsika_config_data,
    }


@pytest.fixture
def simtel_config_data(tmp_test_directory):
    return {
        "simtel_data_directory": str(tmp_test_directory) + "/test-output",
        "primary": "gamma",
        "zenith_angle": 20 * u.deg,
        "azimuth_angle": 0 * u.deg,
    }


@pytest.fixture
def array_model(array_config_data, io_handler, db_config, common_args):
    array_model = ArrayModel(
        label=common_args["label"], array_config_data=array_config_data, mongo_db_config=db_config
    )
    return array_model


@pytest.fixture
def simtel_args(array_model, simtel_config_data):
    return {"array_model": array_model, "config_data": simtel_config_data}


@pytest.fixture
def corsika_simtel_runner(common_args, corsika_args, simtel_args):

    corsika_simtel_runner = CorsikaSimtelRunner(
        common_args=common_args, corsika_args=corsika_args, simtel_args=simtel_args
    )
    return corsika_simtel_runner


def test_prepare_run_script(corsika_simtel_runner):
    # No run number is given

    script = corsika_simtel_runner.prepare_run_script()

    assert script.exists()
    with open(script, "r") as f:
        script_content = f.read()
        assert "/usr/bin/bash" in script_content
        assert "corsika_autoinputs" in script_content
        assert "sim_telarray/bin/pfp" not in script_content

    # Run number is given
    run_number = 3
    script = corsika_simtel_runner.prepare_run_script(run_number=run_number)

    assert script.exists()
    with open(script, "r") as f:
        script_content = f.read()
        assert "/usr/bin/bash" in script_content
        assert "corsika_autoinputs" in script_content
        assert "sim_telarray/bin/pfp" not in script_content
        assert "-R 3" in script_content


def test_prepare_run_script_with_invalid_run(corsika_simtel_runner):
    for run in [-2, "test"]:
        with pytest.raises(ValueError):
            _ = corsika_simtel_runner.prepare_run_script(run_number=run)


def test_export_multipipe_script(corsika_simtel_runner):

    corsika_simtel_runner.export_multipipe_script()
    script = Path(corsika_simtel_runner.corsika_config._config_file_path.parent).joinpath(
        corsika_simtel_runner.corsika_config.get_file_name("multipipe")
    )

    assert script.exists()
    with open(script, "r") as f:
        script_content = f.read()
        assert "bin/sim_telarray" in script_content
        assert "-C telescope_theta=20" in script_content
        assert "-C telescope_phi=0" in script_content
        assert "-C show=all" in script_content


def test_export_multipipe_executable(corsika_simtel_runner):

    corsika_simtel_runner.export_multipipe_script()
    multipipe_file = Path(corsika_simtel_runner.corsika_config._config_file_path.parent).joinpath(
        corsika_simtel_runner.corsika_config.get_file_name("multipipe")
    )
    corsika_simtel_runner._export_multipipe_executable(multipipe_file)
    script = Path(corsika_simtel_runner.corsika_config._config_file_path.parent).joinpath(
        "run_cta_multipipe"
    )

    assert script.exists()
    with open(script, "r") as f:
        script_content = f.read()
        assert "bin/multipipe_corsika" in script_content
        assert f"-c {multipipe_file}" in script_content
        assert "'Fan-out failed'" in script_content


def test_make_run_command(corsika_simtel_runner):

    command = corsika_simtel_runner._make_run_command(input_file="-", run_number=1)
    assert "bin/sim_telarray" in command
    assert "-C telescope_theta=20" in command
    assert "-C telescope_phi=0" in command
    assert "-C show=all" in command
    assert "run000001_gamma_za020deg_azm000deg_North_TestLayout_test" in command


def test_get_info_for_file_name(corsika_simtel_runner):

    info_for_file_name = corsika_simtel_runner.get_info_for_file_name(run_number=1)
    assert info_for_file_name["run"] == 1
    assert info_for_file_name["primary"] == "gamma"
    assert info_for_file_name["array_name"] == "TestLayout"
    assert info_for_file_name["site"] == "North"
    assert info_for_file_name["label"] == "test-corsika-simtel-runner"


def test_get_file_name(corsika_simtel_runner, io_handler):
    info_for_file_name = corsika_simtel_runner.get_info_for_file_name(run_number=1)

    # Test one case of a CORSIKA file. Other cases are tested in the corsika_runner tests
    file_name = "corsika_run000001_gamma_North_TestLayout_test-corsika-simtel-runner"
    assert corsika_simtel_runner.get_file_name(
        "corsika_autoinputs_log", **info_for_file_name
    ) == corsika_simtel_runner._corsika_log_dir.joinpath(f"log_{file_name}.log.gz")

    # Test the histogram case which calls the simtel_runner_array internally
    file_name = "run000001_gamma_za020deg_azm000deg_North_TestLayout_test-corsika-simtel-runner"
    assert corsika_simtel_runner.get_file_name(
        "histogram", **info_for_file_name
    ) == corsika_simtel_runner._simtel_log_dir.joinpath(f"{file_name}.hdata.zst")
