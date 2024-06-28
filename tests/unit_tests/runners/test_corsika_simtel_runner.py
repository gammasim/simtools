#!/usr/bin/python3

import logging
from pathlib import Path

import pytest

from simtools.runners.corsika_simtel_runner import CorsikaSimtelRunner

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture()
def common_args(simtel_path):
    return {
        "label": "test-corsika-simtel-runner",
        "simtel_path": simtel_path,
    }


@pytest.fixture()
def corsika_args(
    array_model_north,
    shower_config_data_north,
):
    # Remove the keys which are not necessary for general CORSIKA configuration
    for key_to_pop in ["site", "run_list", "run_range", "layout_name"]:
        shower_config_data_north.pop(key_to_pop, None)
    return {
        "array_model": array_model_north,
        "corsika_config_data": shower_config_data_north,
    }


@pytest.fixture()
def simtel_config_data(tmp_test_directory, simulator_config_data_north):
    return {
        "simtel_data_directory": str(tmp_test_directory) + "/test-output",
        "primary": simulator_config_data_north["common"]["primary"],
        "zenith_angle": simulator_config_data_north["common"]["zenith"],
        "azimuth_angle": simulator_config_data_north["common"]["azimuth"],
    }


@pytest.fixture()
def simtel_args(array_model_north, simtel_config_data):
    return {"array_model": array_model_north, "config_data": simtel_config_data}


@pytest.fixture()
def corsika_simtel_runner(common_args, corsika_args, simtel_args):
    return CorsikaSimtelRunner(
        common_args=common_args, corsika_args=corsika_args, simtel_args=simtel_args
    )


def test_prepare_run_script(corsika_simtel_runner):
    # No run number is given

    script = corsika_simtel_runner.prepare_run_script()

    assert script.exists()
    with open(script) as f:
        script_content = f.read()
        assert "/usr/bin/env bash" in script_content
        assert "corsika_autoinputs" in script_content
        assert "sim_telarray/bin/pfp" not in script_content

    # Run number is given
    run_number = 3
    script = corsika_simtel_runner.prepare_run_script(run_number=run_number)

    assert script.exists()
    with open(script) as f:
        script_content = f.read()
        assert "/usr/bin/env bash" in script_content
        assert "corsika_autoinputs" in script_content
        assert "sim_telarray/bin/pfp" not in script_content
        assert "-R 3" in script_content


def test_prepare_run_script_with_invalid_run(corsika_simtel_runner):
    for run in [-2, "test"]:
        with pytest.raises(ValueError):
            _ = corsika_simtel_runner.prepare_run_script(run_number=run)


def test_export_multipipe_script(corsika_simtel_runner):
    corsika_simtel_runner.export_multipipe_script()
    script = Path(corsika_simtel_runner.corsika_config.config_file_path.parent).joinpath(
        corsika_simtel_runner.corsika_config.get_file_name("multipipe")
    )

    assert script.exists()
    with open(script) as f:
        script_content = f.read()
        assert "bin/sim_telarray" in script_content
        assert "-C telescope_theta=20" in script_content
        assert "-C telescope_phi=0" in script_content
        assert "-C show=all" in script_content


def test_export_multipipe_executable(corsika_simtel_runner):
    corsika_simtel_runner.export_multipipe_script()
    multipipe_file = Path(corsika_simtel_runner.corsika_config.config_file_path.parent).joinpath(
        corsika_simtel_runner.corsika_config.get_file_name("multipipe")
    )
    corsika_simtel_runner._export_multipipe_executable(multipipe_file)
    script = Path(corsika_simtel_runner.corsika_config.config_file_path.parent).joinpath(
        "run_cta_multipipe"
    )

    assert script.exists()
    with open(script) as f:
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
    assert "run000001_gamma_za020deg_azm000deg_North_test_layout_test" in command


def test_make_run_command_divergent(corsika_simtel_runner):
    corsika_simtel_runner.label = "test-corsika-simtel-runner-divergent-pointing"
    command = corsika_simtel_runner._make_run_command(input_file="-", run_number=1)
    assert "bin/sim_telarray" in command
    assert "-W telescope_theta=20" in command
    assert "-W telescope_phi=0" in command
    assert "-C show=all" in command
    assert "run000001_gamma_za020deg_azm000deg_North_test_layout_test" in command
