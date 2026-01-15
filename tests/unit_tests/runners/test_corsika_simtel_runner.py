#!/usr/bin/python3

import logging

import pytest

from simtools.runners.corsika_runner import CorsikaRunner
from simtools.runners.corsika_simtel_runner import CorsikaSimtelRunner
from simtools.simtel.simulator_array import SimulatorArray

logger = logging.getLogger()


@pytest.fixture
def simtel_command():
    """Basic simtel command."""
    return "bin/sim_telarray"


@pytest.fixture
def show_all():
    """Simtel show all options."""
    return "-C show=all"


@pytest.fixture
def simulation_file(model_version):
    """Base name for simulation test file."""
    return (
        f"proton_run000001_za20deg_azm000deg_South_test_layout_{model_version}_"
        "test-corsika-simtel-runner"
    )


@pytest.fixture
def corsika_simtel_runner(corsika_config_mock_array_model):
    """CorsikaSimtelRunner object."""
    return CorsikaSimtelRunner(
        corsika_config=corsika_config_mock_array_model,
        label="test-corsika-simtel-runner",
    )


@pytest.fixture
def corsika_simtel_runner_calibration(corsika_config_mock_array_model):
    """CorsikaSimtelRunner object."""
    return CorsikaSimtelRunner(
        corsika_config=corsika_config_mock_array_model,
        label="test-corsika-simtel-runner",
        is_calibration_run=True,
    )


def test_corsika_simtel_runner(corsika_simtel_runner):
    assert isinstance(corsika_simtel_runner.corsika_runner, CorsikaRunner)
    assert isinstance(corsika_simtel_runner.simulator_array[0], SimulatorArray)


def test_prepare_run(corsika_simtel_runner, tmp_path):
    # prepare_run now requires sub_script parameter and doesn't return the script path
    script_path = tmp_path / "test_script.sh"
    corsika_simtel_runner.prepare_run(run_number=1, sub_script=script_path)

    assert script_path.exists()
    with open(script_path) as f:
        script_content = f.read()
        assert "/usr/bin/env bash" in script_content

    # Run number is given
    run_number = 3
    script_path2 = tmp_path / "test_script2.sh"
    corsika_simtel_runner.prepare_run(run_number=run_number, sub_script=script_path2)

    assert script_path2.exists()
    with open(script_path2) as f:
        script_content = f.read()
        assert "/usr/bin/env bash" in script_content


def test_prepare_run_with_invalid_run(corsika_simtel_runner, tmp_path):
    script_path = tmp_path / "test_script.sh"
    with pytest.raises(ValueError, match=r"^Invalid type of run number"):
        corsika_simtel_runner.prepare_run(run_number=-2, sub_script=script_path)
    with pytest.raises(ValueError, match=r"^could not convert string to float"):
        corsika_simtel_runner.prepare_run(run_number="test", sub_script=script_path)


def test_export_multipipe_script(corsika_simtel_runner_calibration, simtel_command, show_all):
    corsika_simtel_runner_calibration._export_multipipe_script(run_number=1)
    script = corsika_simtel_runner_calibration.runner_service.get_file_name(
        "multi_pipe_config", run_number=1
    )

    assert script.exists()
    with open(script) as f:
        script_content = f.read()
        assert simtel_command in script_content
        assert "-C telescope_theta=20" in script_content
        assert "-C telescope_phi=0" in script_content
        assert show_all in script_content

    # Test second call
    corsika_simtel_runner_calibration._export_multipipe_script(run_number=1)
    script = corsika_simtel_runner_calibration.runner_service.get_file_name(
        "multi_pipe_config", run_number=1
    )

    assert script.exists()
    with open(script) as f:
        script_content = f.read()
        # For calibration runs, we should see noise settings
        if corsika_simtel_runner_calibration.base_corsika_config.is_calibration_run():
            assert "-C fadc_lg_noise=0.0" in script_content


def test_write_multipipe_script(corsika_simtel_runner):
    corsika_simtel_runner._export_multipipe_script(run_number=1)
    multipipe_file = corsika_simtel_runner.runner_service.get_file_name(
        "multi_pipe_config", run_number=1
    )
    corsika_simtel_runner._write_multipipe_script(multipipe_file, run_number=1)
    script = corsika_simtel_runner.runner_service.get_file_name("multi_pipe_script", run_number=1)

    assert script.exists()
    with open(script) as f:
        script_content = f.read()
        assert "bin/multipipe_corsika" in script_content
        assert f"-c {multipipe_file}" in script_content
        assert "'Fan-out failed'" in script_content
        assert "--sequential" not in script_content


def test_write_multipipe_script_sequential(corsika_simtel_runner):
    # Set the sequential attribute
    corsika_simtel_runner.sequential = "--sequential"

    # Export and write the multipipe script
    corsika_simtel_runner._export_multipipe_script(run_number=1)
    multipipe_file = corsika_simtel_runner.runner_service.get_file_name(
        "multi_pipe_config", run_number=1
    )
    corsika_simtel_runner._write_multipipe_script(multipipe_file, run_number=1)
    script = corsika_simtel_runner.runner_service.get_file_name("multi_pipe_script", run_number=1)

    # Assertions
    assert script.exists()
    with open(script) as f:
        script_content = f.read()
        assert "bin/multipipe_corsika" in script_content
        assert f"-c {multipipe_file}" in script_content
        assert "'Fan-out failed'" in script_content
        assert "--sequential" in script_content
