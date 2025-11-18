#!/usr/bin/python3

import copy
import logging
from pathlib import Path

import pytest

from simtools.runners.corsika_simtel_runner import (
    CorsikaRunner,
    CorsikaSimtelRunner,
    SimulatorArray,
)

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
def corsika_simtel_runner(corsika_config_mock_array_model, simtel_path):
    """CorsikaSimtelRunner object."""
    return CorsikaSimtelRunner(
        corsika_config=corsika_config_mock_array_model,
        simtel_path=simtel_path,
        label="test-corsika-simtel-runner",
        use_multipipe=True,
    )


@pytest.fixture
def corsika_simtel_runner_calibration(corsika_config_mock_array_model, simtel_path):
    """CorsikaSimtelRunner object."""
    return CorsikaSimtelRunner(
        corsika_config=corsika_config_mock_array_model,
        simtel_path=simtel_path,
        label="test-corsika-simtel-runner",
        use_multipipe=True,
        calibration_config={
            "run_mode": "pedestals_nsb_only",
            "number_of_events": 500,
            "nsb_scaling_factor": 1.0,
            "stars": "stars.txt",
        },
    )


def test_corsika_simtel_runner(corsika_simtel_runner):
    assert isinstance(corsika_simtel_runner.corsika_runner, CorsikaRunner)
    assert isinstance(corsika_simtel_runner.simulator_array[0], SimulatorArray)


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
    with pytest.raises(ValueError, match=r"^Invalid type of run number"):
        _ = corsika_simtel_runner.prepare_run_script(run_number=-2)
    with pytest.raises(ValueError, match=r"^could not convert string to float"):
        _ = corsika_simtel_runner.prepare_run_script(run_number="test")


def test_export_multipipe_script(corsika_simtel_runner_calibration, simtel_command, show_all):
    corsika_simtel_runner_calibration._export_multipipe_script(run_number=1)
    script = Path(
        corsika_simtel_runner_calibration.base_corsika_config.config_file_path.parent
    ).joinpath(
        corsika_simtel_runner_calibration.base_corsika_config.get_corsika_config_file_name(
            "multipipe"
        )
    )

    assert script.exists()
    with open(script) as f:
        script_content = f.read()
        assert simtel_command in script_content
        assert "-C telescope_theta=20" in script_content
        assert "-C telescope_phi=0" in script_content
        assert show_all in script_content

    corsika_simtel_runner_calibration._export_multipipe_script(run_number=1)
    script = Path(
        corsika_simtel_runner_calibration.base_corsika_config.config_file_path.parent
    ).joinpath(
        corsika_simtel_runner_calibration.base_corsika_config.get_corsika_config_file_name(
            "multipipe"
        )
    )

    assert script.exists()
    with open(script) as f:
        script_content = f.read()
        assert "-C fadc_lg_noise=0.0" in script_content


def test_write_multipipe_script(corsika_simtel_runner):
    corsika_simtel_runner._export_multipipe_script(run_number=1)
    multipipe_file = Path(
        corsika_simtel_runner.base_corsika_config.config_file_path.parent
    ).joinpath(corsika_simtel_runner.base_corsika_config.get_corsika_config_file_name("multipipe"))
    corsika_simtel_runner._write_multipipe_script(multipipe_file)
    script = Path(corsika_simtel_runner.base_corsika_config.config_file_path.parent).joinpath(
        "run_cta_multipipe"
    )

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
    multipipe_file = Path(
        corsika_simtel_runner.base_corsika_config.config_file_path.parent
    ).joinpath(corsika_simtel_runner.base_corsika_config.get_corsika_config_file_name("multipipe"))
    corsika_simtel_runner._write_multipipe_script(multipipe_file)
    script = Path(corsika_simtel_runner.base_corsika_config.config_file_path.parent).joinpath(
        "run_cta_multipipe"
    )

    # Assertions
    assert script.exists()
    with open(script) as f:
        script_content = f.read()
        assert "bin/multipipe_corsika" in script_content
        assert f"-c {multipipe_file}" in script_content
        assert "'Fan-out failed'" in script_content
        assert "--sequential" in script_content


def test_get_file_name(corsika_simtel_runner, simulation_file):
    assert (
        corsika_simtel_runner.get_file_name(
            simulation_software="corsika", file_type="log", run_number=1
        ).name
        == simulation_file + ".log.gz"
    )

    simtel_simulation_file = simulation_file + ".simtel.zst"

    assert (
        corsika_simtel_runner.get_file_name(
            simulation_software="sim_telarray", file_type="simtel_output", run_number=1
        ).name
        == simtel_simulation_file
    )

    # preference given to simtel runner
    assert (
        corsika_simtel_runner.get_file_name(
            simulation_software=None, file_type="simtel_output", run_number=1
        ).name
        == simtel_simulation_file
    )

    # no simulator_array
    _test_corsika_simtel_runner = copy.deepcopy(corsika_simtel_runner)
    _test_corsika_simtel_runner.simulator_array = None

    assert (
        _test_corsika_simtel_runner.get_file_name(
            simulation_software=None, file_type="simtel_output", run_number=1
        ).name
        == simtel_simulation_file
    )


def test_determine_pointing_option(corsika_simtel_runner):
    assert corsika_simtel_runner._determine_pointing_option(None) is False

    assert corsika_simtel_runner._determine_pointing_option("divergent") is True
    assert corsika_simtel_runner._determine_pointing_option("convergent") is True
    assert corsika_simtel_runner._determine_pointing_option("test") is False
