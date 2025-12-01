#!/usr/bin/python3

import logging
import stat
from pathlib import Path

import pytest

from simtools.runners.simtel_runner import InvalidOutputFileError
from simtools.simtel.simulator_array import SimulatorArray

logger = logging.getLogger()


@pytest.fixture
def simtel_runner(corsika_config_mock_array_model):
    return SimulatorArray(
        corsika_config=corsika_config_mock_array_model,
        label="test-simtel-runner",
        use_multipipe=False,
    )


def test_simtel_runner(simtel_runner):
    sr = simtel_runner
    assert "sim_telarray" in str(sr._directory["data"])
    assert isinstance(sr._directory["data"], Path)


def test_make_run_command(simtel_runner):
    input_file = "test_make_run_command.inp"
    run_command = simtel_runner.make_run_command(
        run_number=3,
        input_file=input_file,
    )
    assert "sim_telarray" in run_command
    assert "-run" in run_command
    assert "3" in run_command
    assert "test-simtel-runner.simtel.zst" in run_command
    assert "test_make_run_command.inp" in run_command
    assert "random_seed" not in run_command

    simtel_runner.sim_telarray_seeds = {
        "seed": 12345,
        "random_instrument_instances": None,
        "seed_file_name": None,
    }
    run_command = simtel_runner.make_run_command(
        run_number=3,
        input_file=input_file,
    )
    assert "random_seed" in run_command
    assert "12345" in run_command

    simtel_runner.sim_telarray_seeds = {
        "seed": None,
        "random_instrument_instances": 10,
        "seed_file_name": "test_file_with_seeds.txt",
    }
    run_command = simtel_runner.make_run_command(
        run_number=3,
        input_file=input_file,
    )
    assert "random_seed" in run_command
    assert "file-by-run" in run_command
    assert "test_file_with_seeds.txt" in run_command

    parts = run_command.split()
    c_args = [parts[i + 1] for i, p in enumerate(parts) if p == "-C"]
    # assert last one is show=all
    assert c_args[-1] == "show=all"


def test_make_run_command_with_calibration_config(simtel_runner):
    """Test make_run_command when calibration_config is set."""
    input_file = "test_calibration.inp"
    simtel_runner.calibration_config = {
        "run_mode": "pedestals",
        "number_of_events": 100,
    }

    run_command = simtel_runner.make_run_command(run_number=5, input_file=input_file)

    assert "sim_telarray" in run_command
    assert "-C pedestal_events=100" in run_command
    assert input_file in run_command

    parts = run_command.split()
    c_args = [parts[i + 1] for i, p in enumerate(parts) if p == "-C"]
    # assert last one is show=all
    assert c_args[-1] == "show=all"


def test_make_run_command_divergent(simtel_runner):
    input_file = "test_make_run_command_divergent.inp"
    run_command = simtel_runner.make_run_command(
        run_number=3,
        input_file=input_file,
        weak_pointing=True,
    )
    assert "-W telescope_theta=20" in run_command
    assert "-W telescope_phi=0" in run_command


def test_check_run_result(simtel_runner):
    expected_pattern = r"sim_telarray output file .+ does not exist\."
    with pytest.raises(InvalidOutputFileError, match=expected_pattern):
        simtel_runner._check_run_result(run_number=3)


def test_get_power_law_for_sim_telarray_histograms():
    from simtools.corsika.primary_particle import PrimaryParticle

    gamma = PrimaryParticle(particle_id="gamma", particle_id_type="common_name")
    electron = PrimaryParticle(particle_id="electron", particle_id_type="common_name")
    proton = PrimaryParticle(particle_id="proton", particle_id_type="common_name")
    helium = PrimaryParticle(particle_id="helium", particle_id_type="common_name")
    assert SimulatorArray.get_power_law_for_sim_telarray_histograms(gamma) == pytest.approx(2.5)
    assert SimulatorArray.get_power_law_for_sim_telarray_histograms(electron) == pytest.approx(3.3)
    assert SimulatorArray.get_power_law_for_sim_telarray_histograms(proton) == pytest.approx(2.68)
    assert SimulatorArray.get_power_law_for_sim_telarray_histograms(helium) == pytest.approx(2.68)


def test_check_run_result_file_exists(simtel_runner, tmp_path):
    output_file = tmp_path / "test_output.zst"
    output_file.touch()
    simtel_runner.get_file_name = lambda file_type, run_number: output_file
    assert simtel_runner._check_run_result(run_number=1) is True


def test_pedestals_nsb_only_command(simtel_runner):
    command = simtel_runner._pedestals_nsb_only_command()
    assert "-C fadc_err_pedestal=0.0" in command
    assert "-C fadc_lg_err_pedestal=-1.0" in command


def test_make_run_command_for_calibration_simulations(simtel_runner):
    calibration_config = {
        "nsb_scaling_factor": 1.5,
        "stars": "stars.txt",
        "run_mode": "pedestals",
        "number_of_events": 100,
    }
    simtel_runner.calibration_config = calibration_config

    run_command = simtel_runner._make_run_command_for_calibration_simulations()
    assert "-C nsb_scaling_factor=1.5" in run_command
    assert "-C stars=stars.txt" in run_command
    assert "-C pedestal_events=100" in run_command

    simtel_runner.calibration_config["run_mode"] = "pedestals_nsb_only"
    run_command = simtel_runner._make_run_command_for_calibration_simulations()
    assert "-C fadc_err_pedestal=0.0" in run_command  # From _pedestals_nsb_only_command


def test_make_run_command_for_calibration_simulations_additional_modes(simtel_runner):
    """Test additional run modes for calibration simulations."""

    # Test pedestals_dark mode
    simtel_runner.calibration_config = {
        "run_mode": "pedestals_dark",
        "number_of_events": 50,
        "number_of_dark_events": 75,
    }
    run_command = simtel_runner._make_run_command_for_calibration_simulations()
    assert "-C dark_events=75" in run_command

    # Test direct_injection mode
    simtel_runner.calibration_config = {
        "run_mode": "direct_injection",
        "number_of_events": 50,
        "number_of_flasher_events": 200,
    }
    run_command = simtel_runner._make_run_command_for_calibration_simulations()
    assert "-C laser_events=200" in run_command


def test_prepare_run_script(simtel_runner, tmp_path):
    """Test prepare_run_script generates correct bash script."""
    input_file = tmp_path / "test_input.corsika"
    input_file.touch()

    script_path = simtel_runner.prepare_run_script(
        test=False, input_file=str(input_file), run_number=1
    )

    assert script_path.exists()
    assert script_path.stat().st_mode & stat.S_IXUSR

    content = script_path.read_text()
    assert "#!/usr/bin/env bash" in content
    assert "set -e" in content
    assert "set -o pipefail" in content
    assert "SECONDS=0" in content
    assert "RUNTIME: $SECONDS" in content
    assert "sim_telarray" in content


def test_prepare_run_script_test_mode(simtel_runner, tmp_path):
    """Test prepare_run_script in test mode generates single run."""
    input_file = tmp_path / "test_input.corsika"
    input_file.touch()

    script_path = simtel_runner.prepare_run_script(
        test=True, input_file=str(input_file), run_number=1
    )

    content = script_path.read_text()
    # Count unique command lines containing sim_telarray
    sim_tel_lines = [
        line
        for line in content.splitlines()
        if line.strip().startswith("SIM_TELARRAY_CONFIG_PATH=''")
    ]
    assert len(sim_tel_lines) == 1


def test_prepare_run_script_with_extra_commands(simtel_runner, tmp_path):
    """Test prepare_run_script with extra commands."""
    input_file = tmp_path / "test_input.corsika"
    input_file.touch()

    extra_commands = ["export TEST_VAR=1", "echo 'Starting simulation'"]

    script_path = simtel_runner.prepare_run_script(
        test=True, input_file=str(input_file), run_number=1, extra_commands=extra_commands
    )

    content = script_path.read_text()
    assert "# Writing extras" in content
    assert "export TEST_VAR=1" in content
    assert "echo 'Starting simulation'" in content
    assert "# End of extras" in content


def test_prepare_run_script_multiple_runs(simtel_runner, tmp_path):
    """Test prepare_run_script generates multiple run commands."""
    input_file = tmp_path / "test_input.corsika"
    input_file.touch()

    simtel_runner.runs_per_set = 3

    script_path = simtel_runner.prepare_run_script(
        test=False, input_file=str(input_file), run_number=1
    )

    content = script_path.read_text()
    # Count unique command lines containing sim_telarray
    sim_tel_lines = [
        line
        for line in content.splitlines()
        if line.strip().startswith("SIM_TELARRAY_CONFIG_PATH=''")
    ]
    assert len(sim_tel_lines) == 3
