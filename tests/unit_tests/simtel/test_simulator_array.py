#!/usr/bin/python3

import logging
import stat

import pytest

from simtools.runners.simtel_runner import InvalidOutputFileError
from simtools.simtel.simulator_array import SimulatorArray

logger = logging.getLogger()


@pytest.fixture
def simtel_runner(corsika_config_mock_array_model):
    return SimulatorArray(
        corsika_config=corsika_config_mock_array_model,
        label="test-simtel-runner",
    )


def test_init_simulator_array(corsika_config_mock_array_model):
    """Test SimulatorArray initialization."""
    simulator = SimulatorArray(
        corsika_config=corsika_config_mock_array_model,
        label="test-label",
    )
    assert simulator.corsika_config == corsika_config_mock_array_model
    assert simulator.label == "test-label"
    assert simulator.sim_telarray_seeds is None
    assert simulator.is_calibration_run is False
    assert simulator._log_file is None


def test_init_simulator_array_with_seeds(corsika_config_mock_array_model):
    """Test SimulatorArray initialization with sim_telarray_seeds."""
    seeds = {"seed": 12345, "random_instrument_instances": True}
    simulator = SimulatorArray(
        corsika_config=corsika_config_mock_array_model,
        label="test-label",
        sim_telarray_seeds=seeds,
        is_calibration_run=True,
    )
    assert simulator.sim_telarray_seeds == seeds
    assert simulator.is_calibration_run is True


def test_prepare_run(simtel_runner, tmp_path, mocker):
    """Test prepare_run method creates script with correct content."""
    # Mock make_run_command
    mocker.patch.object(simtel_runner, "make_run_command", return_value="echo 'test command'")

    # Set up test data
    run_number = 42
    sub_script = tmp_path / "test_script.sh"
    corsika_file = "/path/to/corsika.file"
    extra_commands = ["export TEST_VAR=1", "echo 'extra command'"]

    # Execute prepare_run
    simtel_runner.prepare_run(
        run_number=run_number,
        sub_script=sub_script,
        corsika_file=corsika_file,
        extra_commands=extra_commands,
    )

    # Verify script was created and is executable
    assert sub_script.exists()
    assert sub_script.stat().st_mode & stat.S_IXUSR

    # Check script content
    content = sub_script.read_text()
    assert "#!/usr/bin/env bash" in content
    assert "set -e" in content
    assert "set -o pipefail" in content
    assert "export TEST_VAR=1" in content
    assert "echo 'extra command'" in content
    assert "echo 'test command'" in content
    assert 'echo "RUNTIME: $SECONDS"' in content


def test_prepare_run_no_extra_commands(simtel_runner, tmp_path, mocker):
    """Test prepare_run method without extra commands."""
    mocker.patch.object(simtel_runner, "make_run_command", return_value="sim_telarray command")

    sub_script = tmp_path / "simple_script.sh"
    simtel_runner.prepare_run(run_number=1, sub_script=sub_script, corsika_file="test.corsika")

    content = sub_script.read_text()
    assert "# Writing extras" not in content
    assert "sim_telarray command" in content


def test_make_run_command_shower_simulation(simtel_runner, mocker):
    """Test make_run_command for shower simulations."""
    # Mock runner_service
    simtel_runner.runner_service = mocker.Mock()
    simtel_runner.runner_service.load_files.return_value = {}

    # Mock the method calls we know will be made
    mocker.patch.object(simtel_runner, "_common_run_command", return_value="common_command")
    mocker.patch.object(
        simtel_runner, "_make_run_command_for_shower_simulations", return_value=" shower_opts"
    )
    mocker.patch(
        "simtools.utils.general.clear_default_sim_telarray_cfg_directories",
        return_value="cleared_command",
    )

    result = simtel_runner.make_run_command(run_number=1, corsika_input_file="test.corsika")

    # Just verify that the result contains expected components
    assert isinstance(result, str)
    assert "test.corsika" in result


def test_make_run_command_calibration_simulation(simtel_runner, mocker):
    """Test make_run_command for calibration simulations."""
    simtel_runner.is_calibration_run = True
    simtel_runner.runner_service = mocker.Mock()
    simtel_runner.runner_service.load_files.return_value = {}

    # Mock the methods
    mocker.patch.object(simtel_runner, "_common_run_command", return_value="common_command")
    mocker.patch.object(
        simtel_runner, "_make_run_command_for_calibration_simulations", return_value=" calib_opts"
    )
    mocker.patch(
        "simtools.utils.general.clear_default_sim_telarray_cfg_directories",
        return_value="cleared_command",
    )

    result = simtel_runner.make_run_command(run_number=1, corsika_input_file="test.corsika")

    assert isinstance(result, str)
    assert "test.corsika" in result


def test_make_run_command_for_shower_simulations(simtel_runner, mocker):
    """Test _make_run_command_for_shower_simulations method."""
    # Mock get_power_law_for_sim_telarray_histograms
    mock_power_law = mocker.patch.object(
        SimulatorArray, "get_power_law_for_sim_telarray_histograms", return_value=2.5
    )

    result = simtel_runner._make_run_command_for_shower_simulations()

    # Verify the static method was called
    mock_power_law.assert_called_once_with(simtel_runner.corsika_config.primary_particle)

    # The result should contain the power law configuration
    assert isinstance(result, str)


def test_make_run_command_for_calibration_simulations_basic(simtel_runner, mocker):
    """Test _make_run_command_for_calibration_simulations basic functionality."""
    # Mock settings.config.args
    mock_config = mocker.Mock()
    mock_config.args = {"run_mode": "pedestals", "number_of_events": 1000}
    mocker.patch("simtools.settings.config", mock_config)

    # Mock site model parameter
    mock_param = mocker.Mock()
    mock_param.to_value.return_value = 1800.0
    simtel_runner.corsika_config.array_model.site_model.get_parameter_value_with_unit.return_value = mock_param

    result = simtel_runner._make_run_command_for_calibration_simulations()

    # Should return a string with configuration options
    assert isinstance(result, str)


def test_make_run_command_for_calibration_direct_injection(simtel_runner, mocker):
    """Test _make_run_command_for_calibration_simulations with direct injection."""
    mock_config = mocker.Mock()
    mock_config.args = {"run_mode": "direct_injection", "number_of_events": 1000}
    mocker.patch("simtools.settings.config", mock_config)

    mock_param = mocker.Mock()
    mock_param.to_value.return_value = 1800.0
    simtel_runner.corsika_config.array_model.site_model.get_parameter_value_with_unit.return_value = mock_param

    result = simtel_runner._make_run_command_for_calibration_simulations()

    assert isinstance(result, str)


def test_common_run_command_basic(simtel_runner, mocker):
    """Test _common_run_command method basic functionality."""
    # Mock settings
    mocker.patch("simtools.settings.config", mocker.Mock(sim_telarray_exe="/path/to/sim_telarray"))

    # Mock corsika_config methods
    simtel_runner.corsika_config.array_model.get_config_directory.return_value = "/config/dir"
    simtel_runner.corsika_config.array_model.config_file_path = "/config/file.cfg"
    simtel_runner.corsika_config.array_model.export_all_simtel_config_files = mocker.Mock()
    simtel_runner.corsika_config.zenith_angle = 20.0
    simtel_runner.corsika_config.azimuth_angle = 0.0

    # Mock runner_service
    simtel_runner.runner_service = mocker.Mock()
    simtel_runner.runner_service.get_file_name.side_effect = lambda file_type, run_number: {
        "sim_telarray_log": f"log_{run_number}.log",
        "sim_telarray_histogram": f"hist_{run_number}.hist",
        "sim_telarray_output": f"output_{run_number}.simtel.gz",
    }[file_type]

    result = simtel_runner._common_run_command(run_number=42)

    # Should create basic command structure
    assert isinstance(result, str)
    assert "/path/to/sim_telarray" in result
    simtel_runner.corsika_config.array_model.export_all_simtel_config_files.assert_called_once()


def test_pedestals_nsb_only_command_basic(simtel_runner):
    """Test _pedestals_nsb_only_command method returns string."""
    result = simtel_runner._pedestals_nsb_only_command()

    # Should return a string with noise parameter configurations
    assert isinstance(result, str)
    assert "fadc_noise" in result


def test_check_run_result_success(simtel_runner, mocker, tmp_path):
    """Test _check_run_result when output file exists."""
    # Create a mock output file
    output_file = tmp_path / "output.simtel.gz"
    output_file.touch()

    # Mock runner_service
    simtel_runner.runner_service = mocker.Mock()
    simtel_runner.runner_service.get_file_name.return_value = output_file

    result = simtel_runner._check_run_result(run_number=1)
    assert result is True


def test_check_run_result_file_not_exists(simtel_runner, mocker, tmp_path):
    """Test _check_run_result when output file doesn't exist."""
    output_file = tmp_path / "nonexistent.simtel.gz"

    simtel_runner.runner_service = mocker.Mock()
    simtel_runner.runner_service.get_file_name.return_value = output_file

    with pytest.raises(InvalidOutputFileError, match=r"sim_telarray output file .* does not exist"):
        simtel_runner._check_run_result(run_number=1)


def test_get_power_law_for_sim_telarray_histograms_gamma():
    """Test get_power_law_for_sim_telarray_histograms for gamma particles."""
    # Mock primary particle
    mock_primary = type("MockPrimary", (), {"name": "gamma"})()

    result = SimulatorArray.get_power_law_for_sim_telarray_histograms(mock_primary)
    assert result == 2.5


def test_get_power_law_for_sim_telarray_histograms_unknown():
    """Test get_power_law_for_sim_telarray_histograms for unknown particles."""
    mock_primary = type("MockPrimary", (), {"name": "proton"})()

    result = SimulatorArray.get_power_law_for_sim_telarray_histograms(mock_primary)
    assert result == 2.68
