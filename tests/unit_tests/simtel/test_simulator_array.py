#!/usr/bin/python3

import logging
from copy import deepcopy

import astropy.units as u
import pytest

from simtools.simtel.simulator_array import SimulatorArray
from simtools.simulation.nsb import NsbSettings

logger = logging.getLogger()


@pytest.fixture
def simtel_runner(corsika_config_mock_array_model, run_context):
    return SimulatorArray(
        simulation_parameters=corsika_config_mock_array_model.simulation_parameters,
        array_model=corsika_config_mock_array_model.array_model,
        label="test-simtel-runner",
        context=run_context,
    )


def _set_context_config(simulator, config):
    """Set the simulator context from a test configuration object."""
    simulator.context = simulator.context.replace(args=getattr(config, "args", {}), software=config)


def test_init_simulator_array(corsika_config_mock_array_model, run_context):
    simulator = SimulatorArray(
        simulation_parameters=corsika_config_mock_array_model.simulation_parameters,
        array_model=corsika_config_mock_array_model.array_model,
        label="test-label",
        context=run_context,
    )
    assert simulator.array_model == corsika_config_mock_array_model.array_model
    assert simulator.label == "test-label"
    assert simulator._log_file is None


def test_make_run_command_without_bypass_option_is_unchanged(simtel_runner, mocker):
    simtel_runner.runner_service = mocker.Mock()
    simtel_runner.runner_service.load_files.return_value = {}
    simtel_runner.simulation_parameters.is_calibration_run = mocker.Mock(return_value=False)
    mocker.patch.object(
        simtel_runner, "_common_run_command", side_effect=lambda _run_number: ["common_command"]
    )
    mocker.patch.object(
        simtel_runner,
        "_make_run_command_for_shower_simulations",
        return_value=["-C", "power_law=2.5"],
    )

    legacy = simtel_runner.make_run_command(run_number=1, input_file="-")

    assert legacy == simtel_runner.make_run_command(
        run_number=1, input_file="-", bypass_optics=None
    )
    assert legacy == ["common_command", "-C", "power_law=2.5", "-C", "show=all", "-"]


def test_make_run_command_sets_bypass_optics_before_show_all(simtel_runner, mocker):
    simtel_runner.runner_service = mocker.Mock()
    simtel_runner.runner_service.load_files.return_value = {}
    simtel_runner.simulation_parameters.is_calibration_run = mocker.Mock(return_value=False)
    mocker.patch.object(simtel_runner, "_common_run_command", return_value=["common_command"])
    mocker.patch.object(
        simtel_runner,
        "_make_run_command_for_shower_simulations",
        return_value=["-C", "power_law=2.5"],
    )

    command = simtel_runner.make_run_command(run_number=1, input_file="-", bypass_optics=1)

    assert command == [
        "common_command",
        "-C",
        "power_law=2.5",
        "-C",
        "Bypass_Optics=1",
        "-C",
        "show=all",
        "-",
    ]


@pytest.mark.parametrize("write_hdata", [False, True])
def test_make_run_command_calibration_simulation(simtel_runner, mocker, write_hdata):
    simtel_runner.context = simtel_runner.context.replace(args={"write_hdata": write_hdata})
    simtel_runner.runner_service = mocker.Mock()
    simtel_runner.runner_service.load_files.return_value = {
        "sim_telarray_histogram": "unused.hdata"
    }
    simtel_runner.simulation_parameters.is_calibration_run = mocker.Mock(return_value=True)

    # Mock the methods
    mocker.patch.object(simtel_runner, "_common_run_command", return_value=["common_command"])
    mocker.patch.object(
        simtel_runner, "_make_run_command_for_calibration_simulations", return_value=["calib_opts"]
    )

    result = simtel_runner.make_run_command(run_number=1, input_file="test.corsika")

    # Verify that the result contains expected components
    assert isinstance(result, list)
    assert "common_command" in result
    assert "calib_opts" in result
    assert "-C" in result
    assert "show=all" in result
    assert "test.corsika" in result
    assert ("sim_telarray_histogram" in simtel_runner.file_list) is write_hdata


def test_make_run_command_for_calibration_simulations_basic(simtel_runner, mocker):
    # Supply run-specific arguments through the runner context.
    mock_config = mocker.Mock()
    mock_config.args = {"run_mode": "pedestals", "number_of_events": 1000}
    _set_context_config(simtel_runner, mock_config)

    simtel_runner.array_model.site_model.get_parameter_value_with_unit.side_effect = {
        "reference_point_altitude": 2177 * u.m,
        "corsika_observation_level": 2156 * u.m,
    }.get

    result = simtel_runner._make_run_command_for_calibration_simulations()
    simtel_runner.array_model.site_model.get_parameter_value_with_unit.side_effect = None

    # Should return a string with configuration options
    assert isinstance(result, list)
    assert "Altitude=2156.0" in result


def test_make_run_command_for_calibration_direct_injection(simtel_runner, mocker):
    mock_config = mocker.Mock()
    mock_config.args = {"run_mode": "direct_injection", "number_of_events": 1000}
    _set_context_config(simtel_runner, mock_config)

    mock_param = mocker.Mock()
    mock_param.to_value.return_value = 1800.0
    simtel_runner.array_model.site_model.get_parameter_value_with_unit.return_value = mock_param

    result = simtel_runner._make_run_command_for_calibration_simulations()

    assert isinstance(result, list)
    assert any("laser_events" in str(item) for item in result)


def test_pedestals_nsb_only_command_basic(simtel_runner):
    result = simtel_runner._pedestals_nsb_only_options()

    # Should return a dictionary with noise parameter configurations
    assert isinstance(result, dict)
    assert "fadc_noise" in result


def test_check_run_result_success(simtel_runner, mocker, tmp_path):
    # Create a mock output file
    output_file = tmp_path / "output.simtel.gz"
    output_file.touch()

    # Mock runner_service
    simtel_runner.runner_service = mocker.Mock()
    simtel_runner.runner_service.get_file_name.return_value = output_file

    result = simtel_runner._check_run_result(run_number=1)
    assert result is True
    simtel_runner.runner_service.get_file_name.assert_called_once_with(
        file_type="sim_telarray_output", run_number=1
    )


def test_check_run_result_file_not_exists(simtel_runner, mocker, tmp_path):
    output_file = tmp_path / "nonexistent.simtel.gz"

    simtel_runner.runner_service = mocker.Mock()
    simtel_runner.runner_service.get_file_name.return_value = output_file

    with pytest.raises(FileNotFoundError, match=r"sim_telarray output file .* does not exist"):
        simtel_runner._check_run_result(run_number=1)


def test_get_power_law_for_sim_telarray_histograms_gamma():
    # Mock primary particle
    mock_primary = type("MockPrimary", (), {"name": "gamma"})()

    result = SimulatorArray.get_power_law_for_sim_telarray_histograms(mock_primary)
    assert result == pytest.approx(2.5)


def test_determine_pointing_option(simtel_runner):
    copy_simtel_runner = deepcopy(simtel_runner)
    assert copy_simtel_runner._determine_pointing_option() is False

    copy_simtel_runner.label = "divergent"
    assert copy_simtel_runner._determine_pointing_option() is True
    assert copy_simtel_runner._determine_pointing_option() is True
    copy_simtel_runner.label = "test"
    assert copy_simtel_runner._determine_pointing_option() is False
    copy_simtel_runner.label = None
    assert copy_simtel_runner._determine_pointing_option() is False


@pytest.mark.parametrize("write_hdata", [False, True])
def test_common_run_command_weak_pointing_divergent(simtel_runner, mocker, write_hdata):
    _set_context_config(simtel_runner, mocker.Mock(sim_telarray_exe="/path/to/sim_telarray"))

    simtel_runner.context = simtel_runner.context.replace(args={"write_hdata": write_hdata})
    simtel_runner.label = "divergent"
    simtel_runner.array_model.get_config_directory.return_value = "/config/dir"
    simtel_runner.array_model.config_file_path = "/config/file.cfg"
    simtel_runner.array_model.export_config_files = mocker.Mock()
    simtel_runner.simulation_parameters.zenith_angle = 25.0
    simtel_runner.simulation_parameters.azimuth_angle = 180.0
    mocker.patch(
        "simtools.simtel.simulator_array.get_model_writer",
        return_value=mocker.Mock(
            telescope_model_files_exported=False, array_model_file_exported=False, nsb_settings=None
        ),
    )

    simtel_runner.runner_service = mocker.Mock()
    simtel_runner.runner_service.get_file_name.side_effect = lambda file_type, run_number: {
        "sim_telarray_log": f"log_{run_number}.log",
        "sim_telarray_histogram": f"hist_{run_number}.hist",
        "sim_telarray_output": f"output_{run_number}.simtel.gz",
    }[file_type]

    result = simtel_runner._common_run_command(run_number=1)

    assert isinstance(result, list)
    expected = "hist_1.hist" if write_hdata else "/dev/null"
    assert f"histogram_file={expected}" in result
    assert "-W" in result
    assert any("telescope_theta=25.0" in str(item) for item in result)


def test_make_run_command_for_calibration_simulations_pedestals(simtel_runner, mocker):
    mock_config = mocker.Mock()
    mock_config.args = {
        "run_mode": "pedestals",
        "number_of_events": 1000,
        "number_of_pedestal_events": 500,
    }
    _set_context_config(simtel_runner, mock_config)

    mock_param = mocker.Mock()
    mock_param.to_value.return_value = 1800.0
    simtel_runner.array_model.site_model.get_parameter_value_with_unit.return_value = mock_param

    result = simtel_runner._make_run_command_for_calibration_simulations()

    assert isinstance(result, list)
    assert "-C" in result
    assert "Altitude=2200.0" in result
    assert "pedestal_events=500" in result


def test_make_run_command_for_calibration_simulations_pedestals_nsb_only(simtel_runner, mocker):
    mock_config = mocker.Mock()
    mock_config.args = {
        "run_mode": "pedestals_nsb_only",
        "number_of_events": 1000,
        "number_of_pedestal_events": 800,
    }
    _set_context_config(simtel_runner, mock_config)

    mock_param = mocker.Mock()
    mock_param.to_value.return_value = 1800.0
    simtel_runner.array_model.site_model.get_parameter_value_with_unit.return_value = mock_param

    result = simtel_runner._make_run_command_for_calibration_simulations()

    assert isinstance(result, list)
    assert "pedestal_events=800" in result
    assert any("fadc_noise=0.0" in str(item) for item in result)
    assert any("gain_variation=0.0" in str(item) for item in result)


def test_make_run_command_for_calibration_simulations_pedestals_dark(simtel_runner, mocker):
    mock_config = mocker.Mock()
    mock_config.args = {
        "run_mode": "pedestals_dark",
        "number_of_events": 1000,
        "number_of_dark_events": 600,
    }
    _set_context_config(simtel_runner, mock_config)

    mock_param = mocker.Mock()
    mock_param.to_value.return_value = 1800.0
    simtel_runner.array_model.site_model.get_parameter_value_with_unit.return_value = mock_param

    result = simtel_runner._make_run_command_for_calibration_simulations()

    assert isinstance(result, list)
    assert "dark_events=600" in result


def test_make_run_command_for_calibration_simulations_with_nsb_scaling(simtel_runner, mocker):
    mock_config = mocker.Mock()
    mock_config.args = {
        "run_mode": "pedestals",
        "number_of_events": 1000,
        "nsb_scaling_factor": 1.5,
    }
    _set_context_config(simtel_runner, mock_config)

    mock_param = mocker.Mock()
    mock_param.to_value.return_value = 1800.0
    simtel_runner.array_model.site_model.get_parameter_value_with_unit.return_value = mock_param

    result = simtel_runner._make_run_command_for_calibration_simulations()

    assert isinstance(result, list)
    assert "nsb_scaling_factor=1.5" not in result


def test_make_run_command_for_calibration_simulations_with_stars(simtel_runner, mocker):
    mock_config = mocker.Mock()
    mock_config.args = {
        "run_mode": "pedestals",
        "number_of_events": 1000,
        "stars": 0,
    }
    _set_context_config(simtel_runner, mock_config)

    mock_param = mocker.Mock()
    mock_param.to_value.return_value = 1800.0
    simtel_runner.array_model.site_model.get_parameter_value_with_unit.return_value = mock_param

    result = simtel_runner._make_run_command_for_calibration_simulations()

    assert isinstance(result, list)
    assert "stars=0" in result


def test_common_run_command_puts_include_directories_before_model_directory(simtel_runner, mocker):
    simtel_runner.runner_service = mocker.Mock()
    simtel_runner.runner_service.get_file_name.return_value = "file"
    simtel_runner.runner_service.load_files.return_value = {}
    mocker.patch(
        "simtools.simtel.simulator_array.get_model_writer",
        return_value=mocker.Mock(
            telescope_model_files_exported=False, array_model_file_exported=False, nsb_settings=None
        ),
    )
    simtel_runner.include_directories = ("/scaled/qe-table",)
    mocker.patch.object(simtel_runner, "_determine_pointing_option", return_value=False)

    command = simtel_runner._common_run_command(run_number=1)

    config_dir = str(simtel_runner.array_model.get_config_directory())
    assert command.index("-I/scaled/qe-table") < command.index(f"-I{config_dir}")


def test_include_directories_default_is_empty(simtel_runner):
    assert simtel_runner.include_directories == ()


def test_make_run_command_sets_min_photons_when_overridden(simtel_runner, mocker):
    simtel_runner.runner_service = mocker.Mock()
    simtel_runner.runner_service.load_files.return_value = {}
    mocker.patch.object(simtel_runner, "_common_run_command", return_value=["sim_telarray"])
    mocker.patch.object(simtel_runner, "_make_run_command_for_shower_simulations", return_value=[])
    mocker.patch.object(
        simtel_runner.simulation_parameters, "is_calibration_run", return_value=False
    )

    simtel_runner.min_photons = 0
    command = simtel_runner.make_run_command(run_number=1, input_file="in.dat", bypass_optics=3)
    assert command[command.index("min_photons=0") - 1] == "-C"
    assert command[-1] == "in.dat"

    simtel_runner.min_photons = None
    command = simtel_runner.make_run_command(run_number=1, input_file="in.dat")
    assert not any("min_photons" in item for item in command)


@pytest.mark.parametrize("bypass_optics", [None, 1, 3])
def test_nsb_settings_use_pointing_zenith_with_bypassed_optics(
    simtel_runner, mocker, bypass_optics
):
    """NSB scaling follows the pointing zenith angle, also when optics are bypassed."""
    simtel_runner.runner_service = mocker.Mock()
    simtel_runner.runner_service.load_files.return_value = {}
    writer = mocker.Mock(
        telescope_model_files_exported=False, array_model_file_exported=False, nsb_settings=None
    )
    mocker.patch("simtools.simtel.simulator_array.get_model_writer", return_value=writer)
    mocker.patch.object(simtel_runner, "_determine_pointing_option", return_value=False)
    mocker.patch.object(simtel_runner, "_make_run_command_for_shower_simulations", return_value=[])
    mocker.patch.object(
        simtel_runner.simulation_parameters, "is_calibration_run", return_value=False
    )

    simtel_runner.make_run_command(run_number=1, input_file="in.dat", bypass_optics=bypass_optics)

    assert writer.nsb_settings.zenith_angle == simtel_runner.simulation_parameters.zenith_angle


def test_nsb_settings_rejected_after_telescope_config_export(simtel_runner, mocker):
    """Changed NSB settings must not be applied after telescope configuration files exist."""
    simtel_runner.runner_service = mocker.Mock()
    simtel_runner.runner_service.load_files.return_value = {}
    writer = mocker.Mock(
        telescope_model_files_exported=True, array_model_file_exported=False, nsb_settings=None
    )
    mocker.patch("simtools.simtel.simulator_array.get_model_writer", return_value=writer)
    mocker.patch.object(simtel_runner, "_determine_pointing_option", return_value=False)
    mocker.patch.object(simtel_runner, "_make_run_command_for_shower_simulations", return_value=[])
    mocker.patch.object(
        simtel_runner.simulation_parameters, "is_calibration_run", return_value=False
    )

    with pytest.raises(ValueError, match="NSB settings changed after configuration export"):
        simtel_runner.make_run_command(run_number=1, input_file="in.dat")


def test_nsb_settings_unchanged_after_export_do_not_raise(simtel_runner, mocker):
    """Repeating the NSB settings used for an earlier export is accepted."""
    simtel_runner.runner_service = mocker.Mock()
    simtel_runner.runner_service.load_files.return_value = {}
    simtel_runner.context = simtel_runner.context.replace(args={})
    writer = mocker.Mock(telescope_model_files_exported=True, array_model_file_exported=True)
    writer.nsb_settings = NsbSettings(
        zenith_angle=simtel_runner.simulation_parameters.zenith_angle,
        manual_factor=None,
        autoscale=True,
    )
    mocker.patch("simtools.simtel.simulator_array.get_model_writer", return_value=writer)
    mocker.patch.object(simtel_runner, "_determine_pointing_option", return_value=False)
    mocker.patch.object(simtel_runner, "_make_run_command_for_shower_simulations", return_value=[])
    mocker.patch.object(
        simtel_runner.simulation_parameters, "is_calibration_run", return_value=False
    )

    simtel_runner.make_run_command(run_number=1, input_file="in.dat")

    writer.export_config_files.assert_called_once()
