#!/usr/bin/python3

from pathlib import Path
from unittest.mock import Mock, patch

import astropy.units as u
import numpy as np
import pytest

from simtools.simtel.simulator_light_emission import SimulatorLightEmission


@pytest.fixture
def simulator_instance():
    """Create a fresh mock SimulatorLightEmission instance for each test."""
    inst = object.__new__(SimulatorLightEmission)
    # Create fresh mocks for each test to avoid cross-test contamination
    inst.calibration_model = Mock()
    inst.telescope_model = Mock()
    inst.site_model = Mock()
    inst.light_emission_config = {}
    inst.output_directory = "/test/output"
    inst._logger = Mock()
    return inst


def test_get_prefix_non_none_returns_with_underscore(simulator_instance):
    simulator_instance.light_emission_config = {"output_prefix": "pre", "number_of_events": 1}
    assert simulator_instance._get_prefix() == "pre_"


@patch("simtools.simtel.simulator_light_emission.clear_default_sim_telarray_cfg_directories")
@patch.object(SimulatorLightEmission, "_get_telescope_pointing")
@patch.object(SimulatorLightEmission, "_get_light_emission_application_name")
@patch.object(SimulatorLightEmission, "_get_prefix")
def test__make_simtel_script(
    mock_prefix, mock_app_name, mock_pointing, mock_clear_cfg, simulator_instance
):
    """Test _make_simtel_script method with different conditions."""
    simulator_instance.telescope_model.config_file_directory = "/mock/config"
    simulator_instance.telescope_model.config_file_path = "/mock/config/telescope.cfg"

    mock_altitude = Mock()
    mock_altitude.to.return_value.value = 2200
    simulator_instance.site_model.get_parameter_value_with_unit.return_value = mock_altitude
    simulator_instance.site_model.get_parameter_value.return_value = "atm_trans.dat"

    mock_pointing.return_value = [10.5, 20.0]
    mock_app_name.return_value = "test-app"
    mock_prefix.return_value = "test_"
    mock_clear_cfg.return_value = "mocked_command_string"

    def mock_get_config_option(key, value):
        return f"-C{key}={value}"

    with patch.object(
        simulator_instance.__class__.__bases__[0],
        "get_config_option",
        side_effect=mock_get_config_option,
    ):
        # Test 1: flat_fielding with no light_source_position
        simulator_instance.light_emission_config = {"light_source_type": "flat_fielding"}

        result = simulator_instance._make_simtel_script()

        assert result == "mocked_command_string"
        mock_clear_cfg.assert_called()
        mock_pointing.assert_called_once()

        # Reset mocks for next test
        mock_clear_cfg.reset_mock()
        mock_pointing.reset_mock()

        # Test 2: flat_fielding with light_source_position (should use fixed pointing)
        simulator_instance.light_emission_config = {
            "light_source_type": "flat_fielding",
            "light_source_position": [1.0, 2.0, 3.0],
        }

        result = simulator_instance._make_simtel_script()

        assert result == "mocked_command_string"
        mock_clear_cfg.assert_called()
        # Should still call _get_telescope_pointing
        mock_pointing.assert_called_once()

        # Reset for next test
        mock_clear_cfg.reset_mock()
        mock_pointing.reset_mock()

        # Test 3: illuminator (not flat_fielding)
        simulator_instance.light_emission_config = {"light_source_type": "illuminator"}

        result = simulator_instance._make_simtel_script()

        assert result == "mocked_command_string"
        mock_clear_cfg.assert_called()
        mock_pointing.assert_called_once()


def test__make_simtel_script_bypass_optics_condition(simulator_instance):
    """Test that flat_fielding adds Bypass_Optics option."""
    # Setup minimal mocks
    simulator_instance.telescope_model.config_file_directory = "/mock/config"
    simulator_instance.telescope_model.config_file_path = "/mock/config/telescope.cfg"

    mock_altitude = Mock()
    mock_altitude.to.return_value.value = 2200
    simulator_instance.site_model.get_parameter_value_with_unit.return_value = mock_altitude
    simulator_instance.site_model.get_parameter_value.return_value = "atm_trans.dat"

    def mock_get_config_option(key, value):
        return f"-C{key}={value}"

    # Mock the helper methods
    with (
        patch.object(simulator_instance, "_get_telescope_pointing", return_value=[0, 0]),
        patch.object(
            simulator_instance, "_get_light_emission_application_name", return_value="ff-1m"
        ),
        patch.object(simulator_instance, "_get_prefix", return_value=""),
        patch(
            "simtools.simtel.simulator_light_emission.clear_default_sim_telarray_cfg_directories"
        ) as mock_clear,
        patch.object(
            simulator_instance.__class__.__bases__[0],
            "get_config_option",
            side_effect=mock_get_config_option,
        ) as mock_config,
    ):
        mock_clear.return_value = "final_command"

        # Test flat_fielding - should include Bypass_Optics
        simulator_instance.light_emission_config = {"light_source_type": "flat_fielding"}

        simulator_instance._make_simtel_script()

        # Verify Bypass_Optics was called for flat_fielding
        bypass_calls = [
            call for call in mock_config.call_args_list if call[0][0] == "Bypass_Optics"
        ]
        assert len(bypass_calls) == 1
        assert bypass_calls[0][0] == ("Bypass_Optics", "1")

        # Reset for next test
        mock_config.reset_mock()

        # Test illuminator - should NOT include Bypass_Optics
        simulator_instance.light_emission_config = {"light_source_type": "illuminator"}

        simulator_instance._make_simtel_script()

        # Verify Bypass_Optics was NOT called for illuminator
        bypass_calls = [
            call for call in mock_config.call_args_list if call[0][0] == "Bypass_Optics"
        ]
        assert len(bypass_calls) == 0


def test__get_simulation_output_filename(simulator_instance):
    # Test with flat_fielding and prefix
    simulator_instance.light_emission_config = {
        "light_source_type": "flat_fielding",
        "output_prefix": "pre",
    }
    result = simulator_instance._get_simulation_output_filename()
    assert result == "/test/output/pre_ff-1m.simtel.zst"

    # Test with no prefix (should return empty string for prefix)
    simulator_instance.light_emission_config = {
        "light_source_type": "flat_fielding",
        "output_prefix": None,
    }
    result = simulator_instance._get_simulation_output_filename()
    assert result == "/test/output/ff-1m.simtel.zst"


def test_calculate_distance_focal_plane_calibration_device(simulator_instance):
    simulator_instance.telescope_model.get_parameter_value_with_unit.return_value = 10 * u.m
    simulator_instance.calibration_model.get_parameter_value_with_unit.return_value = [
        0.0 * u.cm,
        0.0 * u.cm,
        20.0 * u.cm,
    ]

    result = simulator_instance.calculate_distance_focal_plane_calibration_device()
    assert result.unit.is_equivalent(u.m)
    assert np.isclose(result.value, 9.8, rtol=1e-9)


def test__get_angular_distribution_string_for_sim_telarray(simulator_instance):
    # Test with width provided
    simulator_instance.calibration_model.get_parameter_value.return_value = "Gauss"
    mock_width = Mock()
    mock_width.to.return_value.value = 5.0
    simulator_instance.calibration_model.get_parameter_value_with_unit.return_value = mock_width

    result = simulator_instance._get_angular_distribution_string_for_sim_telarray()
    assert result == "gauss:5.0"

    simulator_instance.calibration_model.get_parameter_value.assert_called_once_with(
        "flasher_angular_distribution"
    )
    simulator_instance.calibration_model.get_parameter_value_with_unit.assert_called_once_with(
        "flasher_angular_distribution_width"
    )
    mock_width.to.assert_called_once_with(u.deg)

    # Reset mocks for second test
    simulator_instance.calibration_model.reset_mock()

    # Test with width None
    simulator_instance.calibration_model.get_parameter_value.return_value = "Gauss"
    simulator_instance.calibration_model.get_parameter_value_with_unit.return_value = None

    result = simulator_instance._get_angular_distribution_string_for_sim_telarray()
    assert result == "gauss"

    simulator_instance.calibration_model.get_parameter_value.assert_called_once_with(
        "flasher_angular_distribution"
    )
    simulator_instance.calibration_model.get_parameter_value_with_unit.assert_called_once_with(
        "flasher_angular_distribution_width"
    )


def test__get_pulse_shape_string_for_sim_telarray(simulator_instance):
    # Test with unified 3-element list [shape, width_ns, exp_ns]
    simulator_instance.calibration_model.get_parameter_value.return_value = ["Gauss", 5.0, 0.0]

    result = simulator_instance._get_pulse_shape_string_for_sim_telarray()
    assert result == "gauss:5.0"

    simulator_instance.calibration_model.get_parameter_value.assert_called_once_with(
        "flasher_pulse_shape"
    )

    # Reset mocks for second test
    simulator_instance.calibration_model.reset_mock()

    # Test with only shape provided is no longer supported; provide zeroed parameters
    simulator_instance.calibration_model.get_parameter_value.return_value = ["Line", 0.0, 0.0]

    result = simulator_instance._get_pulse_shape_string_for_sim_telarray()
    assert result == "line"

    simulator_instance.calibration_model.get_parameter_value.assert_called_once_with(
        "flasher_pulse_shape"
    )

    # Reset mocks for third test (exponential branch)
    simulator_instance.calibration_model.reset_mock()

    # Test exponential pulse shape with decay only
    simulator_instance.calibration_model.get_parameter_value.return_value = [
        "Exponential",
        0.0,
        3.2,
    ]

    result = simulator_instance._get_pulse_shape_string_for_sim_telarray()
    assert result == "exponential:3.2"

    simulator_instance.calibration_model.get_parameter_value.assert_called_once_with(
        "flasher_pulse_shape"
    )


def test__add_illuminator_command_options(simulator_instance):
    """Test _add_illuminator_command_options with different conditions."""
    # Mock calibration model methods
    mock_wavelength = Mock()
    mock_wavelength.to.return_value.value = 450
    simulator_instance.calibration_model.get_parameter_value_with_unit.side_effect = [
        [1.0 * u.m, 2.0 * u.m, 3.0 * u.m],  # array_element_position_ground
        mock_wavelength,  # flasher_wavelength
    ]

    # Mock helper methods
    with (
        patch.object(
            simulator_instance,
            "_get_angular_distribution_string_for_sim_telarray",
            return_value="gauss:5.0",
        ),
        patch.object(
            simulator_instance,
            "_get_pulse_shape_string_for_sim_telarray",
            return_value="square:2.0",
        ),
        patch.object(
            simulator_instance,
            "_calibration_pointing_direction",
            return_value=([0.1, 0.2, 0.3], []),
        ),
    ):
        # Test 1: No light_source_position, no light_source_pointing (uses defaults)
        simulator_instance.light_emission_config = {"flasher_photons": 1000000}

        result = simulator_instance._add_illuminator_command_options()

        # Verify structure and key values
        assert isinstance(result, list)
        assert len(result) == 8
        assert result[0] == "-x 100.0"  # 1.0m -> 100.0cm
        assert result[1] == "-y 200.0"  # 2.0m -> 200.0cm
        assert result[2] == "-z 300.0"  # 3.0m -> 300.0cm
        assert result[3] == "-d 0.1,0.2,0.3"  # pointing vector from _calibration_pointing_direction
        assert result[4] == "-n 1000000"  # flasher_photons
        assert result[5] == "-s 450"  # wavelength in nm
        assert result[6] == "-p square:2.0"  # pulse shape
        assert result[7] == "-a gauss:5.0"  # angular distribution


def test__add_illuminator_command_options_with_custom_position_and_pointing(simulator_instance):
    """Test _add_illuminator_command_options with custom position and pointing."""
    # Mock calibration model methods (only wavelength needed when position is provided)
    mock_wavelength = Mock()
    mock_wavelength.to.return_value.value = 380
    simulator_instance.calibration_model.get_parameter_value_with_unit.return_value = (
        mock_wavelength
    )

    # Mock helper methods
    with (
        patch.object(
            simulator_instance,
            "_get_angular_distribution_string_for_sim_telarray",
            return_value="uniform",
        ),
        patch.object(
            simulator_instance, "_get_pulse_shape_string_for_sim_telarray", return_value="gauss"
        ),
    ):
        # Test 2: Custom light_source_position and light_source_pointing
        simulator_instance.light_emission_config = {
            "flasher_photons": 2000000,
            "light_source_position": [5.0 * u.m, 6.0 * u.m, 7.0 * u.m],
            "light_source_pointing": [0.5, 0.6, 0.7],
        }

        result = simulator_instance._add_illuminator_command_options()

        # Verify custom values are used
        assert isinstance(result, list)
        assert len(result) == 8
        assert result[0] == "-x 500.0"  # 5.0m -> 500.0cm
        assert result[1] == "-y 600.0"  # 6.0m -> 600.0cm
        assert result[2] == "-z 700.0"  # 7.0m -> 700.0cm
        assert result[3] == "-d 0.5,0.6,0.7"  # custom pointing vector
        assert result[4] == "-n 2000000"  # custom flasher_photons
        assert result[5] == "-s 380"  # wavelength in nm
        assert result[6] == "-p gauss"  # pulse shape
        assert result[7] == "-a uniform"  # angular distribution


def test__add_illuminator_command_options_position_fallback(simulator_instance):
    """Test _add_illuminator_command_options position fallback behavior."""
    # Mock calibration model to return position when light_source_position is None
    mock_wavelength = Mock()
    mock_wavelength.to.return_value.value = 500

    def mock_get_param_with_unit(param_name):
        if param_name == "array_element_position_ground":
            return [10.0 * u.m, 20.0 * u.m, 30.0 * u.m]
        if param_name == "flasher_wavelength":
            return mock_wavelength
        return None

    simulator_instance.calibration_model.get_parameter_value_with_unit.side_effect = (
        mock_get_param_with_unit
    )

    # Mock helper methods
    with (
        patch.object(
            simulator_instance,
            "_get_angular_distribution_string_for_sim_telarray",
            return_value="test_angular",
        ),
        patch.object(
            simulator_instance,
            "_get_pulse_shape_string_for_sim_telarray",
            return_value="test_pulse",
        ),
        patch.object(
            simulator_instance,
            "_calibration_pointing_direction",
            return_value=([0.8, 0.9, 1.0], []),
        ) as mock_pointing,
    ):
        # Test 3: light_source_position is None, should use calibration_model position
        simulator_instance.light_emission_config = {
            "flasher_photons": 500000,
            "light_source_position": None,  # Explicitly None
        }

        result = simulator_instance._add_illuminator_command_options()

        # Verify fallback position is used and _calibration_pointing_direction is called
        assert result[0] == "-x 1000.0"  # 10.0m -> 1000.0cm
        assert result[1] == "-y 2000.0"  # 20.0m -> 2000.0cm
        assert result[2] == "-z 3000.0"  # 30.0m -> 3000.0cm
        assert result[3] == "-d 0.8,0.9,1.0"  # pointing from _calibration_pointing_direction

        # Verify _calibration_pointing_direction was called with the fallback position values
        mock_pointing.assert_called_once()


def test__add_flasher_command_options(simulator_instance):
    """Test _add_flasher_command_options method."""

    # Mock calibration model methods
    def mock_get_param_with_unit(name):
        if name == "flasher_position":
            return [5.0 * u.cm, -3.0 * u.cm]
        if name == "flasher_wavelength":
            return 600.0 * u.nm
        if name == "flasher_pulse_shape":
            # New model parameter is [str, float, float]; supply 0s to avoid table generation path
            return ["Gauss", 0.0, 0.0]
        return None

    simulator_instance.calibration_model.get_parameter_value_with_unit.side_effect = (
        mock_get_param_with_unit
    )

    # Provide specific returns for plain-valued params used inside the call
    def mock_get_param(name):
        if name == "flasher_bunch_size":
            return 10000
        if name == "flasher_pulse_shape":
            return ["Gauss", 0.0, 0.0]
        return None

    simulator_instance.calibration_model.get_parameter_value.side_effect = mock_get_param

    # Mock telescope model methods
    mock_diameter = Mock()
    mock_diameter.to.return_value.value = 200.0  # 200 cm diameter
    simulator_instance.telescope_model.get_parameter_value_with_unit.return_value = mock_diameter
    simulator_instance.telescope_model.get_parameter_value.return_value = "hexagonal"

    # Mock helper methods
    with (
        patch.object(
            simulator_instance, "calculate_distance_focal_plane_calibration_device"
        ) as mock_distance,
        patch.object(
            simulator_instance,
            "_get_angular_distribution_string_for_sim_telarray",
            return_value="gauss:2.5",
        ),
        patch.object(
            simulator_instance,
            "_get_pulse_shape_string_for_sim_telarray",
            return_value="square:3.0",
        ),
        patch(
            "simtools.simtel.simulator_light_emission.fiducial_radius_from_shape", return_value=86.6
        ) as mock_radius,
    ):
        mock_distance_value = Mock()
        mock_distance_value.to.return_value.value = 1200.0  # 12 m = 1200 cm
        mock_distance.return_value = mock_distance_value

        # Set up configuration
        simulator_instance.light_emission_config = {
            "number_of_events": 1000,
            "flasher_photons": 500000,
        }

        result = simulator_instance._add_flasher_command_options()

        # Verify the result structure and values
        assert isinstance(result, list)
        assert len(result) == 9
        assert result[0] == "--events 1000"
        assert result[1] == "--photons 500000"
        assert result[2] == "--bunchsize 10000"
        assert result[3] == "--xy 5.0,-3.0"  # flasher x,y position in cm
        assert result[4] == "--distance 1200.0"  # distance in cm
        assert result[5] == "--camera-radius 86.6"  # calculated camera radius
        assert result[6] == "--spectrum 600"  # wavelength in nm (as int)
        assert result[7] == "--lightpulse square:3.0"  # pulse shape
        assert result[8] == "--angular-distribution gauss:2.5"  # angular distribution

        # Verify method calls
        mock_radius.assert_called_once_with(200.0, "hexagonal")
        mock_distance.assert_called_once()


def test__add_flasher_command_options_different_values(simulator_instance):
    """Test _add_flasher_command_options with different parameter values."""

    # Mock calibration model methods with different values
    def mock_get_param_with_unit_2(name):
        if name == "flasher_position":
            return [-10.0 * u.cm, 15.0 * u.cm]
        if name == "flasher_wavelength":
            return 380.0 * u.nm
        if name == "flasher_pulse_shape":
            # Provide unified 3-element list to satisfy new contract
            return ["Gauss", 0.0, 0.0]
        return None

    simulator_instance.calibration_model.get_parameter_value_with_unit.side_effect = (
        mock_get_param_with_unit_2
    )

    # Provide specific returns for plain-valued params used inside the call
    def mock_get_param2(name):
        if name == "flasher_bunch_size":
            return 5000
        if name == "flasher_pulse_shape":
            # get_parameter_value is not used for list in _add_flasher_command_options anymore,
            # but keep a coherent value for other helpers
            return ["Gauss", 0.0, 0.0]
        return None

    simulator_instance.calibration_model.get_parameter_value.side_effect = mock_get_param2

    # Mock telescope model methods
    mock_diameter = Mock()
    mock_diameter.to.return_value.value = 150.0  # 150 cm diameter
    simulator_instance.telescope_model.get_parameter_value_with_unit.return_value = mock_diameter
    simulator_instance.telescope_model.get_parameter_value.return_value = "circular"

    # Mock helper methods
    with (
        patch.object(
            simulator_instance, "calculate_distance_focal_plane_calibration_device"
        ) as mock_distance,
        patch.object(
            simulator_instance,
            "_get_angular_distribution_string_for_sim_telarray",
            return_value="uniform",
        ),
        patch.object(
            simulator_instance, "_get_pulse_shape_string_for_sim_telarray", return_value="gauss"
        ),
        patch(
            "simtools.simtel.simulator_light_emission.fiducial_radius_from_shape", return_value=75.0
        ),
    ):
        mock_distance_value = Mock()
        mock_distance_value.to.return_value.value = 800.0  # 8 m = 800 cm
        mock_distance.return_value = mock_distance_value

        # Set up different configuration
        simulator_instance.light_emission_config = {
            "number_of_events": 2000,
            "flasher_photons": 750000,
        }

        result = simulator_instance._add_flasher_command_options()

        # Verify the result with different values
        assert isinstance(result, list)
        assert len(result) == 9
        assert result[0] == "--events 2000"
        assert result[1] == "--photons 750000"
        assert result[2] == "--bunchsize 5000"
        assert result[3] == "--xy -10.0,15.0"  # different flasher x,y position
        assert result[4] == "--distance 800.0"  # different distance
        assert result[5] == "--camera-radius 75.0"  # different camera radius
        assert result[6] == "--spectrum 380"  # different wavelength
        assert result[7] == "--lightpulse gauss"  # different pulse shape
        assert result[8] == "--angular-distribution uniform"  # different angular distribution


def test__add_flasher_command_options_with_pulse_table(simulator_instance, tmp_test_directory):
    """When pulse width and decay exist, a pulse table is written and used."""

    # Mock calibration model values
    params_with_unit = {
        "flasher_position": [1.0 * u.cm, 2.0 * u.cm],
        "flasher_wavelength": 450.0 * u.nm,
        # Provide full 3-element pulse shape so writer path can read width/exp
        "flasher_pulse_shape": ["Gauss-Exponential", 2.0, 6.0],
    }
    simulator_instance.calibration_model.get_parameter_value_with_unit.side_effect = (
        lambda name: params_with_unit.get(name)
    )

    # Provide specific returns for plain-valued params used inside the call
    plain_params = {
        "flasher_bunch_size": 8000,
        "flasher_angular_distribution": "gaussian",
        "flasher_pulse_shape": ["Gauss-Exponential", 2.0, 6.0],
    }
    simulator_instance.calibration_model.get_parameter_value.side_effect = (
        lambda name: plain_params.get(name)
    )

    # Mock telescope values
    mock_diameter = Mock()
    mock_diameter.to.return_value.value = 180.0
    simulator_instance.telescope_model.get_parameter_value_with_unit.return_value = mock_diameter
    simulator_instance.telescope_model.get_parameter_value.side_effect = (
        lambda key: 40 if key == "fadc_sum_bins" else "hexagonal"
    )

    # Mock distance and helpers
    with (
        patch.object(
            simulator_instance, "calculate_distance_focal_plane_calibration_device"
        ) as mock_distance,
        patch(
            "simtools.simtel.simulator_light_emission.fiducial_radius_from_shape",
            return_value=90.0,
        ),
        patch(
            "simtools.simtel.simulator_light_emission.SimtelConfigWriter.write_light_pulse_table_gauss_exp_conv"
        ) as mock_writer,
    ):
        mock_distance_value = Mock()
        mock_distance_value.to.return_value.value = 1000.0
        mock_distance.return_value = mock_distance_value

        # Configure IO handler for pulse_shapes directory
        pulse_dir = Path(tmp_test_directory) / "pulse_shapes"
        pulse_dir.mkdir(parents=True, exist_ok=True)
        io_mock = Mock()
        io_mock.get_output_directory.return_value = pulse_dir
        simulator_instance.io_handler = io_mock

        # Config and identifiers used in filename
        simulator_instance.output_directory = Path(tmp_test_directory)
        simulator_instance.light_emission_config = {
            "number_of_events": 10,
            "flasher_photons": 1_000_000,
            "telescope": "LSTN-01",
            "light_source": "NectarCam",
        }

        result = simulator_instance._add_flasher_command_options()

        # Writer called with expected numeric values in ns
        assert mock_writer.called
        kwargs = mock_writer.call_args.kwargs
        assert np.isclose(kwargs["width_ns"], 2.0)
        assert np.isclose(kwargs["exp_decay_ns"], 6.0)
        # Command should reference a pulse table path
        assert any(
            str(item).startswith("--lightpulse ") and str(item).endswith(".dat") for item in result
        )


def test__add_flasher_command_options_writer_fallback(simulator_instance, tmp_test_directory):
    """If pulse table writing fails, a warning is logged and token is used."""

    # Calibration parameters
    def mock_get_param_with_unit(name):
        if name == "flasher_position":
            return [1.0 * u.cm, -1.0 * u.cm]
        if name == "flasher_wavelength":
            return 420.0 * u.nm
        if name == "flasher_pulse_shape":
            return ["Gauss-Exponential", 2.0, 6.0]
        return None

    simulator_instance.calibration_model.get_parameter_value_with_unit.side_effect = (
        mock_get_param_with_unit
    )

    # Provide specific returns for plain-valued params used inside the call
    def mock_get_param(name):
        if name == "flasher_bunch_size":
            return 4000
        if name == "flasher_pulse_shape":
            return ["Gauss-Exponential", 2.0, 6.0]
        return None

    simulator_instance.calibration_model.get_parameter_value.side_effect = mock_get_param

    # Telescope parameters
    mock_diameter = Mock()
    mock_diameter.to.return_value.value = 160.0
    simulator_instance.telescope_model.get_parameter_value_with_unit.return_value = mock_diameter
    simulator_instance.telescope_model.get_parameter_value.side_effect = (
        lambda key: 40 if key == "fadc_sum_bins" else "hexagonal"
    )

    # IO and helpers
    pulse_dir = Path(tmp_test_directory) / "pulse_shapes"
    pulse_dir.mkdir(parents=True, exist_ok=True)
    io_mock = Mock()
    io_mock.get_output_directory.return_value = pulse_dir
    simulator_instance.io_handler = io_mock

    # Distance and other string helpers
    with (
        patch.object(
            simulator_instance, "calculate_distance_focal_plane_calibration_device"
        ) as mock_distance,
        patch(
            "simtools.simtel.simulator_light_emission.fiducial_radius_from_shape",
            return_value=75.0,
        ),
        patch.object(
            simulator_instance,
            "_get_angular_distribution_string_for_sim_telarray",
            return_value="uniform",
        ),
        patch.object(
            simulator_instance,
            "_get_pulse_shape_string_for_sim_telarray",
            return_value="gauss-exponential-token",
        ),
        patch(
            "simtools.simtel.simulator_light_emission.SimtelConfigWriter.write_light_pulse_table_gauss_exp_conv",
            side_effect=OSError("boom"),
        ),
    ):
        mock_distance_value = Mock()
        mock_distance_value.to.return_value.value = 900.0
        mock_distance.return_value = mock_distance_value

        simulator_instance.light_emission_config = {
            "number_of_events": 5,
            "flasher_photons": 250000,
            "telescope": "LSTN-01",
            "light_source": "NectarCam",
        }

        result = simulator_instance._add_flasher_command_options()

        # Expect a warning via the instance logger
        assert simulator_instance._logger.warning.called
        assert any(
            "Failed to write pulse shape table" in str(call.args[0])
            for call in simulator_instance._logger.warning.mock_calls
        )

        # Fallback token should be used for --lightpulse, not a .dat file
        lightpulse_args = [arg for arg in result if str(arg).startswith("--lightpulse ")]
        assert len(lightpulse_args) == 1
        assert lightpulse_args[0] == "--lightpulse gauss-exponential-token"


def test__get_light_source_command(simulator_instance):
    """Test _get_light_source_command method."""
    # Test flat_fielding type
    simulator_instance.light_emission_config = {"light_source_type": "flat_fielding"}

    with patch.object(
        simulator_instance, "_add_flasher_command_options", return_value=["flasher_option"]
    ) as mock_flasher:
        result = simulator_instance._get_light_source_command()
        assert result == ["flasher_option"]
        mock_flasher.assert_called_once()

    # Test illuminator type
    simulator_instance.light_emission_config = {"light_source_type": "illuminator"}

    with patch.object(
        simulator_instance, "_add_illuminator_command_options", return_value=["illuminator_option"]
    ) as mock_illuminator:
        result = simulator_instance._get_light_source_command()
        assert result == ["illuminator_option"]
        mock_illuminator.assert_called_once()

    # Test unknown type raises ValueError
    simulator_instance.light_emission_config = {"light_source_type": "unknown_type"}

    with pytest.raises(ValueError, match="Unknown light_source_type 'unknown_type'"):
        simulator_instance._get_light_source_command()


def test__get_site_command(simulator_instance, tmp_test_directory):
    """Test _get_site_command method."""

    # Mock altitude value
    mock_altitude = Mock()
    mock_altitude.to.return_value.value = 2200

    # Test ff-1m app (flasher path)
    with (
        patch.object(
            simulator_instance, "_prepare_flasher_atmosphere_files", return_value="atm_id_123"
        ) as mock_atmo,
        patch("simtools.simtel.simulator_light_emission.settings") as mock_settings,
    ):
        mock_settings.config.sim_telarray_path = Path("/mock/simtel/sim_telarray")
        result = simulator_instance._get_site_command("ff-1m", "/config/dir", mock_altitude)

        expected = [
            "-I.",
            "-I/mock/simtel/sim_telarray/cfg",
            "-I/config/dir",
            "--altitude 2200",
            "--atmosphere atm_id_123",
        ]
        assert result == expected
        mock_atmo.assert_called_once_with("/config/dir")

    # Test default path (non-flasher)
    with patch.object(
        simulator_instance,
        "_write_telescope_position_file",
        return_value=f"{tmp_test_directory}/telpos.txt",
    ) as mock_telpos:
        result = simulator_instance._get_site_command("other-app", "/config/dir", mock_altitude)

        expected = [
            "-h  2200 ",
            f"--telpos-file {tmp_test_directory}/telpos.txt",
        ]
        assert result == expected
        mock_telpos.assert_called_once()


def test__make_light_emission_script(simulator_instance):
    """Test _make_light_emission_script method."""
    simulator_instance.output_directory = "/output"
    simulator_instance.label = "test_label"

    # Mock io_handler
    mock_io_handler = Mock()
    mock_io_handler.get_model_configuration_directory.return_value = "/config/dir"
    simulator_instance.io_handler = mock_io_handler

    # Mock site model
    mock_obs_level = Mock()
    mock_obs_level.to.return_value.value = 2200
    simulator_instance.site_model.get_parameter_value_with_unit.return_value = mock_obs_level
    simulator_instance.site_model.model_version = "test_version"

    # Mock helper methods
    with (
        patch.object(
            simulator_instance, "_get_light_emission_application_name", return_value="ff-1m"
        ) as mock_app_name,
        patch.object(
            simulator_instance, "_get_site_command", return_value=["-I.", "--altitude 2200"]
        ) as mock_site,
        patch.object(
            simulator_instance, "_get_light_source_command", return_value=["--photons 1000000"]
        ) as mock_light_source,
        patch("simtools.simtel.simulator_light_emission.settings") as mock_settings,
    ):
        # Test flat_fielding (no atmospheric profile)
        simulator_instance.light_emission_config = {"light_source_type": "flat_fielding"}
        mock_settings.config.sim_telarray_path = Path("/mock/simtel/sim_telarray")

        result = simulator_instance._make_light_emission_script()

        expected = (
            "/mock/simtel/sim_telarray/LightEmission/ff-1m -I. --altitude 2200 "
            "--photons 1000000 -o /output/ff-1m.iact.gz \n"
        )
        assert result == expected

        # Verify method calls
        mock_app_name.assert_called_once()
        mock_site.assert_called_once_with("ff-1m", "/config/dir", mock_obs_level)
        mock_light_source.assert_called_once()

    # Test illuminator (with atmospheric profile)
    simulator_instance.telescope_model.get_parameter_value.return_value = "atm_profile.dat"

    with (
        patch.object(
            simulator_instance,
            "_get_light_emission_application_name",
            return_value="illuminator-app",
        ),
        patch.object(simulator_instance, "_get_site_command", return_value=["-h 2200"]),
        patch.object(
            simulator_instance, "_get_light_source_command", return_value=["-x 100", "-y 200"]
        ),
        patch("simtools.simtel.simulator_light_emission.settings") as mock_settings,
    ):
        simulator_instance.light_emission_config = {"light_source_type": "illuminator"}
        mock_settings.config.sim_telarray_path = Path("/mock/simtel/sim_telarray")

        result = simulator_instance._make_light_emission_script()

        expected = (
            "/mock/simtel/sim_telarray/LightEmission/illuminator-app -h 2200 "
            "-x 100 -y 200 -A /config/dir/atm_profile.dat "
            "-o /output/illuminator-app.iact.gz \n"
        )
        assert result == expected


def test__prepare_flasher_atmosphere_files(simulator_instance):
    """Test _prepare_flasher_atmosphere_files method."""
    config_directory = Path("/config/dir")

    # Mock site model
    simulator_instance.site_model.get_parameter_value.return_value = "atm_profile.dat"

    # Mock Path operations
    with (
        patch("pathlib.Path.exists", return_value=False),
        patch("pathlib.Path.is_symlink", return_value=False),
        patch("pathlib.Path.symlink_to") as mock_symlink,
        patch("shutil.copy2"),
    ):
        # Test successful symlink creation
        result = simulator_instance._prepare_flasher_atmosphere_files(config_directory)

        # Should return the default model_id
        assert result == 1

        # Should try to create both atmosphere file aliases
        assert mock_symlink.call_count == 2

        # Verify symlink calls
        expected_calls = [
            ((config_directory / "atm_profile.dat",),),
            ((config_directory / "atm_profile.dat",),),
        ]
        mock_symlink.assert_has_calls(expected_calls, any_order=True)

    # Test with custom model_id
    with (
        patch("pathlib.Path.exists", return_value=False),
        patch("pathlib.Path.is_symlink", return_value=False),
        patch("pathlib.Path.symlink_to") as mock_symlink,
    ):
        result = simulator_instance._prepare_flasher_atmosphere_files(config_directory, model_id=5)

        # Should return the custom model_id
        assert result == 5
        assert mock_symlink.call_count == 2


def test__prepare_flasher_atmosphere_files_with_existing_files(simulator_instance):
    """Test _prepare_flasher_atmosphere_files method when files already exist."""
    config_directory = Path("/config/dir")

    # Mock site model
    simulator_instance.site_model.get_parameter_value.return_value = "atm_profile.dat"

    # Mock existing file that needs unlinking
    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.is_symlink", return_value=False),
        patch("pathlib.Path.unlink") as mock_unlink,
        patch("pathlib.Path.symlink_to") as mock_symlink,
    ):
        result = simulator_instance._prepare_flasher_atmosphere_files(config_directory)

        # Should unlink existing files before creating new ones
        assert mock_unlink.call_count == 2
        assert mock_symlink.call_count == 2
        assert result == 1


def test__prepare_flasher_atmosphere_files_symlink_fallback_to_copy(simulator_instance):
    """Test _prepare_flasher_atmosphere_files when symlink fails and falls back to copy."""
    config_directory = Path("/config/dir")

    # Mock site model
    simulator_instance.site_model.get_parameter_value.return_value = "atm_profile.dat"

    # Mock symlink failure, successful copy
    with (
        patch("pathlib.Path.exists", return_value=False),
        patch("pathlib.Path.is_symlink", return_value=False),
        patch("pathlib.Path.symlink_to", side_effect=OSError("Symlink failed")),
        patch("shutil.copy2") as mock_copy,
    ):
        result = simulator_instance._prepare_flasher_atmosphere_files(config_directory)

        # Should fall back to copy when symlink fails
        assert mock_copy.call_count == 2
        assert result == 1


def test__prepare_flasher_atmosphere_files_copy_also_fails(simulator_instance):
    """Test _prepare_flasher_atmosphere_files when both symlink and copy fail."""
    config_directory = Path("/config/dir")

    # Mock site model
    simulator_instance.site_model.get_parameter_value.return_value = "atm_profile.dat"

    # Mock both symlink and copy failure
    with (
        patch("pathlib.Path.exists", return_value=False),
        patch("pathlib.Path.is_symlink", return_value=False),
        patch("pathlib.Path.symlink_to", side_effect=OSError("Symlink failed")),
        patch("shutil.copy2", side_effect=OSError("Copy failed")),
    ):
        result = simulator_instance._prepare_flasher_atmosphere_files(config_directory)

        # Should log warnings but still return model_id
        assert simulator_instance._logger.warning.call_count == 2
        assert result == 1


def test__get_light_emission_application_name(simulator_instance):
    """Test _get_light_emission_application_name method."""
    # Test flat_fielding type returns ff-1m
    simulator_instance.light_emission_config = {"light_source_type": "flat_fielding"}
    result = simulator_instance._get_light_emission_application_name()
    assert result == "ff-1m"

    # Test any other type returns xyzls (default)
    simulator_instance.light_emission_config = {"light_source_type": "illuminator"}
    result = simulator_instance._get_light_emission_application_name()
    assert result == "xyzls"


def test_prepare_script(simulator_instance, tmp_test_directory):
    """Test prepare_script method."""
    # Setup mocks
    simulator_instance.output_directory = Path(tmp_test_directory) / "output"
    simulator_instance.light_emission_config = {"light_source_type": "illuminator"}

    # Mock the internal methods
    with (
        patch.object(
            simulator_instance, "_get_light_emission_application_name", return_value="xyzls"
        ),
        patch.object(
            simulator_instance,
            "_get_simulation_output_filename",
            return_value="test_output.simtel.gz",
        ),
        patch.object(
            simulator_instance, "_make_light_emission_script", return_value="light_emission_cmd"
        ),
        patch.object(simulator_instance, "_make_simtel_script", return_value="simtel_cmd"),
    ):
        result = simulator_instance.prepare_script()

        # Verify return value is the script path
        expected_path = Path(tmp_test_directory) / "output" / "scripts" / "xyzls-light_emission.sh"
        assert result == expected_path

        # Verify script file was created and contains expected content
        assert result.exists()
        content = result.read_text()
        assert "#!/usr/bin/env bash" in content
        assert "light_emission_cmd" in content
        assert "simtel_cmd" in content

        # Verify script is executable
        import stat

        assert result.stat().st_mode & stat.S_IXUSR


def test_prepare_script_output_file_exists(simulator_instance, tmp_test_directory):
    """Test prepare_script method when output file already exists."""
    simulator_instance.output_directory = Path(tmp_test_directory) / "output"
    simulator_instance.light_emission_config = {"light_source_type": "illuminator"}

    # Create the actual output file to trigger FileExistsError
    output_file_path = Path(tmp_test_directory) / "output" / "existing_output.simtel.gz"
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    output_file_path.touch()  # Create the file

    with patch.object(
        simulator_instance, "_get_simulation_output_filename", return_value=str(output_file_path)
    ):
        # Should raise FileExistsError
        with pytest.raises(FileExistsError, match="sim_telarray output file exists"):
            simulator_instance.prepare_script()


def test_simulate(simulator_instance, tmp_test_directory):
    """Test simulate method."""
    from unittest.mock import mock_open

    # Setup
    simulator_instance.output_directory = Path(tmp_test_directory) / "output"
    simulator_instance.output_directory.mkdir(parents=True, exist_ok=True)

    # Mock the methods called by simulate
    mock_script_path = Path(tmp_test_directory) / "output" / "scripts" / "test_script.sh"
    mock_output_file = Path(tmp_test_directory) / "output" / "test_output.simtel.gz"

    with (
        patch.object(simulator_instance, "prepare_script", return_value=mock_script_path),
        patch.object(
            simulator_instance,
            "_get_simulation_output_filename",
            return_value=str(mock_output_file),
        ),
        patch("subprocess.run") as mock_subprocess,
        patch("builtins.open", mock_open()) as mock_file,
    ):
        # Create the output file to simulate successful run
        mock_output_file.parent.mkdir(parents=True, exist_ok=True)
        mock_output_file.touch()

        result = simulator_instance.simulate()

        # Verify subprocess was called correctly
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args
        assert call_args[0][0] == mock_script_path  # First positional arg is the script
        assert call_args[1]["shell"] is False
        assert call_args[1]["check"] is False
        assert call_args[1]["text"] is True

        # Verify log file was opened
        expected_log_path = Path(tmp_test_directory) / "output" / "logfile.log"
        mock_file.assert_called_once_with(expected_log_path, "w", encoding="utf-8")

        # Verify return value
        assert result == mock_output_file


def test_simulate_output_file_missing(simulator_instance, tmp_test_directory):
    """Test simulate method when output file is missing (logs warning)."""
    from unittest.mock import mock_open

    # Setup
    simulator_instance.output_directory = Path(tmp_test_directory) / "output"
    simulator_instance.output_directory.mkdir(parents=True, exist_ok=True)

    # Mock the methods called by simulate
    mock_script_path = Path(tmp_test_directory) / "output" / "scripts" / "test_script.sh"
    mock_output_file = Path(tmp_test_directory) / "output" / "missing_output.simtel.gz"

    with (
        patch.object(simulator_instance, "prepare_script", return_value=mock_script_path),
        patch.object(
            simulator_instance,
            "_get_simulation_output_filename",
            return_value=str(mock_output_file),
        ),
        patch("subprocess.run"),
        patch("builtins.open", mock_open()),
    ):
        # Don't create the output file to simulate missing output
        result = simulator_instance.simulate()

        # Verify warning was logged
        simulator_instance._logger.warning.assert_called_once_with(
            f"Expected sim_telarray output not found: {mock_output_file}"
        )

        # Should still return the expected path
        assert result == mock_output_file


def test__initialize_light_emission_configuration(simulator_instance):
    """Test _initialize_light_emission_configuration method."""

    # Mock calibration model - flasher_type is called twice
    def mock_get_parameter_value(param_name):
        if param_name == "flasher_type":
            return "LED"
        if param_name == "flasher_photons":
            return 5e6
        return None

    simulator_instance.calibration_model.get_parameter_value.side_effect = mock_get_parameter_value

    # Test basic configuration
    config = {"existing_key": "value"}
    result = simulator_instance._initialize_light_emission_configuration(config)

    # Verify flasher_type was converted to light_source_type (lowercase)
    assert result["light_source_type"] == "led"
    assert result["flasher_photons"] == pytest.approx(5e6)
    assert result["existing_key"] == "value"  # Existing key preserved


def test__initialize_light_emission_configuration_test_mode(simulator_instance):
    """Test _initialize_light_emission_configuration method in test mode."""

    # Mock calibration model
    def mock_get_parameter_value(param_name):
        if param_name == "flasher_type":
            return "Laser"
        if param_name == "flasher_photons":
            return 5e6  # Will be overridden by test mode
        return None

    simulator_instance.calibration_model.get_parameter_value.side_effect = mock_get_parameter_value

    # Test configuration with test=True
    config = {"test": True}
    result = simulator_instance._initialize_light_emission_configuration(config)

    # Verify test mode overrides flasher_photons
    assert result["light_source_type"] == "laser"
    assert result["flasher_photons"] == pytest.approx(1e5)  # Test mode value
    assert result["test"] is True


def test__initialize_light_emission_configuration_with_position(simulator_instance):
    """Test _initialize_light_emission_configuration with light_source_position."""
    import numpy as np

    # Mock calibration model - no flasher_type
    def mock_get_parameter_value(param_name):
        if param_name == "flasher_type":
            return None  # No flasher type
        if param_name == "flasher_photons":
            return 1e7
        return None

    simulator_instance.calibration_model.get_parameter_value.side_effect = mock_get_parameter_value

    # Test configuration with position
    config = {"light_source_position": [1.5, 2.0, 3.5]}
    result = simulator_instance._initialize_light_emission_configuration(config)

    # Verify position was converted to astropy units
    assert hasattr(result["light_source_position"], "unit")
    assert result["light_source_position"].unit == u.m
    np.testing.assert_array_equal(result["light_source_position"].value, [1.5, 2.0, 3.5])
    assert result["flasher_photons"] == pytest.approx(1e7)
    # No light_source_type should be set since flasher_type is None
    assert "light_source_type" not in result


def test___init__(tmp_test_directory):
    """Test __init__ method."""
    from simtools.simtel.simulator_light_emission import SimulatorLightEmission

    # Mock the dependencies
    io_handler_path = "simtools.simtel.simulator_light_emission.io_handler.IOHandler"
    models_path = "simtools.simtel.simulator_light_emission.initialize_simulation_models"

    with patch(io_handler_path) as mock_io_handler, patch(models_path) as mock_init_models:
        # Setup mock returns
        mock_io_instance = Mock()
        output_path = Path(tmp_test_directory) / "output"
        mock_io_instance.get_output_directory.return_value = output_path
        mock_io_handler.return_value = mock_io_instance

        mock_telescope_model = Mock()
        mock_site_model = Mock()
        mock_calibration_model = Mock()
        mock_calibration_model.get_parameter_value.return_value = None
        mock_init_models.return_value = (
            mock_telescope_model,
            mock_site_model,
            mock_calibration_model,
        )

        # Test configuration
        config = {
            "site": "North",
            "telescope": "LSTN-01",
            "light_source": "calibration_device",
            "model_version": "6.0.0",
        }

        # Create instance
        instance = SimulatorLightEmission(config, label="test_label")

        # Verify initialization
        assert hasattr(instance, "_logger")
        assert hasattr(instance, "io_handler")
        assert hasattr(instance, "telescope_model")
        assert hasattr(instance, "site_model")
        assert hasattr(instance, "calibration_model")
        assert hasattr(instance, "light_emission_config")

        # Verify models were initialized correctly
        mock_init_models.assert_called_once_with(
            label="test_label",
            site="North",
            telescope_name="LSTN-01",
            calibration_device_name="calibration_device",
            model_version="6.0.0",
        )

        # Verify telescope model config file was written
        mock_telescope_model.write_sim_telarray_config_file.assert_called_once_with(
            additional_models=mock_site_model
        )


def test__get_telescope_pointing(simulator_instance):
    """Test _get_telescope_pointing method."""
    # Test flat_fielding type returns (0.0, 0.0)
    simulator_instance.light_emission_config = {"light_source_type": "flat_fielding"}
    result = simulator_instance._get_telescope_pointing()
    assert result == (0.0, 0.0)

    # Test with light_source_position returns (0.0, 0.0) and logs info message
    simulator_instance.light_emission_config = {
        "light_source_type": "illuminator",
        "light_source_position": [1.0, 2.0, 3.0],
    }
    result = simulator_instance._get_telescope_pointing()
    assert result == (0.0, 0.0)
    simulator_instance._logger.info.assert_called_with(
        "Using fixed (vertical up) telescope pointing."
    )

    # Test default case uses _calibration_pointing_direction
    simulator_instance.light_emission_config = {"light_source_type": "illuminator"}
    simulator_instance._logger.reset_mock()

    with patch.object(
        simulator_instance,
        "_calibration_pointing_direction",
        return_value=(None, [45.0, 180.0, 90.0, 0.0]),
    ) as mock_pointing:
        result = simulator_instance._get_telescope_pointing()

        # Should return tel_theta, tel_phi (first two angles)
        assert result == (45.0, 180.0)
        mock_pointing.assert_called_once()
        simulator_instance._logger.info.assert_not_called()


def test__write_telescope_position_file(simulator_instance):
    """Test _write_telescope_position_file method."""
    simulator_instance.output_directory = Path("/output")

    # Mock telescope model parameters
    mock_x = Mock()
    mock_x.to.return_value.value = 100.0
    mock_y = Mock()
    mock_y.to.return_value.value = 200.0
    mock_z = Mock()
    mock_z.to.return_value.value = 300.0

    mock_radius = Mock()
    mock_radius.to.return_value.value = 1500.0

    simulator_instance.telescope_model.get_parameter_value_with_unit.side_effect = [
        [mock_x, mock_y, mock_z],  # array_element_position_ground
        mock_radius,  # telescope_sphere_radius
    ]

    # Mock file writing
    mock_file = Mock()
    with (
        patch.object(Path, "joinpath", return_value=mock_file) as mock_joinpath,
        patch.object(mock_file, "write_text") as mock_write,
    ):
        result = simulator_instance._write_telescope_position_file()

        # Should return the telescope position file path
        assert result == mock_file

        # Should create file in output directory
        mock_joinpath.assert_called_once_with("telescope_position.dat")

        # Should write coordinates and radius in correct format
        expected_content = "100.0 200.0 300.0 1500.0\n"
        mock_write.assert_called_once_with(expected_content, encoding="utf-8")

        # Verify unit conversions were called
        mock_x.to.assert_called_once_with(u.cm)
        mock_y.to.assert_called_once_with(u.cm)
        mock_z.to.assert_called_once_with(u.cm)
        mock_radius.to.assert_called_once_with(u.cm)


def test__calibration_pointing_direction(simulator_instance):
    """Test _calibration_pointing_direction method."""
    import numpy as np

    # Mock calibration device position at origin
    cal_x, cal_y, cal_z = 0 * u.m, 0 * u.m, 0 * u.m
    simulator_instance.calibration_model.get_parameter_value_with_unit.return_value = [
        cal_x,
        cal_y,
        cal_z,
    ]

    # Mock telescope position at (10, 0, 10) meters
    tel_x, tel_y, tel_z = 10 * u.m, 0 * u.m, 10 * u.m
    simulator_instance.telescope_model.get_parameter_value_with_unit.return_value = [
        tel_x,
        tel_y,
        tel_z,
    ]

    pointing_vector, angles = simulator_instance._calibration_pointing_direction()

    # Verify calculations - direction vector is [10, 0, 10]
    expected_direction = np.array([10.0, 0.0, 10.0])
    expected_norm = np.linalg.norm(expected_direction)  # sqrt(200) roughly 14.142
    expected_pointing = np.round(expected_direction / expected_norm, 6).tolist()

    assert pointing_vector == expected_pointing
    assert len(angles) == 4  # tel_theta, tel_phi, source_theta, source_phi

    # Verify the angles are calculated correctly
    tel_theta, tel_phi, source_theta, source_phi = angles
    assert abs(tel_theta - 135.0) < 0.1
    assert abs(tel_phi - 180.0) < 0.1
    assert abs(source_theta - 135.0) < 0.1
    assert abs(source_phi + 180.0) < 0.1

    # Verify model calls
    simulator_instance.calibration_model.get_parameter_value_with_unit.assert_called_with(
        "array_element_position_ground"
    )
    simulator_instance.telescope_model.get_parameter_value_with_unit.assert_called_with(
        "array_element_position_ground"
    )


def test__calibration_pointing_direction_with_custom_params(simulator_instance):
    """Test _calibration_pointing_direction method with custom position parameters."""
    import numpy as np

    # Mock telescope position
    tel_x, tel_y, tel_z = 5 * u.m, 5 * u.m, 0 * u.m
    simulator_instance.telescope_model.get_parameter_value_with_unit.return_value = [
        tel_x,
        tel_y,
        tel_z,
    ]

    # Call with custom calibration position parameters
    custom_x = 0 * u.m
    custom_y = 0 * u.m
    custom_z = 5 * u.m

    pointing_vector, angles = simulator_instance._calibration_pointing_direction(
        x_cal=custom_x, y_cal=custom_y, z_cal=custom_z
    )

    # Verify calculations - direction vector is [5, 5, -5]
    expected_direction = np.array([5.0, 5.0, -5.0])
    expected_norm = np.linalg.norm(expected_direction)
    expected_pointing = np.round(expected_direction / expected_norm, 6).tolist()

    assert pointing_vector == expected_pointing
    assert len(angles) == 4

    # Verify calibration model was NOT called (custom params provided)
    simulator_instance.calibration_model.get_parameter_value_with_unit.assert_not_called()
    # But telescope model should still be called
    simulator_instance.telescope_model.get_parameter_value_with_unit.assert_called_with(
        "array_element_position_ground"
    )
