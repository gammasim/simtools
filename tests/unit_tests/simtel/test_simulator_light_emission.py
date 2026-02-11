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
    inst = SimulatorLightEmission.__new__(SimulatorLightEmission)
    # Create fresh mocks for each test to avoid cross-test contamination
    inst.calibration_model = Mock()
    inst.telescope_model = Mock()
    inst.site_model = Mock()
    inst.light_emission_config = {}
    inst.job_files = Mock()
    inst.output_directory = "/test/output"
    inst._logger = Mock()
    inst.runner_service = Mock()
    inst.io_handler = Mock()
    return inst


@patch.object(SimulatorLightEmission, "_get_telescope_pointing")
@patch.object(SimulatorLightEmission, "_get_light_emission_application_name")
def test__make_simtel_script(mock_app_name, mock_pointing, simulator_instance):
    """Test _make_simtel_script method with different conditions."""
    simulator_instance.telescope_model.config_file_directory = "/mock/config"
    simulator_instance.telescope_model.config_file_path = "/mock/config/telescope.cfg"
    simulator_instance.light_emission_config = {"light_source_type": "flat_fielding"}

    mock_altitude = Mock()
    mock_altitude.to.return_value.value = 2200
    simulator_instance.site_model.get_parameter_value_with_unit.return_value = mock_altitude
    simulator_instance.site_model.get_parameter_value.return_value = "atm_trans.dat"

    mock_pointing.return_value = [10.5, 20.0]
    mock_app_name.return_value = "test-app"

    result = simulator_instance._make_simtel_script()
    assert isinstance(result, str)
    assert len(result) > 0


def test__make_simtel_script_bypass_optics_condition(simulator_instance):
    """Test that flat_fielding adds Bypass_Optics option."""
    # Setup minimal mocks
    simulator_instance.telescope_model.config_file_directory = "/mock/config"
    simulator_instance.telescope_model.config_file_path = "/mock/config/telescope.cfg"

    mock_altitude = Mock()
    mock_altitude.to.return_value.value = 2200
    simulator_instance.site_model.get_parameter_value_with_unit.return_value = mock_altitude
    simulator_instance.site_model.get_parameter_value.return_value = "atm_trans.dat"

    # Mock the helper methods
    with (
        patch.object(simulator_instance, "_get_telescope_pointing", return_value=[0, 0]),
        patch.object(
            simulator_instance, "_get_light_emission_application_name", return_value="ff-1m"
        ),
    ):
        # Test flat_fielding - should include Bypass_Optics
        simulator_instance.light_emission_config = {"light_source_type": "flat_fielding"}

        options = simulator_instance._make_simtel_script()
        assert "Bypass_Optics=1" in options

        # Test illuminator - should NOT include Bypass_Optics
        simulator_instance.light_emission_config = {"light_source_type": "illuminator"}

        options = simulator_instance._make_simtel_script()
        assert "Bypass_Optics=1" not in options


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


def test__get_angular_distribution_string_for_sim_telarray_lambertian(
    simulator_instance, tmp_test_directory
):
    """Lambertian distribution should generate a table file and return its path."""
    from pathlib import Path

    # Prepare mocked IO handler directory
    base_dir = Path(tmp_test_directory) / "angular_distributions"
    base_dir.mkdir(parents=True, exist_ok=True)
    io_mock = Mock()
    io_mock.get_output_directory.return_value = base_dir
    simulator_instance.io_handler = io_mock

    # Provide telescope/light_source identifiers for filename construction
    simulator_instance.light_emission_config = {
        "telescope": "TEL01",
        "light_source": "CalibA",
    }

    # Mock calibration model values
    simulator_instance.calibration_model.get_parameter_value.side_effect = (
        lambda name: "Lambertian" if name == "flasher_angular_distribution" else None
    )
    # Width is ignored for Lambertian; no need to mock width conversion.
    simulator_instance.calibration_model.get_parameter_value_with_unit.side_effect = (
        lambda name: None
    )

    result = simulator_instance._get_angular_distribution_string_for_sim_telarray()

    # Result should be a path to the generated table
    table_path = Path(result)
    assert str(table_path).endswith(".dat")
    assert table_path.exists()
    content = table_path.read_text().splitlines()
    assert content[0].startswith("# angle[deg] relative_intensity")
    # Expect 101 lines: header + 100 samples (0..max angle)
    assert len(content) == 101
    # Width parameter should not have been requested for Lambertian
    assert simulator_instance.calibration_model.get_parameter_value_with_unit.call_count == 0


def test__get_angular_distribution_string_for_sim_telarray_lambertian_failure(
    simulator_instance,
):
    """Test Lambertian distribution failure handling."""
    # Mock calibration model values
    simulator_instance.calibration_model.get_parameter_value.side_effect = (
        lambda name: "Lambertian" if name == "flasher_angular_distribution" else None
    )

    # Mock _generate_lambertian_angular_distribution_table to raise OSError
    with patch.object(
        simulator_instance,
        "_generate_lambertian_angular_distribution_table",
        side_effect=OSError("Write failed"),
    ):
        result = simulator_instance._get_angular_distribution_string_for_sim_telarray()

    # Should return the token string "lambertian" and log a warning
    assert result == "lambertian"
    simulator_instance._logger.warning.assert_called_with(
        "Failed to write Lambertian angular distribution table: Write failed; using token instead."
    )


@pytest.mark.parametrize(
    ("pulse_config", "expected_result"),
    [
        (["Gauss", 5.0, 0.0], "gauss:5.0"),
        (["Line", 0.0, 0.0], "line"),
        (["Tophat", 7.5, 0.0], "simple:7.5"),
        (["Gauss-Exponential", 2.0, 6.0], "gauss-exponential:2.0:6.0"),
        (["Exponential", 0.0, 3.2], "exponential:3.2"),
    ],
)
def test__get_pulse_shape_string_for_sim_telarray(
    simulator_instance, pulse_config, expected_result
):
    """Test _get_pulse_shape_string_for_sim_telarray with various pulse shapes."""
    simulator_instance.calibration_model.get_parameter_value.return_value = pulse_config

    result = simulator_instance._get_pulse_shape_string_for_sim_telarray()
    assert result == expected_result

    simulator_instance.calibration_model.get_parameter_value.assert_called_once_with(
        "flasher_pulse_shape"
    )


def _get_mock_param_side_effect_for_test_1(name):
    """Mock side effect for test 1 in parametrize."""
    if name == "array_element_position_ground":
        return [1.0 * u.m, 2.0 * u.m, 3.0 * u.m]
    if name == "flasher_wavelength":
        return 450 * u.nm
    return None


def _get_mock_param_side_effect_for_test_3(name):
    """Mock side effect for test 3 in parametrize."""
    if name == "array_element_position_ground":
        return [10.0 * u.m, 20.0 * u.m, 30.0 * u.m]
    if name == "flasher_wavelength":
        return 500 * u.nm
    return None


@pytest.mark.parametrize(
    (
        "config",
        "wavelength_nm",
        "get_param_side_effect",
        "expected_x",
        "expected_y",
        "expected_z",
        "expected_d",
        "expected_n",
        "expected_s",
        "patch_angular",
        "patch_pulse",
    ),
    [
        # Test 1: Default config (no custom position)
        (
            {"flasher_photons": 1000000},
            450,
            _get_mock_param_side_effect_for_test_1,
            "-x 100.0",
            "-y 200.0",
            "-z 300.0",
            "-d 0.1,0.2,0.3",
            "-n 1000000",
            "-s 450",
            "gauss:5.0",
            "square:2.0",
        ),
        # Test 2: Custom position and pointing
        (
            {
                "flasher_photons": 2000000,
                "light_source_position": [5.0 * u.m, 6.0 * u.m, 7.0 * u.m],
                "light_source_pointing": [0.5, 0.6, 0.7],
            },
            380,
            lambda name: 380 * u.nm if name == "flasher_wavelength" else None,
            "-x 500.0",
            "-y 600.0",
            "-z 700.0",
            "-d 0.5,0.6,0.7",
            "-n 2000000",
            "-s 380",
            "uniform",
            "gauss",
        ),
        # Test 3: Fallback position (None position)
        (
            {"flasher_photons": 500000, "light_source_position": None},
            500,
            _get_mock_param_side_effect_for_test_3,
            "-x 1000.0",
            "-y 2000.0",
            "-z 3000.0",
            "-d 0.8,0.9,1.0",
            "-n 500000",
            "-s 500",
            "test_angular",
            "test_pulse",
        ),
    ],
)
def test__add_illuminator_command_options(
    simulator_instance,
    config,
    wavelength_nm,
    get_param_side_effect,
    expected_x,
    expected_y,
    expected_z,
    expected_d,
    expected_n,
    expected_s,
    patch_angular,
    patch_pulse,
):
    """Test _add_illuminator_command_options with various configurations."""
    simulator_instance.calibration_model.get_parameter_value_with_unit.side_effect = (
        get_param_side_effect
    )

    if "light_source_position" not in config:
        pointing_vector = [0.1, 0.2, 0.3]
    elif config.get("light_source_position") is None:
        pointing_vector = [0.8, 0.9, 1.0]
    else:
        pointing_vector = [0.5, 0.6, 0.7]

    with (
        patch.object(
            simulator_instance,
            "_get_angular_distribution_string_for_sim_telarray",
            return_value=patch_angular,
        ),
        patch.object(
            simulator_instance,
            "_get_pulse_shape_string_for_sim_telarray",
            return_value=patch_pulse,
        ),
        patch.object(
            simulator_instance,
            "_calibration_pointing_direction",
            return_value=(pointing_vector, []),
        ),
    ):
        simulator_instance.light_emission_config = config
        result = simulator_instance._add_illuminator_command_options()

        assert isinstance(result, list)
        assert len(result) == 8
        assert result[0] == expected_x
        assert result[1] == expected_y
        assert result[2] == expected_z
        assert result[3] == expected_d
        assert result[4] == expected_n
        assert result[5] == expected_s
        assert result[6] == f"-p {patch_pulse}"
        assert result[7] == f"-a {patch_angular}"


@pytest.mark.parametrize(
    (
        "flasher_x",
        "flasher_y",
        "bunch_size",
        "num_events",
        "photons",
        "wavelength",
        "diameter",
        "shape_type",
        "distance",
        "radius",
        "tel_shape",
    ),
    [
        # Test 1: Standard values
        (5.0, -3.0, 10000, 1000, 500000, 600, 200.0, "square:3.0", 1200.0, 86.6, "hexagonal"),
        # Test 2: Different values
        (-10.0, 15.0, 5000, 2000, 750000, 380, 150.0, "gauss", 800.0, 75.0, "circular"),
    ],
)
def test__add_flasher_command_options(
    simulator_instance,
    flasher_x,
    flasher_y,
    bunch_size,
    num_events,
    photons,
    wavelength,
    diameter,
    shape_type,
    distance,
    radius,
    tel_shape,
):
    """Test _add_flasher_command_options with various parameter values."""

    def mock_get_param_with_unit(name):
        if name == "flasher_position":
            return [flasher_x * u.cm, flasher_y * u.cm]
        if name == "flasher_wavelength":
            return float(wavelength) * u.nm
        if name == "flasher_pulse_shape":
            return ["Gauss", 0.0, 0.0]
        return None

    simulator_instance.calibration_model.get_parameter_value_with_unit.side_effect = (
        mock_get_param_with_unit
    )

    def mock_get_param(name):
        if name == "flasher_bunch_size":
            return bunch_size
        if name == "flasher_pulse_shape":
            return ["Gauss", 0.0, 0.0]
        return None

    simulator_instance.calibration_model.get_parameter_value.side_effect = mock_get_param

    mock_diam = Mock()
    mock_diam.to.return_value.value = diameter
    simulator_instance.telescope_model.get_parameter_value_with_unit.return_value = mock_diam
    simulator_instance.telescope_model.get_parameter_value.return_value = tel_shape

    with (
        patch.object(
            simulator_instance, "calculate_distance_focal_plane_calibration_device"
        ) as mock_distance,
        patch.object(
            simulator_instance,
            "_get_angular_distribution_string_for_sim_telarray",
            return_value="gauss:2.5" if np.isclose(distance, 1200.0) else "uniform",
        ),
        patch.object(
            simulator_instance,
            "_get_pulse_shape_string_for_sim_telarray",
            return_value=shape_type,
        ),
        patch(
            "simtools.simtel.simulator_light_emission.fiducial_radius_from_shape",
            return_value=radius,
        ) as mock_radius,
    ):
        mock_distance_value = Mock()
        mock_distance_value.to.return_value.value = distance
        mock_distance.return_value = mock_distance_value

        simulator_instance.light_emission_config = {
            "number_of_events": num_events,
            "flasher_photons": photons,
        }

        result = simulator_instance._add_flasher_command_options()

        assert isinstance(result, list)
        assert len(result) == 9
        assert result[0] == f"--events {num_events}"
        assert result[1] == f"--photons {photons}"
        assert result[2] == f"--bunchsize {bunch_size}"
        assert result[3] == f"--xy {flasher_x},{flasher_y}"
        assert result[4] == f"--distance {distance}"
        assert result[5] == f"--camera-radius {radius}"
        assert result[6] == f"--spectrum {wavelength}"
        assert result[7] == f"--lightpulse {shape_type}"
        assert (
            result[8]
            == f"--angular-distribution {'gauss:2.5' if distance == 1200.0 else 'uniform'}"
        )

        mock_radius.assert_called_once_with(diameter, tel_shape)
        mock_distance.assert_called_once()


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


def test__add_flasher_command_options_invalid_gauss_exponential_width(simulator_instance):
    """Gauss-Exponential with non-positive width must raise ValueError."""

    # Minimal calibration mocks
    def mock_get_param_with_unit(name):
        if name == "flasher_position":
            return [0.0 * u.cm, 0.0 * u.cm, 0.0 * u.cm]
        if name == "flasher_wavelength":
            return 400.0 * u.nm
        if name == "flasher_pulse_shape":
            return ["Gauss-Exponential", 0.0, 5.0]  # invalid width
        return None

    simulator_instance.calibration_model.get_parameter_value_with_unit.side_effect = (
        mock_get_param_with_unit
    )
    simulator_instance.calibration_model.get_parameter_value.side_effect = (
        lambda k: 8000 if k == "flasher_bunch_size" else ["Gauss-Exponential", 0.0, 5.0]
    )

    # Telescope minimal mocks
    mock_diameter = Mock()
    mock_diameter.to.return_value.value = 200.0
    simulator_instance.telescope_model.get_parameter_value_with_unit.return_value = mock_diameter
    simulator_instance.telescope_model.get_parameter_value.side_effect = (
        lambda k: 40 if k == "fadc_sum_bins" else "hexagonal"
    )

    simulator_instance.light_emission_config = {"number_of_events": 1, "flasher_photons": 100}

    # Bypass geometry shape validation to exercise Gauss-Exponential parameter check
    with (
        patch(
            "simtools.simtel.simulator_light_emission.fiducial_radius_from_shape",
            return_value=75.0,
        ),
        patch.object(
            simulator_instance,
            "calculate_distance_focal_plane_calibration_device",
            return_value=Mock(**{"to.return_value.value": 900.0}),
        ),
    ):
        with pytest.raises(
            ValueError,
            match="Gauss-Exponential pulse shape requires positive width and exponential decay values",
        ):
            simulator_instance._add_flasher_command_options()


def test__add_flasher_command_options_invalid_gauss_exponential_decay(simulator_instance):
    """Gauss-Exponential with non-positive decay must raise ValueError."""

    # Minimal calibration mocks
    def mock_get_param_with_unit(name):
        if name == "flasher_position":
            return [0.0 * u.cm, 0.0 * u.cm, 0.0 * u.cm]
        if name == "flasher_wavelength":
            return 420.0 * u.nm
        if name == "flasher_pulse_shape":
            return ["Gauss-Exponential", 2.0, 0.0]  # invalid decay
        return None

    simulator_instance.calibration_model.get_parameter_value_with_unit.side_effect = (
        mock_get_param_with_unit
    )
    simulator_instance.calibration_model.get_parameter_value.side_effect = (
        lambda k: 4000 if k == "flasher_bunch_size" else ["Gauss-Exponential", 2.0, 0.0]
    )

    # Telescope minimal mocks
    mock_diameter = Mock()
    mock_diameter.to.return_value.value = 160.0
    simulator_instance.telescope_model.get_parameter_value_with_unit.return_value = mock_diameter
    simulator_instance.telescope_model.get_parameter_value.side_effect = (
        lambda k: 40 if k == "fadc_sum_bins" else "hexagonal"
    )

    simulator_instance.light_emission_config = {"number_of_events": 1, "flasher_photons": 100}

    # Bypass geometry shape validation to exercise Gauss-Exponential parameter check
    with (
        patch(
            "simtools.simtel.simulator_light_emission.fiducial_radius_from_shape",
            return_value=75.0,
        ),
        patch.object(
            simulator_instance,
            "calculate_distance_focal_plane_calibration_device",
            return_value=Mock(**{"to.return_value.value": 900.0}),
        ),
    ):
        with pytest.raises(
            ValueError,
            match="Gauss-Exponential pulse shape requires positive width and exponential decay values",
        ):
            simulator_instance._add_flasher_command_options()


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
    """Test _make_light_emission_command method."""
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

    # Mock runner_service.get_file_name to return a fixed string
    simulator_instance.runner_service.get_file_name.return_value = "/output/test_log.log"

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

        result = simulator_instance._make_light_emission_command("/output/ff-1m.iact.gz")

        expected = (
            "/mock/simtel/sim_telarray/LightEmission/ff-1m -I. --altitude 2200 "
            "--photons 1000000 -o /output/ff-1m.iact.gz "
            "| gzip > /output/test_log.log 2>&1\n"
        )
        assert result == expected

        # Verify method calls
        mock_app_name.assert_called_once()
        mock_site.assert_called_once_with("ff-1m", "/config/dir", mock_obs_level)
        mock_light_source.assert_called_once()
        simulator_instance.runner_service.get_file_name.assert_called_once_with(
            file_type="light_emission_log"
        )

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

        result = simulator_instance._make_light_emission_command("/output/illuminator-app.iact.gz")

        expected = (
            "/mock/simtel/sim_telarray/LightEmission/illuminator-app -h 2200 "
            "-x 100 -y 200 -A /config/dir/atm_profile.dat "
            "-o /output/illuminator-app.iact.gz "
            "| gzip > /output/test_log.log 2>&1\n"
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


def test_prepare_run(simulator_instance, tmp_test_directory):
    """Test prepare_run method."""
    # Setup mocks
    simulator_instance.output_directory = Path(tmp_test_directory) / "output"
    simulator_instance.light_emission_config = {"light_source_type": "illuminator"}

    script_dir = Path(tmp_test_directory) / "output" / "scripts"
    script_dir.mkdir(parents=True, exist_ok=True)
    script_path = script_dir / "xyzls-light_emission.sh"

    # Mock job_files.get_file_name to return the script path
    def job_files_get_file_name_side_effect(file_type):
        if file_type == "sub_script":
            return script_path
        return Path(tmp_test_directory) / "output" / f"{file_type}.tmp"

    simulator_instance.job_files.get_file_name.side_effect = job_files_get_file_name_side_effect

    # Mock runner_service.get_file_name to return paths
    def get_file_name_side_effect(file_type):
        if file_type == "sim_telarray_output":
            return Path(tmp_test_directory) / "output" / "test_output.simtel.gz"
        if file_type == "iact_output":
            return Path(tmp_test_directory) / "output" / "iact.dat"
        return Path(tmp_test_directory) / "output" / f"{file_type}.tmp"

    simulator_instance.runner_service.get_file_name.side_effect = get_file_name_side_effect

    # Mock the internal methods
    with (
        patch.object(
            simulator_instance, "_make_light_emission_command", return_value="light_emission_cmd"
        ),
        patch.object(simulator_instance, "_make_simtel_script", return_value="simtel_cmd"),
    ):
        result = simulator_instance.prepare_run()

        # Verify return value is the script path
        expected_path = Path(tmp_test_directory) / "output" / "scripts" / "xyzls-light_emission.sh"
        assert result == expected_path

        # Verify script file was created and contains expected content
        assert result.exists()
        content = result.read_text()
        assert "#!/usr/bin/env bash" in content
        assert "light_emission_cmd" in content
        assert "simtel_cmd" in content


def test_prepare_run_output_file_exists(simulator_instance, tmp_test_directory):
    """Test prepare_run method when output file already exists."""
    simulator_instance.output_directory = Path(tmp_test_directory) / "output"
    simulator_instance.light_emission_config = {"light_source_type": "illuminator"}

    # Create the actual output file to trigger FileExistsError
    output_file_path = Path(tmp_test_directory) / "output" / "existing_output.simtel.gz"
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    output_file_path.touch()  # Create the file

    # Setup mock to return the output file that already exists
    def get_file_name_side_effect(file_type):
        if file_type == "sim_telarray_output":
            return output_file_path
        if file_type == "sub_script":
            return Path(tmp_test_directory) / "output" / "script.sh"
        if file_type == "iact_output":
            return Path(tmp_test_directory) / "output" / "iact.dat"
        return Path(tmp_test_directory) / "output" / f"{file_type}.tmp"

    simulator_instance.runner_service.get_file_name.side_effect = get_file_name_side_effect

    # Should raise FileExistsError
    with pytest.raises(FileExistsError, match="sim_telarray output file exists"):
        simulator_instance.prepare_run()


def test_simulate(simulator_instance, tmp_test_directory):
    """Test simulate method."""
    # Setup
    simulator_instance.output_directory = Path(tmp_test_directory) / "output"
    simulator_instance.output_directory.mkdir(parents=True, exist_ok=True)

    # Mock the methods called by simulate
    mock_script_path = Path(tmp_test_directory) / "output" / "scripts" / "test_script.sh"
    mock_script_path.parent.mkdir(parents=True, exist_ok=True)
    mock_output_file = Path(tmp_test_directory) / "output" / "test_output.simtel.gz"

    # Setup job_files mock to return the script path
    def job_files_get_file_name_side_effect(file_type):
        if file_type == "sub_script":
            return mock_script_path
        return Path(tmp_test_directory) / "output" / f"{file_type}.tmp"

    simulator_instance.job_files.get_file_name.side_effect = job_files_get_file_name_side_effect

    # Setup runner_service mock to return the output file and other paths
    def get_file_name_side_effect(file_type):
        if file_type == "sim_telarray_output":
            return mock_output_file
        if file_type == "sub_out":
            return Path(tmp_test_directory) / "output" / "logfile.log"
        if file_type == "sub_err":
            return Path(tmp_test_directory) / "output" / "logfile.err"
        if file_type == "iact_output":
            return Path(tmp_test_directory) / "output" / "iact.dat"
        return Path(tmp_test_directory) / "output" / f"{file_type}.tmp"

    simulator_instance.runner_service.get_file_name.side_effect = get_file_name_side_effect

    with patch("simtools.job_execution.job_manager.submit") as mock_job_submit:
        # Mock make_run_command to return a simple script
        with patch.object(
            simulator_instance, "make_run_command", return_value=["#!/bin/bash\n", "echo test\n"]
        ):
            simulator_instance.simulate()

        # Create the output file to simulate successful run (this happens during simulate())
        mock_output_file.parent.mkdir(parents=True, exist_ok=True)
        mock_output_file.touch()

        # Verify job_manager.submit was called correctly
        mock_job_submit.assert_called_once()
        call_args = mock_job_submit.call_args
        assert call_args[0][0] == mock_script_path  # First positional arg is the script


@pytest.mark.parametrize(
    (
        "flasher_type",
        "input_config",
        "expected_light_source_type",
        "expected_photons",
        "test_mode_enabled",
    ),
    [
        # Test 1: Basic configuration with LED
        ("LED", {"existing_key": "value"}, "led", 5e6, False),
        # Test 2: Test mode with Laser (photons should be overridden to 1e5)
        ("Laser", {"test": True}, "laser", 1e5, True),
        # Test 3: No flasher_type with position
        (None, {"light_source_position": [1.5, 2.0, 3.5]}, None, 1e7, False),
    ],
)
def test__initialize_light_emission_configuration(
    simulator_instance,
    flasher_type,
    input_config,
    expected_light_source_type,
    expected_photons,
    test_mode_enabled,
):
    """Test _initialize_light_emission_configuration with various configurations."""
    import numpy as np

    def mock_get_parameter_value(param_name):
        if param_name == "flasher_type":
            return flasher_type
        if param_name == "flasher_photons":
            return 5e6 if flasher_type in ("LED", "Laser") else 1e7
        return None

    simulator_instance.calibration_model.get_parameter_value.side_effect = mock_get_parameter_value

    config_copy = input_config.copy()
    result = simulator_instance._initialize_light_emission_configuration(config_copy)

    # Verify light_source_type is set correctly
    if expected_light_source_type is not None:
        assert result["light_source_type"] == expected_light_source_type
    else:
        assert "light_source_type" not in result

    # Verify flasher_photons
    assert result["flasher_photons"] == pytest.approx(expected_photons)

    # Verify position handling
    if "light_source_position" in input_config:
        assert hasattr(result["light_source_position"], "unit")
        assert result["light_source_position"].unit == u.m
        np.testing.assert_array_equal(result["light_source_position"].value, [1.5, 2.0, 3.5])

    # Verify test flag
    if test_mode_enabled:
        assert result.get("test") is True

    # Verify existing keys are preserved
    if "existing_key" in input_config:
        assert result["existing_key"] == "value"


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


def test__write_telescope_position_file(simulator_instance, tmp_test_directory):
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

    # Use real temporary directory for file writing
    mock_output_dir = Path(tmp_test_directory)
    mock_output_dir.mkdir(parents=True, exist_ok=True)
    simulator_instance.io_handler.get_output_directory.return_value = mock_output_dir
    expected_file = mock_output_dir / "telescope_position.dat"

    # Call the method
    result = simulator_instance._write_telescope_position_file()

    # Should return the telescope position file path
    assert result == expected_file

    # Verify file was created with correct content
    assert result.exists()
    content = result.read_text(encoding="utf-8")
    expected_content = "100.0 200.0 300.0 1500.0\n"
    assert content == expected_content

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

    result = simulator_instance._calibration_pointing_direction(
        x_cal=custom_x, y_cal=custom_y, z_cal=custom_z
    )

    # The method returns a tuple: (pointing_vector, [theta, phi, source_theta, source_phi])
    pointing_vector, _ = result

    # Verify calculations - direction vector is [5, 5, -5]
    expected_direction = np.array([5.0, 5.0, -5.0])
    expected_norm = np.linalg.norm(expected_direction)
    expected_pointing = np.round(expected_direction / expected_norm, 6).tolist()

    assert pointing_vector == expected_pointing

    # Verify calibration model was NOT called (custom params provided)
    simulator_instance.calibration_model.get_parameter_value_with_unit.assert_not_called()
    # But telescope model should still be called
    simulator_instance.telescope_model.get_parameter_value_with_unit.assert_called_with(
        "array_element_position_ground"
    )


def test__get_angular_distribution_string_for_sim_telarray_isotropic(simulator_instance):
    """Test isotropic distribution returns just the token."""
    simulator_instance.calibration_model.get_parameter_value.return_value = "Isotropic"

    # Even if width is available (though it shouldn't be for isotropic), it should be ignored
    mock_width = Mock()
    mock_width.to.return_value.value = 10.0
    simulator_instance.calibration_model.get_parameter_value_with_unit.return_value = mock_width

    result = simulator_instance._get_angular_distribution_string_for_sim_telarray()
    assert result == "isotropic"

    # Verify width was NOT requested: the implementation returns early for isotropic distributions
    # before attempting to fetch the width via get_parameter_value_with_unit.
    simulator_instance.calibration_model.get_parameter_value_with_unit.assert_not_called()
