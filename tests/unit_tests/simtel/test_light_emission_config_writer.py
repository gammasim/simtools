from pathlib import Path
from unittest.mock import Mock, patch

import astropy.units as u
import numpy as np
import pytest

from simtools.model.model_parameter import InvalidModelParameterError
from simtools.simulation.configuration import get_light_source_writer


def test__get_angular_distribution_string_for_sim_telarray(simulator_instance):
    # Test with width provided
    simulator_instance.calibration_model.get_parameter_value.return_value = "Gauss"
    mock_width = Mock()
    mock_width.to.return_value.value = 5.0
    simulator_instance.calibration_model.get_parameter_value_with_unit.return_value = mock_width

    result = get_light_source_writer(
        simulator_instance
    ).get_angular_distribution_string_for_sim_telarray()
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
    simulator_instance.calibration_model.get_parameter_value.side_effect = lambda name: {
        "flasher_angular_distribution": "Lambertian",
    }.get(name)

    # Mock the unit-aware method for width parameter
    simulator_instance.calibration_model.get_parameter_value_with_unit.return_value = 45.0 * u.deg

    result = get_light_source_writer(
        simulator_instance
    ).get_angular_distribution_string_for_sim_telarray()

    # Result should be a string path to the generated table
    table_path = Path(result)
    assert str(table_path).endswith(".dat")
    assert table_path.exists()
    content = table_path.read_text().splitlines()
    assert content[0].startswith("# angle[deg] relative_intensity")
    # Expect 101 lines: header + 100 samples (0..max angle)
    assert len(content) == 101
    # Verify that width parameter was requested for Lambertian
    simulator_instance.calibration_model.get_parameter_value_with_unit.assert_called_once_with(
        "flasher_angular_distribution_width"
    )


def test__get_angular_distribution_string_for_sim_telarray_lambertian_failure(
    simulator_instance,
):
    # Mock calibration model values
    simulator_instance.calibration_model.get_parameter_value.side_effect = lambda name: (
        "Lambertian" if name == "flasher_angular_distribution" else None
    )

    # Mock _generate_lambertian_angular_distribution_table to raise OSError
    with patch.object(
        get_light_source_writer(simulator_instance),
        "generate_lambertian_angular_distribution_table",
        side_effect=OSError("Write failed"),
    ):
        result = get_light_source_writer(
            simulator_instance
        ).get_angular_distribution_string_for_sim_telarray()

    # Should return the token string "lambertian" and log a warning
    assert result == "lambertian"
    simulator_instance._logger.warning.assert_called_with(
        "Failed to write Lambertian angular distribution table: Write failed; using token instead."
    )


def test__get_pulse_shape_string_token(simulator_instance):
    # Test Gauss with width
    result = get_light_source_writer(simulator_instance).get_pulse_shape_string_token(
        "Gauss", 5.0, 0.0
    )
    assert result == "gauss:5.0"

    # Test with only shape (no width or exp)
    result = get_light_source_writer(simulator_instance).get_pulse_shape_string_token(
        "Line", 0.0, 0.0
    )
    assert result == "line"


def test__get_pulse_shape_string_token_exponential(simulator_instance):
    # Test exponential pulse shape with decay only
    result = get_light_source_writer(simulator_instance).get_pulse_shape_string_token(
        "Exponential", 0.0, 3.2
    )
    assert result == "exponential:3.2"


def test__get_pulse_shape_argument_for_sim_telarray_simple_shapes(simulator_instance):
    # Mock simple pulse shape
    simulator_instance.calibration_model.get_parameter_value.return_value = ["Gauss", 5.0, 0.0]

    result = get_light_source_writer(simulator_instance).get_pulse_shape_argument_for_sim_telarray()
    assert result == "gauss:5.0"

    # Mock Tophat shape
    simulator_instance.calibration_model.get_parameter_value.return_value = ["Tophat", 3.0, 0.0]

    result = get_light_source_writer(simulator_instance).get_pulse_shape_argument_for_sim_telarray()
    assert result == "simple:3.0"


def test__get_pulse_shape_argument_for_sim_telarray_gauss_exp_dat_file(
    simulator_instance, tmp_test_directory
):
    # Mock Gauss-Exponential pulse shape
    simulator_instance.calibration_model.get_parameter_value.return_value = [
        "Gauss-Exponential",
        2.0,
        6.0,
    ]

    # Mock telescope parameter
    simulator_instance.telescope_model.get_parameter_value.return_value = 40  # fadc_sum_bins

    # Configure IO handler
    pulse_dir = Path(tmp_test_directory) / "pulse_shapes"
    pulse_dir.mkdir(parents=True, exist_ok=True)
    io_mock = Mock()
    io_mock.get_output_directory.return_value = pulse_dir
    simulator_instance.io_handler = io_mock

    # Config for filename
    simulator_instance.light_emission_config = {
        "telescope": "LSTN-01",
        "light_source": "NectarCam",
    }

    with patch(
        "simtools.simtel.light_emission_config_writer."
        "simtel_file_writer.write_light_pulse_table_gauss_exp_conv"
    ) as mock_writer:
        result = get_light_source_writer(
            simulator_instance
        ).get_pulse_shape_argument_for_sim_telarray()

        # Should call the writer with correct params
        assert mock_writer.called
        kwargs = mock_writer.call_args.kwargs
        assert kwargs["width_ns"] == pytest.approx(2.0)
        assert kwargs["exp_decay_ns"] == pytest.approx(6.0)
        assert kwargs["fadc_sum_bins"] == 40

        # Should return path to DAT file
        assert result.endswith(".dat")
        assert "flasher_pulse_shape" in result


def test__get_pulse_shape_argument_for_sim_telarray_gauss_exp_failure(simulator_instance):
    # Mock Gauss-Exponential pulse shape
    simulator_instance.calibration_model.get_parameter_value.return_value = [
        "Gauss-Exponential",
        2.0,
        6.0,
    ]

    # Mock telescope parameter
    simulator_instance.telescope_model.get_parameter_value.return_value = 40

    # Configure IO handler to cause write failure
    io_mock = Mock()
    io_mock.get_output_directory.side_effect = OSError("Failed to create directory")
    simulator_instance.io_handler = io_mock

    simulator_instance.light_emission_config = {
        "telescope": "LSTN-01",
        "light_source": "NectarCam",
    }

    # Should log warning and return token string instead of raising
    result = get_light_source_writer(simulator_instance).get_pulse_shape_argument_for_sim_telarray()

    # Verify warning was logged
    simulator_instance._logger.warning.assert_called_once()
    assert "Failed to write pulse shape table" in simulator_instance._logger.warning.call_args[0][0]

    # Verify token string was returned instead of raising
    assert isinstance(result, str)
    assert result == "gauss-exponential"


def test__add_illuminator_command_options(simulator_instance):
    # Mock calibration model methods
    mock_wavelength = Mock()
    mock_wavelength.to.return_value.value = 450
    simulator_instance.calibration_model.get_parameter_value_with_unit.side_effect = [
        [1.0 * u.m, 2.0 * u.m, 3.0 * u.m],  # array_element_position_ground
        9.0 * u.m,  # illuminator_tower_height
        mock_wavelength,  # flasher_wavelength
    ]

    # Mock helper methods
    with (
        patch.object(
            get_light_source_writer(simulator_instance),
            "get_angular_distribution_string_for_sim_telarray",
            return_value="gauss:5.0",
        ),
        patch.object(
            get_light_source_writer(simulator_instance),
            "get_pulse_shape_argument_for_sim_telarray",
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

        result = get_light_source_writer(simulator_instance).add_illuminator_command_options()

        # Verify structure and key values
        assert isinstance(result, list)
        assert len(result) == 8
        assert result[0] == "-x 100.0"  # 1.0m -> 100.0cm
        assert result[1] == "-y 200.0"  # 2.0m -> 200.0cm
        assert result[2] == "-z 1200.0"  # ground altitude plus 9.0m tower height
        assert result[3] == "-d 0.1,0.2,0.3"  # pointing vector from _calibration_pointing_direction
        assert result[4] == "-n 1000000"  # flasher_photons
        assert result[5] == "-s 450"  # wavelength in nm
        assert result[6] == "-p square:2.0"  # pulse shape
        assert result[7] == "-a gauss:5.0"  # angular distribution


def test__add_illuminator_command_options_with_custom_position_and_pointing(simulator_instance):
    # Mock calibration model methods (only wavelength needed when position is provided)
    mock_wavelength = Mock()
    mock_wavelength.to.return_value.value = 380
    simulator_instance.calibration_model.get_parameter_value_with_unit.return_value = (
        mock_wavelength
    )

    # Mock helper methods
    with (
        patch.object(
            get_light_source_writer(simulator_instance),
            "get_angular_distribution_string_for_sim_telarray",
            return_value="uniform",
        ),
        patch.object(
            get_light_source_writer(simulator_instance),
            "get_pulse_shape_argument_for_sim_telarray",
            return_value="gauss",
        ),
    ):
        # Test 2: Custom light_source_position and light_source_pointing
        simulator_instance.light_emission_config = {
            "flasher_photons": 2000000,
            "light_source_position": [5.0 * u.m, 6.0 * u.m, 7.0 * u.m],
            "light_source_pointing": [0.5, 0.6, 0.7],
        }

        result = get_light_source_writer(simulator_instance).add_illuminator_command_options()

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


def test__add_flasher_command_options(simulator_instance):

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
            get_light_source_writer(simulator_instance),
            "get_angular_distribution_string_for_sim_telarray",
            return_value="gauss:2.5",
        ),
        patch.object(
            get_light_source_writer(simulator_instance),
            "get_pulse_shape_argument_for_sim_telarray",
            return_value="square:3.0",
        ),
        patch(
            "simtools.simtel.light_emission_config_writer.fiducial_radius_from_shape",
            return_value=86.6,
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

        result = get_light_source_writer(simulator_instance).add_flasher_command_options()

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


@pytest.mark.parametrize(
    ("number_of_events", "expected_events"),
    [
        ([100, 50, 20], "100,50,20"),
        (100, "100,100,100"),
    ],
    ids=["one-event-per-intensity", "single-event-broadcast"],
)
def test__add_flasher_command_options_multi_intensity(
    simulator_instance, number_of_events, expected_events
):
    _setup_multi_intensity_flasher_test(simulator_instance)

    with (
        patch.object(
            simulator_instance, "calculate_distance_focal_plane_calibration_device"
        ) as mock_distance,
        patch(
            "simtools.simtel.light_emission_config_writer.fiducial_radius_from_shape",
            return_value=90.0,
        ),
        patch.object(
            get_light_source_writer(simulator_instance),
            "get_angular_distribution_string_for_sim_telarray",
            return_value="gauss:2.5",
        ),
        patch.object(
            get_light_source_writer(simulator_instance),
            "get_pulse_shape_argument_for_sim_telarray",
            return_value="gauss:2.0",
        ),
    ):
        mock_distance_value = Mock()
        mock_distance_value.to.return_value.value = 1000.0
        mock_distance.return_value = mock_distance_value

        simulator_instance.light_emission_config = {
            "number_of_events": number_of_events,
            "flasher_photons": [1000000, 2000000, 3000000],
        }

        result = get_light_source_writer(simulator_instance).add_flasher_command_options()

        assert f"--events {expected_events}" in result
        assert "--photons 1000000,2000000,3000000" in result


def test__add_flasher_command_options_with_pulse_table(simulator_instance, tmp_test_directory):

    # Mock calibration model values
    params_with_unit = {
        "flasher_position": [1.0 * u.cm, 2.0 * u.cm],
        "flasher_wavelength": 450.0 * u.nm,
        # Provide full 3-element pulse shape so writer path can read width/exp
        "flasher_pulse_shape": ["Gauss-Exponential", 2.0, 6.0],
    }
    simulator_instance.calibration_model.get_parameter_value_with_unit.side_effect = lambda name: (
        params_with_unit.get(name)
    )

    # Provide specific returns for plain-valued params used inside the call
    plain_params = {
        "flasher_bunch_size": 8000,
        "flasher_angular_distribution": "gaussian",
        "flasher_pulse_shape": ["Gauss-Exponential", 2.0, 6.0],
    }
    simulator_instance.calibration_model.get_parameter_value.side_effect = lambda name: (
        plain_params.get(name)
    )

    # Mock telescope values
    mock_diameter = Mock()
    mock_diameter.to.return_value.value = 180.0
    simulator_instance.telescope_model.get_parameter_value_with_unit.return_value = mock_diameter
    simulator_instance.telescope_model.get_parameter_value.side_effect = lambda key: (
        40 if key == "fadc_sum_bins" else "hexagonal"
    )

    # Mock distance and helpers
    with (
        patch.object(
            simulator_instance, "calculate_distance_focal_plane_calibration_device"
        ) as mock_distance,
        patch(
            "simtools.simtel.light_emission_config_writer.fiducial_radius_from_shape",
            return_value=90.0,
        ),
        patch(
            "simtools.simtel.light_emission_config_writer."
            "simtel_file_writer.write_light_pulse_table_gauss_exp_conv"
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

        result = get_light_source_writer(simulator_instance).add_flasher_command_options()

        # Writer called with expected numeric values in ns
        assert mock_writer.called
        kwargs = mock_writer.call_args.kwargs
        assert np.isclose(kwargs["width_ns"], 2.0)
        assert np.isclose(kwargs["exp_decay_ns"], 6.0)
        # Command should reference a pulse table path
        assert any(
            str(item).startswith("--lightpulse ") and str(item).endswith(".dat") for item in result
        )


def test__add_flasher_command_options_invalid_gauss_exponential_width(simulator_instance):

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
    simulator_instance.calibration_model.get_parameter_value.side_effect = lambda k: (
        8000 if k == "flasher_bunch_size" else ["Gauss-Exponential", 0.0, 5.0]
    )

    # Telescope minimal mocks
    mock_diameter = Mock()
    mock_diameter.to.return_value.value = 200.0
    simulator_instance.telescope_model.get_parameter_value_with_unit.return_value = mock_diameter
    simulator_instance.telescope_model.get_parameter_value.side_effect = lambda k: (
        40 if k == "fadc_sum_bins" else "hexagonal"
    )

    simulator_instance.light_emission_config = {"number_of_events": 1, "flasher_photons": 100}

    # Bypass geometry shape validation to exercise Gauss-Exponential parameter check
    with (
        patch(
            "simtools.simtel.light_emission_config_writer.fiducial_radius_from_shape",
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
            match=(
                "Gauss-Exponential pulse shape requires positive width and exponential decay values"
            ),
        ):
            get_light_source_writer(simulator_instance).add_flasher_command_options()


def test__add_flasher_command_options_invalid_gauss_exponential_decay(simulator_instance):

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
    simulator_instance.calibration_model.get_parameter_value.side_effect = lambda k: (
        4000 if k == "flasher_bunch_size" else ["Gauss-Exponential", 2.0, 0.0]
    )

    # Telescope minimal mocks
    mock_diameter = Mock()
    mock_diameter.to.return_value.value = 160.0
    simulator_instance.telescope_model.get_parameter_value_with_unit.return_value = mock_diameter
    simulator_instance.telescope_model.get_parameter_value.side_effect = lambda k: (
        40 if k == "fadc_sum_bins" else "hexagonal"
    )

    simulator_instance.light_emission_config = {"number_of_events": 1, "flasher_photons": 100}

    # Bypass geometry shape validation to exercise Gauss-Exponential parameter check
    with (
        patch(
            "simtools.simtel.light_emission_config_writer.fiducial_radius_from_shape",
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
            match=(
                "Gauss-Exponential pulse shape requires positive width and exponential decay values"
            ),
        ):
            get_light_source_writer(simulator_instance).add_flasher_command_options()


def test__get_light_source_command(simulator_instance):
    # Test flat_fielding type
    simulator_instance.light_emission_config = {"light_source_type": "flat_fielding"}

    with patch.object(
        get_light_source_writer(simulator_instance),
        "add_flasher_command_options",
        return_value=["flasher_option"],
    ) as mock_flasher:
        result = get_light_source_writer(simulator_instance).get_light_source_command()
        assert result == ["flasher_option"]
        mock_flasher.assert_called_once()

    # Test illuminator type
    simulator_instance.light_emission_config = {"light_source_type": "illuminator"}

    with patch.object(
        get_light_source_writer(simulator_instance),
        "add_illuminator_command_options",
        return_value=["illuminator_option"],
    ) as mock_illuminator:
        result = get_light_source_writer(simulator_instance).get_light_source_command()
        assert result == ["illuminator_option"]
        mock_illuminator.assert_called_once()

    # Test unknown type raises ValueError
    simulator_instance.light_emission_config = {"light_source_type": "unknown_type"}

    with pytest.raises(ValueError, match="Unknown light_source_type 'unknown_type'"):
        get_light_source_writer(simulator_instance).get_light_source_command()


def test__get_site_command(simulator_instance, tmp_test_directory):

    # Mock altitude value
    mock_altitude = Mock()
    mock_altitude.to.return_value.value = 2200

    # Test ff-1m app (flasher path)
    with (
        patch.object(
            get_light_source_writer(simulator_instance),
            "prepare_flasher_atmosphere_files",
            return_value="atm_id_123",
        ) as mock_atmo,
        patch("simtools.simtel.light_emission_config_writer.settings") as mock_settings,
    ):
        mock_settings.config.sim_telarray_path = Path("/mock/simtel/sim_telarray")
        result = get_light_source_writer(simulator_instance).get_site_command(
            "ff-1m", "/config/dir", mock_altitude
        )

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
    with (
        patch.object(
            simulator_instance,
            "write_telescope_position_file",
            return_value=f"{tmp_test_directory}/telpos.txt",
        ) as mock_telpos,
        patch.object(
            simulator_instance,
            "get_illuminator_position",
            return_value=[1.0 * u.m, 2.0 * u.m, 3.0 * u.m],
        ),
    ):
        # Default-down pointing: do not use telpos file.
        simulator_instance.light_emission_config = {
            "light_source_type": "illuminator",
            "light_source_pointing": [0.0, 0.0, -1.0],
        }
        result = get_light_source_writer(simulator_instance).get_site_command(
            "other-app", "/config/dir", mock_altitude
        )
        assert result == ["-h  2200 "]
        mock_telpos.assert_not_called()

        # Non-default pointing: use telpos file.
        simulator_instance.light_emission_config = {
            "light_source_type": "illuminator",
            "light_source_pointing": [0.0, 0.0, -0.9],
        }
        result = get_light_source_writer(simulator_instance).get_site_command(
            "other-app", "/config/dir", mock_altitude
        )
        assert result == ["-h  2200 ", f"--telpos-file {tmp_test_directory}/telpos.txt"]
        assert mock_telpos.call_count == 1


def test__make_light_emission_script(simulator_instance):
    simulator_instance.output_directory = "/output"
    simulator_instance.label = "test_label"

    simulator_instance.telescope_model.config_file_directory = Path("/config/dir")

    # Mock site model
    mock_obs_level = Mock()
    mock_obs_level.to.return_value.value = 2200
    simulator_instance.site_model.get_parameter_value_with_unit.return_value = mock_obs_level
    simulator_instance.site_model.model_version = "test_version"

    # Mock helper methods
    with (
        patch.object(
            get_light_source_writer(simulator_instance),
            "get_light_emission_application_name",
            return_value="ff-1m",
        ) as mock_app_name,
        patch.object(
            get_light_source_writer(simulator_instance),
            "get_site_command",
            return_value=["-I.", "--altitude 2200"],
        ) as mock_site,
        patch.object(
            get_light_source_writer(simulator_instance),
            "get_light_source_command",
            return_value=["--photons 1000000"],
        ) as mock_light_source,
        patch("simtools.simtel.light_emission_config_writer.settings") as mock_settings,
    ):
        # Test flat_fielding (no atmospheric profile)
        simulator_instance.light_emission_config = {"light_source_type": "flat_fielding"}
        mock_settings.config.sim_telarray_path = Path("/mock/simtel/sim_telarray")

        # Mock runner_service.get_file_name to return a string path
        simulator_instance.runner_service.get_file_name.return_value = Path("/output/ff-1m.log.gz")

        result = get_light_source_writer(simulator_instance).make_command("/output/ff-1m.iact.gz")

        expected = (
            "/mock/simtel/sim_telarray/LightEmission/ff-1m -I. --altitude 2200 "
            "--photons 1000000 -o /output/ff-1m.iact.gz 2>&1 | gzip > /output/ff-1m.log.gz\n"
        )
        assert result == expected

        # Verify method calls
        mock_app_name.assert_called_once()
        mock_site.assert_called_once_with("ff-1m", Path("/config/dir"), mock_obs_level)
        mock_light_source.assert_called_once()

    # Test illuminator (with atmospheric profile)
    simulator_instance.site_model.get_parameter_value.return_value = "atm_profile.dat"
    simulator_instance.telescope_model.get_parameter_value.side_effect = InvalidModelParameterError(
        "The atmospheric profile belongs to the site model."
    )

    with (
        patch.object(
            get_light_source_writer(simulator_instance),
            "get_light_emission_application_name",
            return_value="illuminator-app",
        ),
        patch.object(
            get_light_source_writer(simulator_instance),
            "get_site_command",
            return_value=["-h 2200"],
        ),
        patch.object(
            get_light_source_writer(simulator_instance),
            "get_light_source_command",
            return_value=["-x 100", "-y 200"],
        ),
        patch("simtools.simtel.light_emission_config_writer.settings") as mock_settings,
    ):
        simulator_instance.light_emission_config = {"light_source_type": "illuminator"}
        mock_settings.config.sim_telarray_path = Path("/mock/simtel/sim_telarray")

        # Mock runner_service.get_file_name to return a string path
        simulator_instance.runner_service.get_file_name.return_value = Path(
            "/output/illuminator-app.log.gz"
        )

        result = get_light_source_writer(simulator_instance).make_command(
            "/output/illuminator-app.iact.gz"
        )

        expected = (
            "/mock/simtel/sim_telarray/LightEmission/illuminator-app -h 2200 "
            "-x 100 -y 200 -A /config/dir/atm_profile.dat "
            "-o /output/illuminator-app.iact.gz 2>&1 | gzip > /output/illuminator-app.log.gz\n"
        )
        assert result == expected
        simulator_instance.site_model.get_parameter_value.assert_called_once_with(
            "atmospheric_profile"
        )
        simulator_instance.telescope_model.get_parameter_value.assert_not_called()


def test__prepare_flasher_atmosphere_files(simulator_instance):
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
        result = get_light_source_writer(simulator_instance).prepare_flasher_atmosphere_files(
            config_directory
        )

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
        result = get_light_source_writer(simulator_instance).prepare_flasher_atmosphere_files(
            config_directory, model_id=5
        )

        # Should return the custom model_id
        assert result == 5
        assert mock_symlink.call_count == 2


def test__prepare_flasher_atmosphere_files_with_existing_files(simulator_instance):
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
        result = get_light_source_writer(simulator_instance).prepare_flasher_atmosphere_files(
            config_directory
        )

        # Should unlink existing files before creating new ones
        assert mock_unlink.call_count == 2
        assert mock_symlink.call_count == 2
        assert result == 1


def test__prepare_flasher_atmosphere_files_symlink_fallback_to_copy(simulator_instance):
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
        result = get_light_source_writer(simulator_instance).prepare_flasher_atmosphere_files(
            config_directory
        )

        # Should fall back to copy when symlink fails
        assert mock_copy.call_count == 2
        assert result == 1


def test__prepare_flasher_atmosphere_files_copy_also_fails(simulator_instance):
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
        result = get_light_source_writer(simulator_instance).prepare_flasher_atmosphere_files(
            config_directory
        )

        # Should log warnings but still return model_id
        assert simulator_instance._logger.warning.call_count == 2
        assert result == 1


def test__get_light_emission_application_name(simulator_instance):
    # Test flat_fielding type returns ff-1m
    simulator_instance.light_emission_config = {"light_source_type": "flat_fielding"}
    result = get_light_source_writer(simulator_instance).get_light_emission_application_name()
    assert result == "ff-1m"

    # Test any other type returns xyzls (default)
    simulator_instance.light_emission_config = {"light_source_type": "illuminator"}
    result = get_light_source_writer(simulator_instance).get_light_emission_application_name()
    assert result == "xyzls"


def test__get_angular_distribution_string_for_sim_telarray_isotropic(simulator_instance):
    simulator_instance.calibration_model.get_parameter_value.return_value = "Isotropic"

    # Even if width is available (though it shouldn't be for isotropic), it should be ignored
    mock_width = Mock()
    mock_width.to.return_value.value = 10.0
    simulator_instance.calibration_model.get_parameter_value_with_unit.return_value = mock_width

    result = get_light_source_writer(
        simulator_instance
    ).get_angular_distribution_string_for_sim_telarray()
    assert result == "isotropic"

    # Verify width was NOT requested: the implementation returns early for isotropic distributions
    # before attempting to fetch the width via get_parameter_value_with_unit.
    simulator_instance.calibration_model.get_parameter_value_with_unit.assert_not_called()


def _setup_multi_intensity_flasher_test(simulator_instance):
    """Set up common mocks for multi-intensity flasher command-option tests."""

    def mock_get_param_with_unit(name):
        if name == "flasher_position":
            return [1.0 * u.cm, 1.0 * u.cm]
        if name == "flasher_wavelength":
            return 450.0 * u.nm
        return None

    simulator_instance.calibration_model.get_parameter_value_with_unit.side_effect = (
        mock_get_param_with_unit
    )
    simulator_instance.calibration_model.get_parameter_value.side_effect = lambda name: (
        4000 if name == "flasher_bunch_size" else ["Gauss", 0.0, 0.0]
    )

    mock_diameter = Mock()
    mock_diameter.to.return_value.value = 180.0
    simulator_instance.telescope_model.get_parameter_value_with_unit.return_value = mock_diameter
    simulator_instance.telescope_model.get_parameter_value.return_value = "hexagonal"
