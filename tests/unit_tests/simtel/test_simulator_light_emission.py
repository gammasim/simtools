#!/usr/bin/python3

from pathlib import Path
from unittest.mock import ANY, Mock, patch

import astropy.units as u
import numpy as np
import pytest

from simtools.simtel.simulator_light_emission import SimulatorLightEmission
from simtools.simulation.configuration import get_light_source_writer


def test__make_simtel_script_bypass_optics_condition(simulator_instance):
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
            get_light_source_writer(simulator_instance),
            "get_light_emission_application_name",
            return_value="ff-1m",
        ),
        patch("simtools.simtel.simulator_light_emission.settings") as mock_settings,
    ):
        mock_settings.config.sim_telarray_exe = "/mock/simtel/bin/sim_telarray"
        mock_settings.config.sim_telarray_path = Path("/mock/simtel")
        # Test flat_fielding - should include Bypass_Optics
        simulator_instance.light_emission_config = {"light_source_type": "flat_fielding"}

        options = simulator_instance._make_simtel_script()
        assert "Bypass_Optics=1" in options
        assert "-I/mock/simtel/bin/sim_telarray" not in options

        # Test illuminator - should NOT include Bypass_Optics
        simulator_instance.light_emission_config = {"light_source_type": "illuminator"}

        options = simulator_instance._make_simtel_script()
        assert "Bypass_Optics=1" not in options


def test__make_simtel_script_uses_generated_atmospheric_transmission_file(simulator_instance):
    simulator_instance.telescope_model.config_file_path = "/mock/config/CTAO-MSTN-04.cfg"
    simulator_instance.site_model.get_parameter_value_with_unit.return_value = 2200 * u.m
    simulator_instance.site_model.get_parameter_value.return_value = (
        "atmospheric_transmission-2.0.0.ecsv"
    )
    simulator_instance.light_emission_config = {"light_source_type": "illuminator"}

    with (
        patch.object(simulator_instance, "_get_telescope_pointing", return_value=[0, 0]),
        patch("simtools.simtel.simulator_light_emission.settings") as mock_settings,
    ):
        mock_settings.config.sim_telarray_exe = "/mock/simtel/bin/sim_telarray"

        script = simulator_instance._make_simtel_script()

    assert "-C atmospheric_transmission=atmospheric_transmission-CTAO-MSTN-04.dat" in script


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


def test__add_flasher_command_options_event_length_mismatch_raises(simulator_instance):

    simulator_instance.light_emission_config = {
        "number_of_events": [100, 50],
        "flasher_photons": [1000000, 2000000, 3000000],
    }

    with pytest.raises(
        ValueError,
        match=(
            r"Invalid number_of_events list length\. "
            r"Use one value or one value per photon intensity"
        ),
    ):
        simulator_instance.build_flasher_event_and_photon_sequences()


def test_get_illuminator_pointing_vector_computed_from_position(simulator_instance):
    simulator_instance.light_emission_config = {}
    simulator_instance.calibration_model.get_parameter_value_with_unit.side_effect = [
        [10.0 * u.m, 20.0 * u.m, 30.0 * u.m],
        9.0 * u.m,
    ]

    with patch.object(
        simulator_instance,
        "_calibration_pointing_direction",
        return_value=([0.1, 0.2, 0.3], []),
    ) as mock_pointing:
        vec = simulator_instance.get_illuminator_pointing_vector()

    assert vec == [0.1, 0.2, 0.3]
    mock_pointing.assert_called_once()


def test_get_illuminator_position_adds_tower_height(simulator_instance):
    simulator_instance.calibration_model.get_parameter_value_with_unit.side_effect = [
        [10.0 * u.m, 20.0 * u.m, 30.0 * u.m],
        9.0 * u.m,
    ]

    position = simulator_instance.get_illuminator_position()

    assert position == (10.0 * u.m, 20.0 * u.m, 39.0 * u.m)
    assert simulator_instance.calibration_model.get_parameter_value_with_unit.call_args_list == [
        (("array_element_position_ground",),),
        (("illuminator_tower_height",),),
    ]


def test_get_illuminator_position_keeps_configured_position(simulator_instance):
    configured_position = [10.0 * u.m, 20.0 * u.m, 39.0 * u.m]
    simulator_instance.light_emission_config = {"light_source_position": configured_position}

    position = simulator_instance.get_illuminator_position()

    assert position == configured_position
    simulator_instance.calibration_model.get_parameter_value_with_unit.assert_not_called()


def test_uses_telescope_position_file_rule(simulator_instance):
    assert simulator_instance.uses_telescope_position_file([0.0, 0.0, -1.0]) is False
    assert simulator_instance.uses_telescope_position_file([0.0, 0.0, -0.9]) is True
    assert simulator_instance.uses_telescope_position_file([1.0, 0.0, 0.0]) is True
    assert simulator_instance.uses_telescope_position_file(None) is True
    assert simulator_instance.uses_telescope_position_file(["a", "b", "c"]) is True


def test_prepare_run(simulator_instance, tmp_test_directory):
    # Setup mocks
    simulator_instance.output_directory = Path(tmp_test_directory) / "output"
    simulator_instance.light_emission_config = {"light_source_type": "illuminator"}

    script_dir = Path(tmp_test_directory) / "output" / "scripts"
    script_dir.mkdir(parents=True, exist_ok=True)
    script_path = script_dir / "xyzls-light_emission.sh"

    # Mock submission_files.get_file_name to return the script path
    def job_files_get_file_name_side_effect(file_type):
        if file_type == "sub_script":
            return script_path
        return Path(tmp_test_directory) / "output" / f"{file_type}.tmp"

    simulator_instance.submission_files.get_file_name.side_effect = (
        job_files_get_file_name_side_effect
    )

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
            get_light_source_writer(simulator_instance),
            "make_command",
            return_value="light_emission_cmd",
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
    # Setup
    simulator_instance.output_directory = Path(tmp_test_directory) / "output"
    simulator_instance.output_directory.mkdir(parents=True, exist_ok=True)

    # Mock the methods called by simulate
    mock_script_path = Path(tmp_test_directory) / "output" / "scripts" / "test_script.sh"
    mock_script_path.parent.mkdir(parents=True, exist_ok=True)
    mock_output_file = Path(tmp_test_directory) / "output" / "test_output.simtel.gz"

    # Setup submission_files mock to return the script path
    def job_files_get_file_name_side_effect(file_type):
        if file_type == "sub_script":
            return mock_script_path
        return Path(tmp_test_directory) / "output" / f"{file_type}.tmp"

    simulator_instance.submission_files.get_file_name.side_effect = (
        job_files_get_file_name_side_effect
    )

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


def test__initialize_light_emission_configuration(simulator_instance):

    # Mock calibration model responses
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


def test__initialize_light_emission_configuration_with_flasher_photons_override(
    simulator_instance,
):

    def mock_get_parameter_value(param_name):
        if param_name == "flasher_type":
            return "flat_fielding"
        if param_name == "flasher_photons":
            return 1234567
        return None

    simulator_instance.calibration_model.get_parameter_value.side_effect = mock_get_parameter_value

    config = {"flasher_photons": 1234567}
    result = simulator_instance._initialize_light_emission_configuration(config)

    assert result["light_source_type"] == "flat_fielding"
    assert result["flasher_photons"] == pytest.approx(1234567)
    simulator_instance.calibration_model.overwrite_model_parameter.assert_called_once_with(
        "flasher_photons", 1234567
    )


def test__initialize_light_emission_configuration_preserves_run_mode(simulator_instance):

    def mock_get_parameter_value(param_name):
        if param_name == "flasher_type":
            return "illuminator"
        if param_name == "flasher_photons":
            return 5e6
        return None

    simulator_instance.calibration_model.get_parameter_value.side_effect = mock_get_parameter_value

    result = simulator_instance._initialize_light_emission_configuration(
        {"run_mode": "full_simulation"}
    )

    assert result["light_source_type"] == "illuminator"
    assert result["run_mode"] == "full_simulation"
    assert result["flasher_photons"] == pytest.approx(5e6)


def test__initialize_light_emission_configuration_with_position(simulator_instance):
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
        simulator_light_emission_instance = SimulatorLightEmission(config, label="test_label")

        # Verify initialization
        assert hasattr(simulator_light_emission_instance, "_logger")
        assert hasattr(simulator_light_emission_instance, "io_handler")
        assert hasattr(simulator_light_emission_instance, "telescope_model")
        assert hasattr(simulator_light_emission_instance, "site_model")
        assert hasattr(simulator_light_emission_instance, "calibration_model")
        assert hasattr(simulator_light_emission_instance, "light_emission_config")

        # Verify models were initialized correctly
        mock_init_models.assert_called_once_with(
            label="test_label_LSTN-01",
            site="North",
            telescope_name="LSTN-01",
            calibration_device_name="calibration_device",
            calibration_device_type=None,
            model_version="6.0.0",
            model_reader=simulator_light_emission_instance.model_reader,
        )

        # Verify telescope model config file was written
        mock_telescope_model.write_sim_telarray_config_file.assert_called_once_with(
            additional_models=mock_site_model
        )


def test___init___passes_custom_model_directory(tmp_test_directory):
    """A configured model directory is passed to the simulation models."""
    io_handler_path = "simtools.simtel.simulator_light_emission.io_handler.IOHandler"
    models_path = "simtools.simtel.simulator_light_emission.initialize_simulation_models"
    model_directory = Path(tmp_test_directory) / "model" / "isolated"

    with patch(io_handler_path), patch(models_path) as mock_init_models:
        mock_init_models.return_value = (Mock(), Mock(), Mock())
        SimulatorLightEmission(
            {
                "site": "North",
                "telescope": "LSTN-01",
                "light_source": "calibration_device",
                "model_version": "6.0.0",
                "model_directory": model_directory,
            },
            label="test_label",
        )

    assert mock_init_models.call_args.kwargs["model_directory"] == model_directory


def test___init___with_wavelength(tmp_test_directory):

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
        # Mock wavelength validation to return allowed wavelengths
        mock_calibration_model.get_parameter_value_with_unit.return_value = [
            355 * u.nm,
            473 * u.nm,
        ]
        mock_init_models.return_value = (
            mock_telescope_model,
            mock_site_model,
            mock_calibration_model,
        )

        # Test configuration with wavelength
        config = {
            "site": "North",
            "telescope": "LSTN-01",
            "light_source": "calibration_device",
            "model_version": "6.0.0",
            "wavelength": 355 * u.nm,
        }

        # Create instance
        _ = SimulatorLightEmission(config, label="test_label")

        # Verify models were initialized with label including telescope and wavelength
        mock_init_models.assert_called_once_with(
            label="test_label_LSTN-01_355nm",
            site="North",
            telescope_name="LSTN-01",
            calibration_device_name="calibration_device",
            calibration_device_type=None,
            model_version="6.0.0",
            model_reader=ANY,
        )


def test__get_telescope_pointing(simulator_instance):
    # Test flat_fielding type returns (0.0, 0.0)
    simulator_instance.light_emission_config = {"light_source_type": "flat_fielding"}
    result = simulator_instance._get_telescope_pointing()
    assert result == (0.0, 0.0)

    # Test with light_source_position keeps fixed vertical-up telescope pointing
    simulator_instance.light_emission_config = {
        "light_source_type": "illuminator",
        "light_source_position": [1.0 * u.m, 2.0 * u.m, 3.0 * u.m],
    }
    with patch.object(
        simulator_instance, "_calibration_pointing_direction"
    ) as mock_pointing_for_pos:
        result = simulator_instance._get_telescope_pointing()
        assert result == (0.0, 0.0)
        mock_pointing_for_pos.assert_not_called()
        simulator_instance._logger.info.assert_called_with(
            "Using fixed (vertical up) telescope pointing."
        )

    # Test default case uses _calibration_pointing_direction
    simulator_instance.light_emission_config = {"light_source_type": "illuminator"}
    simulator_instance._logger.reset_mock()

    with (
        patch.object(
            simulator_instance,
            "_calibration_pointing_direction",
            return_value=(None, [45.0, 180.0, 90.0, 0.0]),
        ) as mock_pointing,
        patch.object(
            simulator_instance,
            "get_illuminator_position",
            return_value=[10.0 * u.m, 20.0 * u.m, 39.0 * u.m],
        ) as mock_position,
    ):
        result = simulator_instance._get_telescope_pointing()

        # Should return tel_theta, tel_phi (first two angles)
        assert result == (45.0, 180.0)
        mock_pointing.assert_called_once_with(10.0 * u.m, 20.0 * u.m, 39.0 * u.m)
        mock_position.assert_called_once_with()
        simulator_instance._logger.info.assert_not_called()


def test_write_telescope_position_file(simulator_instance, tmp_test_directory):
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

    def mock_get_param_with_unit(name):
        if name == "array_element_position_ground":
            return [mock_x, mock_y, mock_z]
        if name == "axes_offsets":
            return [0.0 * u.m, 0.0 * u.m]
        if name == "telescope_sphere_radius":
            return mock_radius
        return None

    simulator_instance.telescope_model.get_parameter_value_with_unit.side_effect = (
        mock_get_param_with_unit
    )

    # Use real temporary directory for file writing
    mock_output_dir = Path(tmp_test_directory)
    mock_output_dir.mkdir(parents=True, exist_ok=True)
    simulator_instance.io_handler.get_output_directory.return_value = mock_output_dir
    simulator_instance.light_emission_config = {
        "telescope": "MSTS-04",
        "light_source": "ILLS-02",
    }
    expected_file = mock_output_dir / "telescope_position_MSTS-04_ILLS-02.dat"

    # Call the method
    result = simulator_instance.write_telescope_position_file(
        illuminator_position=[0.0 * u.m, 0.0 * u.m, 1000.0 * u.m]
    )

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


def test__get_telescope_position_ground_with_axis_offset_rotates_with_azimuth(simulator_instance):

    def mock_get_param_with_unit(name):
        if name == "array_element_position_ground":
            return [0.0 * u.m, 0.0 * u.m, 0.0 * u.m]
        if name == "axes_offsets":
            return [1.0 * u.m, 0.0 * u.m]
        return None

    simulator_instance.telescope_model.get_parameter_value_with_unit.side_effect = (
        mock_get_param_with_unit
    )

    x_tel, y_tel, _ = simulator_instance._get_telescope_position_ground_with_axis_offset(
        x_cal=100.0 * u.m,
        y_cal=0.0 * u.m,
        z_cal=100.0 * u.m,
    )
    assert x_tel.to(u.m).value == pytest.approx(1.0, abs=1e-6)
    assert y_tel.to(u.m).value == pytest.approx(0.0, abs=1e-6)

    x_tel, y_tel, _ = simulator_instance._get_telescope_position_ground_with_axis_offset(
        x_cal=0.0 * u.m,
        y_cal=100.0 * u.m,
        z_cal=100.0 * u.m,
    )
    assert x_tel.to(u.m).value == pytest.approx(0.0, abs=1e-6)
    assert y_tel.to(u.m).value == pytest.approx(1.0, abs=1e-6)


def test__get_telescope_position_ground_with_axis_offset_uses_second_offset_component(
    simulator_instance,
):

    def mock_get_param_with_unit(name):
        if name == "array_element_position_ground":
            return [0.0 * u.m, 0.0 * u.m, 0.0 * u.m]
        if name == "axes_offsets":
            return [0.0 * u.m, 1.0 * u.m]
        return None

    simulator_instance.telescope_model.get_parameter_value_with_unit.side_effect = (
        mock_get_param_with_unit
    )

    # Source on +x axis: perpendicular direction from transformation points along +z.
    x_tel, y_tel, z_tel = simulator_instance._get_telescope_position_ground_with_axis_offset(
        x_cal=100.0 * u.m,
        y_cal=0.0 * u.m,
        z_cal=0.0 * u.m,
    )

    assert x_tel.to(u.m).value == pytest.approx(0.0, abs=1e-6)
    assert y_tel.to(u.m).value == pytest.approx(0.0, abs=1e-6)
    assert z_tel.to(u.m).value == pytest.approx(1.0, abs=1e-6)


def test__calibration_pointing_direction(simulator_instance):
    import numpy as np

    # Mock calibration device position at origin
    cal_x, cal_y, cal_z = 0 * u.m, 0 * u.m, 0 * u.m
    simulator_instance.calibration_model.get_parameter_value_with_unit.return_value = [
        cal_x,
        cal_y,
        cal_z,
    ]

    # Mock telescope position at (10, 0, 10) meters and no horizontal offset
    tel_x, tel_y, tel_z = 10 * u.m, 0 * u.m, 10 * u.m

    def mock_get_telescope_param_with_unit(name):
        if name == "array_element_position_ground":
            return [tel_x, tel_y, tel_z]
        if name == "axes_offsets":
            return [0.0 * u.m, 0.0 * u.m]
        return None

    simulator_instance.telescope_model.get_parameter_value_with_unit.side_effect = (
        mock_get_telescope_param_with_unit
    )

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
    telescope_calls = [
        call.args[0]
        for call in simulator_instance.telescope_model.get_parameter_value_with_unit.call_args_list
    ]
    assert "array_element_position_ground" in telescope_calls
    assert "axes_offsets" in telescope_calls


def test__calibration_pointing_direction_with_custom_params(simulator_instance):
    import numpy as np

    # Mock telescope position and no horizontal offset
    tel_x, tel_y, tel_z = 5 * u.m, 5 * u.m, 0 * u.m

    def mock_get_telescope_param_with_unit(name):
        if name == "array_element_position_ground":
            return [tel_x, tel_y, tel_z]
        if name == "axes_offsets":
            return [0.0 * u.m, 0.0 * u.m]
        return None

    simulator_instance.telescope_model.get_parameter_value_with_unit.side_effect = (
        mock_get_telescope_param_with_unit
    )

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
    telescope_calls = [
        call.args[0]
        for call in simulator_instance.telescope_model.get_parameter_value_with_unit.call_args_list
    ]
    assert "array_element_position_ground" in telescope_calls
    assert "axes_offsets" in telescope_calls


def test_validate_simulations_success(simulator_instance, tmp_test_directory):
    output_file = Path(tmp_test_directory) / "output.iact"
    output_file.write_text("test data", encoding="utf-8")

    simulator_instance.runner_service.get_file_name.return_value = output_file

    # Mock the validator to avoid actual validation
    with patch(
        "simtools.simtel.simulator_light_emission.simtel_output_validator.validate_sim_telarray"
    ):
        result = simulator_instance.validate_simulations()

    # validate_simulations doesn't return anything (returns None on success)
    assert result is None


def test__initialize_light_emission_configuration_with_invalid_wavelength(simulator_instance):

    # Mock calibration model responses
    def mock_get_parameter_value(param_name):
        if param_name == "flasher_type":
            return "illuminator"
        if param_name == "flasher_photons":
            return 5e6
        return None

    simulator_instance.calibration_model.get_parameter_value.side_effect = mock_get_parameter_value

    # Mock the get_parameter_value_with_unit to return allowed wavelengths
    allowed_wavelengths = [266 * u.nm, 355 * u.nm, 473 * u.nm, 532 * u.nm]
    simulator_instance.calibration_model.get_parameter_value_with_unit.return_value = (
        allowed_wavelengths
    )

    # Test with an invalid wavelength (400 nm - not in the allowed list)
    config = {"wavelength": 400.0 * u.nm}

    with pytest.raises(
        ValueError,
        match=r"Wavelength 400\.0 nm is not supported.*Allowed wavelengths are.*",
    ):
        simulator_instance._initialize_light_emission_configuration(config)


def test_get_available_wavelengths(simulator_instance):
    # Setup mock wavelengths
    mock_wavelengths = [266 * u.nm, 355 * u.nm, 473 * u.nm, 532 * u.nm]
    simulator_instance.calibration_model.get_parameter_value_with_unit.return_value = (
        mock_wavelengths
    )

    # Call method
    wavelengths = simulator_instance.get_available_wavelengths()

    # Verify it returns the wavelengths from the calibration model
    assert wavelengths == mock_wavelengths
    simulator_instance.calibration_model.get_parameter_value_with_unit.assert_called_once_with(
        "flasher_wavelength"
    )


def test_get_available_wavelengths_from_config():
    # Setup mock calibration model
    mock_calibration_model = Mock()
    mock_calibration_model.get_parameter_value_with_unit.return_value = [
        266 * u.nm,
        355 * u.nm,
        473 * u.nm,
        532 * u.nm,
    ]

    # Config
    config = {
        "site": "North",
        "light_source": "ILLN-01",
        "model_version": "7.0.0",
    }

    # Patch both imports that happen inside the static method
    with (
        patch(
            "simtools.simtel.simulator_light_emission.CalibrationModel",
            return_value=mock_calibration_model,
        ) as mock_calibration_class,
        patch("simtools.simtel.simulator_light_emission.read_overwrite_model_parameter_dict"),
    ):
        # Call static method
        wavelengths = SimulatorLightEmission.get_available_wavelengths_from_config(config)

        # Verify it returns all 4 wavelengths
        assert len(wavelengths) == 4
        assert wavelengths[0] == 266 * u.nm
        assert wavelengths[1] == 355 * u.nm
        assert wavelengths[2] == 473 * u.nm
        assert wavelengths[3] == 532 * u.nm

        # Verify CalibrationModel was constructed correctly (no telescope/site models)
        mock_calibration_class.assert_called_once()
        call_kwargs = mock_calibration_class.call_args[1]
        assert call_kwargs["site"] == "North"
        assert call_kwargs["calibration_device_model_name"] == "ILLN-01"
        assert call_kwargs["model_version"] == "7.0.0"
        assert "temp_wavelength_query" in call_kwargs["label"]

        mock_calibration_model.get_parameter_value_with_unit.assert_called_once_with(
            "flasher_wavelength"
        )


def test_get_available_wavelengths_from_config_missing_keys():
    # Config missing required keys
    config_missing_site = {
        "light_source": "ILLN-01",
        "model_version": "7.0.0",
    }

    config_missing_light_source = {
        "site": "North",
        "model_version": "7.0.0",
    }

    config_missing_version = {
        "site": "North",
        "light_source": "ILLN-01",
    }

    # Should raise ValueError for each missing key
    with pytest.raises(ValueError, match="Missing required configuration keys"):
        SimulatorLightEmission.get_available_wavelengths_from_config(config_missing_site)

    with pytest.raises(ValueError, match="Missing required configuration keys"):
        SimulatorLightEmission.get_available_wavelengths_from_config(config_missing_light_source)

    with pytest.raises(ValueError, match="Missing required configuration keys"):
        SimulatorLightEmission.get_available_wavelengths_from_config(config_missing_version)
