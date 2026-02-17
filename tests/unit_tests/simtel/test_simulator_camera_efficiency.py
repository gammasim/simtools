#!/usr/bin/python3

import logging
from pathlib import Path

import astropy.units as u
import pytest

from simtools.camera.camera_efficiency import CameraEfficiency
from simtools.simtel.simulator_camera_efficiency import SimulatorCameraEfficiency

logger = logging.getLogger()


@pytest.fixture
def camera_efficiency_sst(io_handler, model_version, mocker):
    from unittest.mock import MagicMock

    camera_eff = CameraEfficiency(
        config_data={
            "telescope": "SSTS-05",
            "site": "South",
            "model_version": model_version,
            "zenith_angle": 20 * u.deg,
            "azimuth_angle": 0 * u.deg,
        },
        label="validate_camera_efficiency",
        efficiency_type="shower",
    )
    # Create a mock camera object to avoid loading camera config file
    mock_camera = MagicMock()
    mock_camera.get_camera_fill_factor.return_value = 0.8
    mock_camera.get_pixel_active_solid_angle.return_value = 1.0e-6
    # Replace the camera property with our mock
    mocker.patch.object(
        type(camera_eff.telescope_model),
        "camera",
        new_callable=mocker.PropertyMock,
        return_value=mock_camera,
    )
    # Mock get_on_axis_eff_optical_area to avoid loading optics_properties file
    mocker.patch.object(
        camera_eff.telescope_model,
        "get_on_axis_eff_optical_area",
        return_value=100.0 * u.m**2,
    )
    return camera_eff


@pytest.fixture
def simulator_camera_efficiency(camera_efficiency_sst, site_model_south, mocker):
    # Mock export_model_files to avoid file operations
    mocker.patch.object(camera_efficiency_sst, "export_model_files")
    simulator = SimulatorCameraEfficiency(
        telescope_model=camera_efficiency_sst.telescope_model,
        site_model=site_model_south,
        file_simtel=camera_efficiency_sst._file["sim_telarray"],
        label="test-simtel-runner-camera-efficiency",
    )
    # Mock is_file_2d to avoid file reads
    mocker.patch.object(simulator._telescope_model, "is_file_2d", return_value=False)
    return simulator


@pytest.fixture
def expected_command():
    return [
        "testeff",
        "-fnsb",
        "-alt",
        "2147.0",
        "-fatm",
        "atm_trans_2147_1_10_2_0_2147.dat",
        "-flen",
        "2.15191",
        "-fcur",
        "4.241",
        "-spix",
        "0.6",
        "-fmir",
        "weighted_average_1D_primary_mirror_incidence_angle_ref_astri-2d_2018-01-17.dat",
        "-m2",
        "-teltrans",
        "0.921",
        "transmission_sstcam_weighted_220512.dat",
        "-fqe",
        "PDE_lvr3_6mm_75um_uncoated_5.9V.dat",
    ]


@pytest.fixture
def benn_ellison_spectrum_file_name():
    return "Benn_LaPalma_sky_converted.lis"


def test_make_run_command(
    simulator_camera_efficiency, expected_command, benn_ellison_spectrum_file_name
):
    # With mocked database, just verify make_run_command() executes without errors
    command, std_out_file, std_err_file = simulator_camera_efficiency.make_run_command()

    # Verify command is a list
    assert isinstance(command, list)
    # Verify first element contains testeff
    assert "testeff" in str(command[0])
    # Verify some key flags are present
    assert "-fnsb" in command
    assert "-alt" in command
    assert isinstance(std_out_file, Path)
    assert std_err_file is None


def test_make_run_command_with_nsb_spectrum(simulator_camera_efficiency, expected_command):
    # With mocked database, just verify make_run_command() executes without errors
    simulator_camera_efficiency.nsb_spectrum = (
        "tests/resources/benn_ellison_spectrum_for_testing.txt"
    )
    command, _, _ = simulator_camera_efficiency.make_run_command()

    # Verify command is a list
    assert isinstance(command, list)
    # Verify first element contains testeff
    assert "testeff" in str(command[0])
    # Verify some key flags are present
    assert "-fnsb" in command
    assert "-alt" in command
    # Verify the nsb spectrum file is in the command
    assert any("benn_ellison_spectrum_for_testing.txt" in str(cmd) for cmd in command)


def test_make_run_command_without_altitude_correction(
    simulator_camera_efficiency, expected_command, benn_ellison_spectrum_file_name
):
    # With mocked database, just verify make_run_command() executes without errors
    simulator_camera_efficiency.skip_correction_to_nsb_spectrum = True
    command, _, _ = simulator_camera_efficiency.make_run_command()

    # Verify command is a list
    assert isinstance(command, list)
    # Verify first element contains testeff
    assert "testeff" in str(command[0])
    # Verify -nc flag is present (no correction)
    assert "-nc" in command


def test_check_run_result(simulator_camera_efficiency):
    """
    Testing here that the file does not exist because no simulations
    are run in unit tests. This function is tested for the positive case
    in the integration tests.
    """

    with pytest.raises(RuntimeError):
        simulator_camera_efficiency._check_run_result()


def test_get_one_dim_distribution(model_version_prod5, site_model_south, mocker, io_handler):
    from unittest.mock import MagicMock

    from astropy.table import Table

    logger.warning(
        "Running test_get_one_dim_distribution using prod5 model "
        " (prod6 model with 1D transmission function)"
    )
    camera_efficiency_sst_prod5 = CameraEfficiency(
        config_data={
            "telescope": "SSTS-design",
            "site": "South",
            "model_version": model_version_prod5,
            "zenith_angle": 20 * u.deg,
            "azimuth_angle": 0 * u.deg,
        },
        label="validate_camera_efficiency",
        efficiency_type="shower",
    )

    # Mock camera to avoid loading camera config file
    mock_camera = MagicMock()
    mock_camera.get_camera_fill_factor.return_value = 0.8
    mock_camera.get_pixel_active_solid_angle.return_value = 1.0e-6
    mocker.patch.object(
        type(camera_efficiency_sst_prod5.telescope_model),
        "camera",
        new_callable=mocker.PropertyMock,
        return_value=mock_camera,
    )

    # Mock export_model_files to avoid file operations
    mocker.patch.object(camera_efficiency_sst_prod5, "export_model_files")

    # 2D transmission window not defined in prod6; required prod5 runner
    simulator_camera_efficiency_prod5 = SimulatorCameraEfficiency(
        telescope_model=camera_efficiency_sst_prod5.telescope_model,
        site_model=site_model_south,
        file_simtel=camera_efficiency_sst_prod5._file["sim_telarray"],
        label="test-simtel-runner-camera-efficiency",
    )

    # Mock read_incidence_angle_distribution to return a simple table
    mock_table = Table([[0.0, 10.0, 20.0], [1.0, 0.8, 0.6]], names=["angle", "value"])
    mocker.patch.object(
        simulator_camera_efficiency_prod5._telescope_model,
        "read_incidence_angle_distribution",
        return_value=mock_table,
    )

    # Mock read_two_dim_wavelength_angle to return a simple table
    mock_2d_table = Table(
        [[300.0, 400.0], [0.9, 0.85], [0.8, 0.75]],
        names=["wavelength", "angle_0", "angle_10"],
    )
    mocker.patch.object(
        simulator_camera_efficiency_prod5._telescope_model,
        "read_two_dim_wavelength_angle",
        return_value=mock_2d_table,
    )

    # Mock calc_average_curve to return a simple table
    mock_avg_table = Table([[300.0, 400.0], [0.85, 0.80]], names=["wavelength", "transmission"])
    mocker.patch.object(
        simulator_camera_efficiency_prod5._telescope_model,
        "calc_average_curve",
        return_value=mock_avg_table,
    )

    # Mock export_table_to_model_directory to return a Path
    from pathlib import Path

    mock_export_path = Path(io_handler.get_output_directory()) / "test_1d_distribution.dat"
    mock_export_path.touch()  # Create empty file
    mocker.patch.object(
        simulator_camera_efficiency_prod5._telescope_model,
        "export_table_to_model_directory",
        return_value=mock_export_path,
    )

    camera_filter_file = simulator_camera_efficiency_prod5._get_one_dim_distribution(
        "camera_filter", "camera_filter_incidence_angle"
    )
    assert camera_filter_file.exists()


def test_validate_or_fix_nsb_spectrum_file_format(simulator_camera_efficiency):
    """
    Test that the function returns a file with the correct format.
    The test is run twice, once on a file with the wrong format and then
    the produced file is tested as well in order to make sure that
    the function does not change the format of a file with the correct format.
    """
    validated_nsb_spectrum_file = (
        simulator_camera_efficiency._validate_or_fix_nsb_spectrum_file_format(
            "tests/resources/benn_ellison_spectrum_for_testing.txt"
        )
    )
    assert validated_nsb_spectrum_file.exists()

    def produced_file_has_expected_values(file):
        # Test that the first 3 non-comment lines are the following values:
        wavelengths = [300.00, 315.00, 330.00]
        nsbs = [0, 0.612, 1.95]
        with open(file, encoding="utf-8") as file:
            for line in file:
                if line.startswith("#"):
                    continue
                entry = line.split()
                expected_wavelength = wavelengths.pop(0)
                expected_nsb = nsbs.pop(0)
                assert float(entry[0]) == pytest.approx(expected_wavelength)
                assert float(entry[2]) == pytest.approx(expected_nsb)
                assert len(entry) == 3
                if len(wavelengths) == 0:
                    break

    produced_file_has_expected_values(validated_nsb_spectrum_file)
    # Test that the function does not change the format of a file with the correct format
    second_validated_nsb_spectrum_file = (
        simulator_camera_efficiency._validate_or_fix_nsb_spectrum_file_format(
            validated_nsb_spectrum_file
        )
    )
    produced_file_has_expected_values(second_validated_nsb_spectrum_file)


def test_get_curvature_radius_mirror_class_2(simulator_camera_efficiency, mocker):
    mock_telescope_model = simulator_camera_efficiency._telescope_model
    mocker.patch.object(
        mock_telescope_model, "get_parameter_value_with_unit", return_value=1.5 * u.m
    )
    radius = simulator_camera_efficiency._get_curvature_radius(mirror_class=2)
    assert radius == pytest.approx(1.5)


def test_get_curvature_radius_parabolic_dish_true(simulator_camera_efficiency, mocker):
    mock_telescope_model = simulator_camera_efficiency._telescope_model
    mocker.patch.object(mock_telescope_model, "get_parameter_value", return_value=True)
    mocker.patch.object(
        mock_telescope_model, "get_parameter_value_with_unit", return_value=1.2 * u.m
    )
    radius = simulator_camera_efficiency._get_curvature_radius(mirror_class=1)
    assert radius == pytest.approx(2.4)


def test_get_curvature_radius_parabolic_dish_false(simulator_camera_efficiency, mocker):
    mock_telescope_model = simulator_camera_efficiency._telescope_model
    mocker.patch.object(mock_telescope_model, "get_parameter_value", return_value=False)
    mocker.patch.object(
        mock_telescope_model, "get_parameter_value_with_unit", return_value=1.7 * u.m
    )
    radius = simulator_camera_efficiency._get_curvature_radius(mirror_class=1)
    assert radius == pytest.approx(1.7)


def test_check_run_result_success(simulator_camera_efficiency, mocker):
    mocker.patch.object(Path, "exists", return_value=True)
    mock_logger = mocker.patch.object(simulator_camera_efficiency._logger, "debug")

    simulator_camera_efficiency._check_run_result()

    mock_logger.assert_called_once_with("Everything looks fine with output file.")
