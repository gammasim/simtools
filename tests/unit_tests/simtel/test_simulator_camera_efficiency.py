#!/usr/bin/python3

import logging

import astropy.units as u
import pytest

from simtools.camera_efficiency import CameraEfficiency
from simtools.simtel.simulator_camera_efficiency import SimulatorCameraEfficiency

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture()
def simulator_camera_efficiency(camera_efficiency_sst, simtel_path):
    camera_efficiency_sst.export_model_files()
    return SimulatorCameraEfficiency(
        simtel_path=simtel_path,
        telescope_model=camera_efficiency_sst.telescope_model,
        file_simtel=camera_efficiency_sst._file["simtel"],
        label="test-simtel-runner-camera-efficiency",
    )


def test_make_run_command(simulator_camera_efficiency):
    command = simulator_camera_efficiency._make_run_command()

    assert "testeff" in command
    assert "-fnsb" not in command
    assert "alt 2147.0 -fatm atm_trans_2147_1_10_2_0_2147.dat" in command
    assert "-flen 2.15191 -spix 0.6" in command
    assert "weighted_average_1D_ref_astri-2d_2018-01-17.dat -m2" in command
    assert "-teltrans 0.921" in command
    assert "transmission_sstcam_weighted_220512.dat" in command
    assert "-fqe PDE_lvr3_6mm_75um_uncoated_5.9V.dat" in command


def test_make_run_command_with_nsb_spectrum(simulator_camera_efficiency):
    simulator_camera_efficiency.nsb_spectrum = (
        "tests/resources/benn_ellison_spectrum_for_testing.txt"
    )
    command = simulator_camera_efficiency._make_run_command()

    assert "testeff" in command
    assert "-fnsb" in command
    assert "benn_ellison_spectrum_for_testing.txt" in command
    assert "alt 2147.0 -fatm atm_trans_2147_1_10_2_0_2147.dat" in command
    assert "-flen 2.15191 -spix 0.6" in command
    assert "weighted_average_1D_ref_astri-2d_2018-01-17.dat -m2" in command
    assert "-teltrans 0.921" in command
    assert "transmission_sstcam_weighted_220512.dat" in command
    assert "-fqe PDE_lvr3_6mm_75um_uncoated_5.9V.dat" in command


def test_check_run_result(simulator_camera_efficiency):
    """
    Testing here that the file does not exist because no simulations
    are run in unit tests. This function is tested for the positive case
    in the integration tests.
    """

    with pytest.raises(RuntimeError):
        simulator_camera_efficiency._check_run_result()


@pytest.mark.xfail(reason="Test requires Derived-Values Database")
def test_get_one_dim_distribution(io_handler, db_config, simtel_path):

    logger.warning(
        "Running test_get_one_dim_distribution using prod5 model "
        " (prod6 model with 1D transmission function)"
    )
    camera_efficiency_sst_prod5 = CameraEfficiency(
        config_data={
            "telescope": "SSTS-design",
            "site": "South",
            "model_version": "prod5",
            "zenith_angle": 20 * u.deg,
            "azimuth_angle": 0 * u.deg,
        },
        db_config=db_config,
        simtel_path=simtel_path,
        label="validate_camera_efficiency",
        test=True,
    )

    # 2D transmission window not defined in prod6; required prod5 runner
    camera_efficiency_sst_prod5.export_model_files()
    simulator_camera_efficiency_prod5 = SimulatorCameraEfficiency(
        simtel_path=simtel_path,
        telescope_model=camera_efficiency_sst_prod5.telescope_model,
        file_simtel=camera_efficiency_sst_prod5._file["simtel"],
        label="test-simtel-runner-camera-efficiency",
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
                assert float(entry[0]) == pytest.approx(wavelengths.pop(0))
                assert float(entry[2]) == pytest.approx(nsbs.pop(0))
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
