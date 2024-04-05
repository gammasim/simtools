#!/usr/bin/python3

import logging

import pytest

from simtools.camera_efficiency import CameraEfficiency
from simtools.simtel.simtel_runner_camera_efficiency import SimtelRunnerCameraEfficiency

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def camera_efficiency_sst(telescope_model_sst, simtel_path, site_model_south):

    telescope_model_sst.export_model_files()
    camera_efficiency_sst = CameraEfficiency(
        telescope_model=telescope_model_sst,
        site_model=site_model_south,
        simtel_source_path=simtel_path,
        test=True,
    )
    return camera_efficiency_sst


@pytest.fixture
def simtel_runner_camera_efficiency(camera_efficiency_sst, telescope_model_sst, simtel_path):
    simtel_runner_camera_efficiency = SimtelRunnerCameraEfficiency(
        simtel_source_path=simtel_path,
        telescope_model=telescope_model_sst,
        file_simtel=camera_efficiency_sst._file["simtel"],
        label="test-simtel-runner-camera-efficiency",
    )
    return simtel_runner_camera_efficiency


def test_shall_run(simtel_runner_camera_efficiency):
    """
    Testing here that the file does not exist because no simulations
    are run in unit tests. This function is tested for the positive case
    in the integration tests.
    """

    assert simtel_runner_camera_efficiency._shall_run()


def test_make_run_command(simtel_runner_camera_efficiency):
    command = simtel_runner_camera_efficiency._make_run_command()

    assert "testeff" in command
    assert "-fnsb" not in command
    assert "alt 2147.0 -fatm atm_trans_2147_1_10_2_0_2147.dat" in command
    assert "-flen 2.15191 -spix 0.6" in command
    assert "weighted_average_1D_ref_astri-2d_2018-01-17.dat -m2" in command
    assert "-teltrans 0.921" in command
    assert "transmission_sstcam_weighted_220512.dat" in command
    assert "-fqe PDE_lvr3_6mm_75um_uncoated_5.9V.dat" in command


def test_make_run_command_with_nsb_spectrum(simtel_runner_camera_efficiency):
    simtel_runner_camera_efficiency.nsb_spectrum = (
        "tests/resources/benn_ellison_spectrum_for_testing.txt"
    )
    command = simtel_runner_camera_efficiency._make_run_command()

    assert "testeff" in command
    assert "-fnsb" in command
    assert "benn_ellison_spectrum_for_testing.txt" in command
    assert "alt 2147.0 -fatm atm_trans_2147_1_10_2_0_2147.dat" in command
    assert "-flen 2.15191 -spix 0.6" in command
    assert "weighted_average_1D_ref_astri-2d_2018-01-17.dat -m2" in command
    assert "-teltrans 0.921" in command
    assert "transmission_sstcam_weighted_220512.dat" in command
    assert "-fqe PDE_lvr3_6mm_75um_uncoated_5.9V.dat" in command


def test_check_run_result(simtel_runner_camera_efficiency):
    """
    Testing here that the file does not exist because no simulations
    are run in unit tests. This function is tested for the positive case
    in the integration tests.
    """

    with pytest.raises(RuntimeError):
        simtel_runner_camera_efficiency._check_run_result()


@pytest.mark.xfail(reason="Missing camera_filter_incidence_angle for prod6 in Derived-DB")
def test_get_one_dim_distribution(simtel_runner_camera_efficiency):
    camera_filter_file = simtel_runner_camera_efficiency._get_one_dim_distribution(
        "camera_filter", "camera_filter_incidence_angle"
    )
    assert camera_filter_file.exists()


def test_validate_or_fix_nsb_spectrum_file_format(simtel_runner_camera_efficiency):
    """
    Test that the function returns a file with the correct format.
    The test is run twice, once on a file with the wrong format and then
    the produced file is tested as well in order to make sure that
    the function does not change the format of a file with the correct format.
    """
    validated_nsb_spectrum_file = (
        simtel_runner_camera_efficiency._validate_or_fix_nsb_spectrum_file_format(
            "tests/resources/benn_ellison_spectrum_for_testing.txt"
        )
    )
    assert validated_nsb_spectrum_file.exists()

    def produced_file_has_expected_values(file):
        # Test that the first 3 non-comment lines are the following values:
        wavelengths = [300.00, 315.00, 330.00]
        NSBs = [0, 0.612, 1.95]
        with open(file, "r", encoding="utf-8") as file:
            for line in file:
                if line.startswith("#"):
                    continue
                entry = line.split()
                assert float(entry[0]) == pytest.approx(wavelengths.pop(0))
                assert float(entry[2]) == pytest.approx(NSBs.pop(0))
                assert len(entry) == 3
                if len(wavelengths) == 0:
                    break

    produced_file_has_expected_values(validated_nsb_spectrum_file)
    # Test that the function does not change the format of a file with the correct format
    second_validated_nsb_spectrum_file = (
        simtel_runner_camera_efficiency._validate_or_fix_nsb_spectrum_file_format(
            validated_nsb_spectrum_file
        )
    )
    produced_file_has_expected_values(second_validated_nsb_spectrum_file)
