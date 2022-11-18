#!/usr/bin/python3

import logging

import pytest

from simtools.camera_efficiency import CameraEfficiency
from simtools.simtel.simtel_runner_camera_efficiency import SimtelRunnerCameraEfficiency

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def camera_efficiency_sst(telescope_model_sst, simtel_path):
    camera_efficiency_sst = CameraEfficiency(
        telescope_model=telescope_model_sst, simtel_source_path=simtel_path, test=True
    )
    return camera_efficiency_sst


@pytest.fixture
def simtel_runner_camera_efficiency(camera_efficiency_sst, telescope_model_sst, simtel_path):
    simtel_runner_camera_efficiency = SimtelRunnerCameraEfficiency(
        simtel_source_path=simtel_path,
        telescope_model=telescope_model_sst,
        file_simtel=camera_efficiency_sst._file_simtel,
        label="test-simtel-runner-camera-efficiency",
    )
    return simtel_runner_camera_efficiency


def test_shall_run(simtel_runner_camera_efficiency):
    assert not simtel_runner_camera_efficiency._shall_run()


def test_make_run_command(simtel_runner_camera_efficiency):

    # command = simtel_runner_camera_efficiency._make_run_command()

    # FIXME - check that the command is as expected
    assert True


def test_get_one_dim_distribution(telescope_model_sst, simtel_runner_camera_efficiency):

    telescope_model_sst.export_model_files()
    camera_filter_file = simtel_runner_camera_efficiency._get_one_dim_distribution(
        "camera_filter", "camera_filter_incidence_angle"
    )
    assert camera_filter_file.exists()
