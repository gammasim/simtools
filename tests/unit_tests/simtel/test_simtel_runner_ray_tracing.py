#!/usr/bin/python3

import logging

import astropy.units as u
import pytest

from simtools.ray_tracing import RayTracing
from simtools.simtel.simtel_runner_ray_tracing import SimtelRunnerRayTracing

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def ray_tracing_sst(telescope_model_sst, simtel_path):
    # telescope_model_sst.export_model_files()

    config_data = {
        "source_distance": 10 * u.km,
        "zenith_angle": 20 * u.deg,
        "off_axis_angle": [0, 2] * u.deg,
        "single_mirror_mode": False,
    }

    ray_tracing_sst = RayTracing(
        telescope_model=telescope_model_sst,
        simtel_source_path=simtel_path,
        config_data=config_data,
        label="test-simtel-runner-ray-tracing",
    )

    return ray_tracing_sst


@pytest.fixture
def simtel_runner_ray_tracing(ray_tracing_sst, telescope_model_sst, simtel_path):
    simtel_runner_ray_tracing = SimtelRunnerRayTracing(
        simtel_source_path=simtel_path,
        telescope_model=telescope_model_sst,
        config_data={
            "zenith_angle": ray_tracing_sst.config.zenith_angle * u.deg,
            "source_distance": ray_tracing_sst._source_distance * u.km,
            "off_axis_angle": 0 * u.deg,
            "mirror_numbers": 0,
            "use_random_focal_length": ray_tracing_sst.config.use_random_focal_length,
            "single_mirror_mode": ray_tracing_sst.config.single_mirror_mode,
        },
        label="test-simtel-runner-ray-tracing",
    )
    return simtel_runner_ray_tracing


def test_load_required_files(simtel_runner_ray_tracing):
    simtel_runner_ray_tracing._load_required_files(force_simulate=False)

    # This file is not actually needed and does not exist in simtools.
    # However, its name is needed too provide the name of a CORSIKA input file to sim_telarray
    # so here we check the it does not actually exist.
    assert not simtel_runner_ray_tracing._corsika_file.exists()
    assert simtel_runner_ray_tracing._photons_file.exists()
    assert simtel_runner_ray_tracing._stars_file.exists()


def test_shall_run(simtel_runner_ray_tracing):
    """
    Testing here that the file does not exist because no simulations
    are run in unit tests. This function is tested for the positive case
    in the integration tests.
    """

    assert simtel_runner_ray_tracing._shall_run()


def test_make_run_command(simtel_runner_ray_tracing):
    command = simtel_runner_ray_tracing._make_run_command()

    assert "bin/sim_telarray" in command
    assert "model/CTA-South-SST-D-2020-06-28_test-telescope-model-sst.cfg" in command
    assert "altitude=2147.0 -C telescope_theta=20.0 -C star_photons=100000" in command
    assert "log-South-SST-D-d10.0-za20.0-off0.000_test-simtel-runner-ray-tracing.log" in command


def test_check_run_result(simtel_runner_ray_tracing):
    """
    Testing here that the file does not exist because no simulations
    are run in unit tests. This function is tested for the positive case
    in the integration tests.
    """

    with pytest.raises(RuntimeError):
        simtel_runner_ray_tracing._check_run_result()


def test_is_photon_list_file_ok(simtel_runner_ray_tracing):
    """
    Testing here that the file does not exist because no simulations
    are run in unit tests. This function is tested for the positive case
    in the integration tests.
    """
    assert not simtel_runner_ray_tracing._is_photon_list_file_ok()

    # Now add manually entries to the photons file to test the function works as expected
    simtel_runner_ray_tracing._load_required_files(force_simulate=False)
    with simtel_runner_ray_tracing._photons_file.open("a") as file:
        file.writelines(150 * [f"{1}\n"])

    assert simtel_runner_ray_tracing._is_photon_list_file_ok()
