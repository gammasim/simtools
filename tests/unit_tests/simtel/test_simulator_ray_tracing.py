#!/usr/bin/python3

import logging

import astropy.units as u
import pytest

from simtools.ray_tracing import RayTracing
from simtools.simtel.simulator_ray_tracing import SimulatorRayTracing

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture()
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


@pytest.fixture()
def simulator_ray_tracing(ray_tracing_sst, telescope_model_sst, simtel_path):
    simulator_ray_tracing = SimulatorRayTracing(
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
    return simulator_ray_tracing


def test_load_required_files(simulator_ray_tracing):
    simulator_ray_tracing._load_required_files(force_simulate=False)

    # This file is not actually needed and does not exist in simtools.
    # However, its name is needed too provide the name of a CORSIKA input file to sim_telarray
    # so here we check the it does not actually exist.
    assert not simulator_ray_tracing._corsika_file.exists()
    assert simulator_ray_tracing._photons_file.exists()
    assert simulator_ray_tracing._stars_file.exists()


def test_shall_run(simulator_ray_tracing):
    """
    Testing here that the file does not exist because no simulations
    are run in unit tests. This function is tested for the positive case
    in the integration tests.
    """

    assert simulator_ray_tracing._shall_run()


def test_make_run_command(simulator_ray_tracing, model_version):
    command = simulator_ray_tracing._make_run_command()

    assert "bin/sim_telarray" in command
    assert (
        "model/CTA-South-SSTS-design-"
        + model_version.replace("_", "-")
        + "_test-telescope-model-sst.cfg"
        in command
    )
    assert "altitude=2147.0 -C telescope_theta=20.0 -C star_photons=100000" in command
    assert (
        "log-South-SSTS-design-d10.0km-za20.0deg-off0.000deg_test-simtel-runner-ray-tracing.log"
        in command
    )


def test_check_run_result(simulator_ray_tracing):
    """
    Testing here that the file does not exist because no simulations
    are run in unit tests. This function is tested for the positive case
    in the integration tests.
    """

    with pytest.raises(RuntimeError):
        simulator_ray_tracing._check_run_result()


def test_is_photon_list_file_ok(simulator_ray_tracing):
    """
    Testing here that the file does not exist because no simulations
    are run in unit tests. This function is tested for the positive case
    in the integration tests.
    """
    assert not simulator_ray_tracing._is_photon_list_file_ok()

    # Now add manually entries to the photons file to test the function works as expected
    simulator_ray_tracing._load_required_files(force_simulate=False)
    with simulator_ray_tracing._photons_file.open("a") as file:
        file.writelines(150 * [f"{1}\n"])

    assert simulator_ray_tracing._is_photon_list_file_ok()
