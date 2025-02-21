#!/usr/bin/python3

import logging

import astropy.units as u
import pytest

from simtools.ray_tracing.ray_tracing import RayTracing
from simtools.simtel.simulator_ray_tracing import SimulatorRayTracing

logger = logging.getLogger()


@pytest.fixture
def ray_tracing_sst(telescope_model_sst, simtel_path):
    return RayTracing(
        telescope_model=telescope_model_sst,
        simtel_path=simtel_path,
        label="test-simtel-runner-ray-tracing",
        source_distance=10 * u.km,
        zenith_angle=20 * u.deg,
        off_axis_angle=[0, 2] * u.deg,
        single_mirror_mode=False,
    )


@pytest.fixture
def ray_tracing_mst(telescope_model_mst, simtel_path):
    return RayTracing(
        telescope_model=telescope_model_mst,
        simtel_path=simtel_path,
        label="test-simtel-runner-ray-tracing",
        source_distance=10 * u.km,
        zenith_angle=20 * u.deg,
        off_axis_angle=[0, 2] * u.deg,
        single_mirror_mode=False,
    )


@pytest.fixture
def simulator_ray_tracing_sst(ray_tracing_sst, telescope_model_sst, simtel_path):
    return SimulatorRayTracing(
        simtel_path=simtel_path,
        telescope_model=telescope_model_sst,
        config_data={
            "zenith_angle": ray_tracing_sst.zenith_angle,
            "off_axis_angle": 0.0,
            "source_distance": 10,
            "single_mirror_mode": ray_tracing_sst.single_mirror_mode,
            "use_random_focal_length": ray_tracing_sst.use_random_focal_length,
            "mirror_numbers": 0,
        },
        label="test-simtel-runner-ray-tracing",
    )


@pytest.fixture
def simulator_ray_tracing_single_mirror(ray_tracing_mst, telescope_model_mst, simtel_path):
    return SimulatorRayTracing(
        simtel_path=simtel_path,
        telescope_model=telescope_model_mst,
        config_data={
            "zenith_angle": ray_tracing_mst.zenith_angle,
            "off_axis_angle": 0.0,
            "source_distance": 10.0 * u.km,
            "single_mirror_mode": True,
            "use_random_focal_length": ray_tracing_mst.use_random_focal_length,
            "mirror_numbers": 0,
        },
        label="test-simtel-runner-ray-tracing",
    )


@pytest.fixture
def single_pixel_camera_file_content():
    return [
        "# Single pixel camera",
        'PixType 1   0  0 300   1 300 0.00   "funnel_perfect.dat"',
        "Pixel 0 1 0. 0.  0  0  0 0x00 1",
        "Trigger 1 of 0",
    ]


@pytest.fixture
def funnel_perfect_file_content():
    return [
        "# Perfect light collection where the angular efficiency of funnels is needed",
        "0    1.0",
        "30   1.0",
        "60   1.0",
        "90   1.0",
    ]


def test_load_required_files(simulator_ray_tracing_sst):
    simulator_ray_tracing_sst._load_required_files(force_simulate=False)

    # This file is not actually needed and does not exist in simtools.
    # However, its name is needed too provide the name of a CORSIKA input file to sim_telarray
    # so here we check the it does not actually exist.
    assert not simulator_ray_tracing_sst._corsika_file.exists()
    assert simulator_ray_tracing_sst._photons_file.exists()
    assert simulator_ray_tracing_sst._stars_file.exists()
    assert not simulator_ray_tracing_sst.telescope_model.config_file_directory.joinpath(
        "single_pixel_camera.dat"
    ).exists()


def assert_single_pixel_and_perfect_funnel_files(
    simulator_ray_tracing_to_test, single_pixel_camera_file_content, funnel_perfect_file_content
):
    single_pixel_camera_file = (
        simulator_ray_tracing_to_test.telescope_model.config_file_directory.joinpath(
            "single_pixel_camera.dat"
        )
    )
    funnel_perfect_file = (
        simulator_ray_tracing_to_test.telescope_model.config_file_directory.joinpath(
            "funnel_perfect.dat"
        )
    )

    for created_file, content in zip(
        [single_pixel_camera_file, funnel_perfect_file],
        [single_pixel_camera_file_content, funnel_perfect_file_content],
    ):
        assert created_file.exists()
        with created_file.open("r") as file:
            lines = file.readlines()
            for i_line, line in enumerate(lines):
                assert line.strip() == content[i_line]


def test_load_required_files_single_mirror(
    simulator_ray_tracing_single_mirror,
    single_pixel_camera_file_content,
    funnel_perfect_file_content,
):
    simulator_ray_tracing_single_mirror._load_required_files(force_simulate=False)

    # This file is not actually needed and does not exist in simtools.
    # However, its name is needed too provide the name of a CORSIKA input file to sim_telarray
    # so here we check the it does not actually exist.
    assert not simulator_ray_tracing_single_mirror._corsika_file.exists()
    assert simulator_ray_tracing_single_mirror._photons_file.exists()
    assert simulator_ray_tracing_single_mirror._stars_file.exists()

    assert_single_pixel_and_perfect_funnel_files(
        simulator_ray_tracing_single_mirror,
        single_pixel_camera_file_content,
        funnel_perfect_file_content,
    )


def test_make_run_command(simulator_ray_tracing_sst, model_version):
    command, stdout_file, stderr_file = simulator_ray_tracing_sst._make_run_command()

    assert "bin/sim_telarray" in command
    assert (
        "model/CTA-South-SSTS-design-"
        + model_version.replace("_", "-")
        + "_test-telescope-model-sst.cfg"
        in command
    )
    assert "altitude=2147.0 -C telescope_theta=20.0 -C star_photons=100000" in command
    log_file = (
        "log-South-SSTS-design-d10.0km-za20.0deg-off0.000deg_test-simtel-runner-ray-tracing.log"
    )
    assert stdout_file.name == log_file
    assert stderr_file.name == log_file


def test_make_run_command_single_mirror(simulator_ray_tracing_single_mirror):
    command, _, _ = simulator_ray_tracing_single_mirror._make_run_command()

    assert "bin/sim_telarray" in command
    assert "focus_offset" in command


def test_check_run_result(simulator_ray_tracing_sst):
    with pytest.raises(RuntimeError):
        simulator_ray_tracing_sst._check_run_result()

    # Add manually entries to the photons file to test the function works as expected
    simulator_ray_tracing_sst._load_required_files(force_simulate=False)
    with simulator_ray_tracing_sst._photons_file.open("a") as file:
        file.writelines(150 * [f"{1}\n"])
    assert simulator_ray_tracing_sst._check_run_result()


def test_write_out_single_pixel_camera_file(
    simulator_ray_tracing_sst, single_pixel_camera_file_content, funnel_perfect_file_content
):
    simulator_ray_tracing_sst._write_out_single_pixel_camera_file()

    assert_single_pixel_and_perfect_funnel_files(
        simulator_ray_tracing_sst, single_pixel_camera_file_content, funnel_perfect_file_content
    )
