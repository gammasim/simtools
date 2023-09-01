#!/usr/bin/python3

import logging

import astropy.units as u
import pytest

from simtools.ray_tracing import RayTracing


def test_ray_tracing_from_dict(simtel_path, io_handler, telescope_model_mst, caplog):
    config_data = {
        "source_distance": 10 * u.km,
        "zenith_angle": 30 * u.deg,
        "off_axis_angle": [0, 2] * u.deg,
    }

    with caplog.at_level(logging.DEBUG):
        ray = RayTracing(
            telescope_model=telescope_model_mst,
            simtel_source_path=simtel_path,
            config_data=config_data,
        )

    assert ray.config.zenith_angle == 30
    assert len(ray.config.off_axis_angle) == 2
    assert "Initializing RayTracing class" in caplog.text
    assert "RayTracing contains a valid TelescopeModel" in caplog.text
    assert ray._simtel_source_path == simtel_path
    assert repr(ray) == f"RayTracing(label={telescope_model_mst.label})\n"


def test_ray_tracing_from_kwargs(io_handler, simtel_path, telescope_model_mst):
    source_distance = 10 * u.km
    zenith_angle = 30 * u.deg
    off_axis_angle = [0, 2] * u.deg

    ray = RayTracing.from_kwargs(
        telescope_model=telescope_model_mst,
        simtel_source_path=simtel_path,
        source_distance=source_distance,
        zenith_angle=zenith_angle,
        off_axis_angle=off_axis_angle,
    )

    assert ray.config.zenith_angle == 30
    assert len(ray.config.off_axis_angle) == 2


def test_ray_tracing_single_mirror_mode(simtel_path, io_handler, telescope_model_mst, caplog):
    telescope_model_mst.export_config_file()
    config_data = {
        "source_distance": 10 * u.km,
        "zenith_angle": 30 * u.deg,
        "off_axis_angle": [0, 2] * u.deg,
        "single_mirror_mode": True,
    }

    with caplog.at_level(logging.DEBUG):
        ray = RayTracing(
            telescope_model=telescope_model_mst,
            simtel_source_path=simtel_path,
            config_data=config_data,
        )

    assert ray.config.zenith_angle == 30
    assert len(ray.config.off_axis_angle) == 2
    assert ray.config.single_mirror_mode == True
    assert "Single mirror mode is activated" in caplog.text


def test_ray_tracing_single_mirror_mode_mirror_numbers(
    simtel_path, io_handler, telescope_model_mst
):
    telescope_model_mst.export_config_file()
    config_data = {
        "source_distance": 10 * u.km,
        "zenith_angle": 30 * u.deg,
        "off_axis_angle": [0, 2] * u.deg,
        "single_mirror_mode": True,
        "mirror_numbers": [1, 2, 3],
    }

    ray = RayTracing(
        telescope_model=telescope_model_mst,
        simtel_source_path=simtel_path,
        config_data=config_data,
    )

    assert ray._mirror_numbers == [1, 2, 3]


def test_ray_tracing_invalid_telescope_model(simtel_path, io_handler, caplog):
    config_data = {
        "source_distance": 10 * u.km,
        "zenith_angle": 30 * u.deg,
        "off_axis_angle": [0, 2] * u.deg,
    }

    with pytest.raises(ValueError):
        RayTracing(
            telescope_model=None,
            simtel_source_path=simtel_path,
            config_data=config_data,
        )
        assert "Invalid TelescopeModel" in caplog.text
