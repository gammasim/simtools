#!/usr/bin/python3

import logging
import shutil

import astropy.units as u
import pytest

from simtools.ray_tracing import RayTracing
from simtools.utils import names


@pytest.fixture
def ray_tracing_lst(telescope_model_lst, simtel_path, io_handler):
    """A RayTracing instance with results read in that were simulated before"""
    config_data = {
        "source_distance": 10 * u.km,
        "zenith_angle": 20 * u.deg,
        "off_axis_angle": [0, 0] * u.deg,
    }

    ray_tracing_lst = RayTracing(
        telescope_model=telescope_model_lst,
        simtel_source_path=simtel_path,
        config_data=config_data,
        label="validate_optics",
    )

    output_directory = ray_tracing_lst._output_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    shutil.copy(
        "tests/resources/ray-tracing-North-LST-1-d10.0-za20.0_validate_optics.ecsv",
        output_directory.joinpath("results"),
    )
    shutil.copy(
        "tests/resources/photons-North-LST-1-d10.0-za20.0-off0.000_validate_optics.lis.gz",
        output_directory,
    )
    return ray_tracing_lst


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
    assert ray.config.single_mirror_mode
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


def test_ray_tracing_read_results(ray_tracing_lst):
    ray_tracing_lst.analyze(force=False)

    assert ray_tracing_lst._has_results is True
    assert len(ray_tracing_lst._results) > 0
    assert ray_tracing_lst.get_mean("d80_cm").value == pytest.approx(4.256768651160611, abs=1e-5)


def test_export_results(simtel_path, telescope_model_lst, caplog):
    """
    Test the export_results method of the RayTracing class without results
    """

    config_data = {
        "source_distance": 10 * u.km,
        "zenith_angle": 20 * u.deg,
        "off_axis_angle": [0, 0] * u.deg,
    }

    ray = RayTracing(
        telescope_model=telescope_model_lst,
        simtel_source_path=simtel_path,
        config_data=config_data,
        label="export_results",
    )
    ray.export_results()
    assert "Cannot export results because it does not exist" in caplog.text


def test_ray_tracing_plot(ray_tracing_lst, caplog):
    """
    Test the plot method of the RayTracing class with an invalid key and a valid key
    """

    ray_tracing_lst.analyze(force=False)
    # First test a wrong key
    with pytest.raises(KeyError):
        ray_tracing_lst.plot(key="invalid_key")
        assert "Invalid key" in caplog.text

    # Now test a valid key
    ray_tracing_lst.plot(key="d80_cm", save=True)
    assert "Saving fig in" in caplog.text
    plot_file_name = names.ray_tracing_plot_file_name(
        "d80_cm",
        ray_tracing_lst._telescope_model.site,
        ray_tracing_lst._telescope_model.name,
        ray_tracing_lst._source_distance,
        ray_tracing_lst.config.zenith_angle,
        ray_tracing_lst.label,
    )
    plot_file = ray_tracing_lst._output_directory.joinpath("figures").joinpath(plot_file_name)
    assert plot_file.exists() is True


def test_ray_tracing_invalid_key(ray_tracing_lst, caplog):
    """
    Test the a few methods of the RayTracing class with an invalid key
    """

    with pytest.raises(KeyError):
        ray_tracing_lst.plot_histogram(key="invalid_key")
        assert "Invalid key" in caplog.text

    with pytest.raises(KeyError):
        ray_tracing_lst.get_mean(key="invalid_key")
        assert "Invalid key" in caplog.text

    with pytest.raises(KeyError):
        ray_tracing_lst.get_std_dev(key="invalid_key")
        assert "Invalid key" in caplog.text


def test_ray_tracing_get_std_dev(ray_tracing_lst):
    """Test the get_std_dev method of the RayTracing class"""

    ray_tracing_lst.analyze(force=False)
    assert ray_tracing_lst.get_std_dev(key="d80_cm").value == pytest.approx(
        0.8418404935128992, abs=1e-5
    )


def test_ray_tracing_no_images(ray_tracing_lst, caplog):
    """Test the images method of the RayTracing class with no images"""

    assert ray_tracing_lst.images() is None
    assert "No image found" in caplog.text
