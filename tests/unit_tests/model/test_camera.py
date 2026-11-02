#!/usr/bin/python3

import logging

import numpy as np
import pytest

from simtools.model.camera import Camera

logger = logging.getLogger()


@pytest.fixture
def camera_config_file():
    return "dummy.dat"


@pytest.fixture
def simple_camera_dict():
    """Simple camera config dictionary for testing."""
    return {
        "pixel_size": np.array([0.5, 0.5, 0.5, 0.5]),
        "common_pixel_shape": 1,
        "pixel_x": np.array([0, 1, 0, 1]),
        "pixel_y": np.array([0, 0, 1, 1]),
    }


@pytest.fixture
def simple_pixels_dict():
    """Simple pixels dictionary for testing."""
    return {
        "pixel_diameter": 0.5,
        "pixel_shape": 1,
        "pixel_spacing": 0.6,
        "lightguide_efficiency_angle_file": "test_angle.dat",
        "lightguide_efficiency_wavelength_file": "test_wavelength.dat",
        "rotate_angle": 0,
        "x": [0.0, 1.0, 0.0, 1.0],
        "y": [0.0, 0.0, 1.0, 1.0],
        "pix_id": [0, 1, 2, 3],
        "pix_on": [1, 1, 1, 1],
    }


def test_focal_length():
    with pytest.raises(ValueError, match="The focal length must be larger than zero"):
        Camera(telescope_name="test_camera", camera_config_file="test_config", focal_length=-1)


def test_from_configuration_reads_resolved_components():
    camera = Camera.from_configuration(
        "test_camera",
        {
            "rotate": 0.0,
            "pixel_types": [{"funnel_shape": 2, "funnel_diameter_cm": 0.5}],
            "pixels": [
                {"pixel_id": 0, "x_cm": 0.0, "y_cm": 0.0, "enabled": 1},
                {"pixel_id": 1, "x_cm": 1.0, "y_cm": 0.0, "enabled": 0},
            ],
        },
        10.0,
    )

    assert camera.get_pixel_shape() == 2
    assert camera.get_pixel_diameter() == pytest.approx(0.5)
    assert camera.get_number_of_pixels() == 2
    assert camera.pixels["pix_on"] == [True, False]


def test_read_pixel_list_from_dict_nonunique_diameter():
    camera_dict = {
        "pixel_size": np.array([0.5, 0.6, 0.5, 0.5]),
        "common_pixel_shape": 1,
        "pixel_x": np.array([0, 1, 0, 1]),
        "pixel_y": np.array([0, 0, 1, 1]),
    }
    with pytest.raises(ValueError, match="Pixel diameter is not unique"):
        Camera.read_pixel_list_from_dict(camera_dict)


def test_get_pixel_active_solid_angle_hexagonal(simple_camera_dict):
    camera = Camera("TestTel", None, 5.6, camera_config_dict=simple_camera_dict)
    solid_angle = camera.get_pixel_active_solid_angle()
    expected = (0.5**2 * np.sqrt(3) / 2) / (5.6**2)
    assert solid_angle == pytest.approx(expected)


def test_calc_neighbor_pixels_square():
    camera_dict = {
        "pixel_size": np.array([0.5, 0.5, 0.5, 0.5]),
        "common_pixel_shape": 2,
        "pixel_x": np.array([0, 1, 0, 1]),
        "pixel_y": np.array([0, 0, 1, 1]),
    }
    camera = Camera("TestTel", None, 5.6, camera_config_dict=camera_dict)
    neighbors = camera._calc_neighbor_pixels(camera.pixels)
    assert len(neighbors) == 4


def test_get_edge_pixels_cached(simple_camera_dict):
    camera = Camera("TestTel", None, 5.6, camera_config_dict=simple_camera_dict)
    edge1 = camera.get_edge_pixels()
    edge2 = camera.get_edge_pixels()
    assert edge1 == edge2


def test_calc_fov_method(simple_camera_dict):
    camera = Camera("TestTel", None, 5.6, camera_config_dict=simple_camera_dict)
    fov, avg_edge_dist = camera.calc_fov()
    assert fov > 0
    assert avg_edge_dist > 0


def test_get_camera_fill_factor(simple_camera_dict):
    camera = Camera("TestTel", None, 5.6, camera_config_dict=simple_camera_dict)
    fill_factor = camera.get_camera_fill_factor()
    assert 0 < fill_factor <= 1


def test_read_pixel_list_from_file(tmp_path):
    camera_file = tmp_path / "camera_test.dat"
    content = """PixType 0 0 1 1 3 0.5 5 "angle.dat" "wavelength.dat"
Rotate 30.0
Pixel 0 1 1 0.0 0.5 0.5 0 0 1 1
Pixel 1 1 1 1.0 0.5 0.5 0 0 1 1
"""
    camera_file.write_text(content)
    camera = Camera("TestTel", str(camera_file), 5.6)
    assert camera.get_number_of_pixels() == 2
    assert camera.get_pixel_diameter() == pytest.approx(0.5)
    assert camera.get_pixel_shape() == 3
