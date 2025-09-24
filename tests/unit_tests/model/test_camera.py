#!/usr/bin/python3

import logging

import numpy as np
import pytest

from simtools.model.camera import Camera

logger = logging.getLogger()


@pytest.fixture
def camera_config_file():
    return "dummy.dat"


def test_focal_length():
    with pytest.raises(ValueError, match="The focal length must be larger than zero"):
        Camera(
            telescope_model_name="test_camera", camera_config_file="test_config", focal_length=-1
        )


def test_get_number_of_pixels(telescope_model_lst):
    telescope_model_lst.export_model_files()
    assert telescope_model_lst.camera.get_number_of_pixels() == 1855  # Value for LST


def test_pixel_solid_angle(telescope_model_lst):
    tel_model = telescope_model_lst
    telescope_model_lst.export_model_files()
    pix_solid_angle = tel_model.camera.get_pixel_active_solid_angle()
    logger.debug(f"Pixel solid angle is {pix_solid_angle}")

    assert pix_solid_angle == pytest.approx(2.43 / 1.0e6, 0.01)  # Value for LST


def test_find_neighbors_square():
    x_pos = np.array([0, 0, 1, 1])
    y_pos = np.array([0, 1, 0, 1])

    # Test with radius 1
    radius_1 = 1.0
    expected_neighbors_radius_1 = [
        [1, 2],  # Neighbors of point (0, 0)
        [0, 3],  # Neighbors of point (0, 1)
        [0, 3],  # Neighbors of point (1, 0)
        [1, 2],  # Neighbors of point (1, 1)
    ]
    neighbors_radius_1 = Camera._find_neighbors(x_pos, y_pos, radius_1)
    assert neighbors_radius_1 == expected_neighbors_radius_1

    # Test with radius sqrt(2)
    radius_sqrt_2 = np.sqrt(2)
    expected_neighbors_radius_sqrt_2 = [
        [1, 2, 3],  # Neighbors of point (0, 0)
        [0, 2, 3],  # Neighbors of point (0, 1)
        [0, 1, 3],  # Neighbors of point (1, 0)
        [0, 1, 2],  # Neighbors of point (1, 1)
    ]
    neighbors_radius_sqrt_2 = Camera._find_neighbors(x_pos, y_pos, radius_sqrt_2)
    assert neighbors_radius_sqrt_2 == expected_neighbors_radius_sqrt_2


def test_validate_pixels_valid(camera_config_file):
    """Test validate_pixels with a valid pixel dictionary."""
    pixels = {
        "pixel_diameter": 10,
        "pixel_shape": 1,
        "pixel_spacing": 12,
        "lightguide_efficiency_angle_file": "none",
        "lightguide_efficiency_wavelength_file": "none",
        "rotate_angle": 0,
        "x": [],
        "y": [],
        "pix_id": [],
        "pix_on": [],
    }
    Camera.validate_pixels(pixels, camera_config_file)  # Should not raise an exception


def test_validate_pixels_invalid_diameter(camera_config_file):
    """Test validate_pixels with an invalid pixel diameter."""
    pixels = {
        "pixel_diameter": 9999,
        "pixel_shape": 1,
        "pixel_spacing": 12,
        "lightguide_efficiency_angle_file": "none",
        "lightguide_efficiency_wavelength_file": "none",
        "rotate_angle": 0,
        "x": [],
        "y": [],
        "pix_id": [],
        "pix_on": [],
    }
    with pytest.raises(
        ValueError, match=f"Could not read the pixel diameter from {camera_config_file} file"
    ):
        Camera.validate_pixels(pixels, camera_config_file)


def test_validate_pixels_invalid_shape(camera_config_file):
    """Test validate_pixels with an invalid pixel shape."""
    pixels = {
        "pixel_diameter": 10,
        "pixel_shape": 4,
        "pixel_spacing": 12,
        "lightguide_efficiency_angle_file": "none",
        "lightguide_efficiency_wavelength_file": "none",
        "rotate_angle": 0,
        "x": [],
        "y": [],
        "pix_id": [],
        "pix_on": [],
    }
    with pytest.raises(
        ValueError,
        match=f"Pixel shape in {camera_config_file} unrecognized \\(has to be 1, 2 or 3\\)",
    ):
        Camera.validate_pixels(pixels, camera_config_file)
