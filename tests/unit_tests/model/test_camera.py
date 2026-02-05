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


def test_find_neighbors_square():
    x_pos = np.array([0, 0, 1, 1])
    y_pos = np.array([0, 1, 0, 1])

    # Test with radius 1
    radius_1 = 1.0
    expected_neighbors_radius_1 = [
        [1, 2],
        [0, 3],
        [0, 3],
        [1, 2],
    ]
    neighbors_radius_1 = Camera._find_neighbors(x_pos, y_pos, radius_1)
    assert neighbors_radius_1 == expected_neighbors_radius_1

    # Test with radius sqrt(2)
    radius_sqrt_2 = np.sqrt(2)
    expected_neighbors_radius_sqrt_2 = [
        [1, 2, 3],
        [0, 2, 3],
        [0, 1, 3],
        [0, 1, 2],
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
    Camera.validate_pixels(pixels, camera_config_file)


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


def test_initialize_pixel_dict():
    """Test pixel dictionary initialization."""
    pixels = Camera.initialize_pixel_dict()
    assert pixels["pixel_diameter"] == 9999
    assert pixels["pixel_shape"] == 9999
    assert pixels["rotate_angle"] == 0
    assert pixels["x"] == []


def test_read_pixel_list_from_dict(simple_camera_dict):
    """Test reading pixel list from dictionary."""
    pixels = Camera.read_pixel_list_from_dict(simple_camera_dict)
    assert pixels["pixel_diameter"] == 0.5
    assert pixels["pixel_shape"] == 1
    assert len(pixels["x"]) == 4
    assert pixels["pix_id"] == [0, 1, 2, 3]


def test_read_pixel_list_from_dict_nonunique_diameter():
    """Test error when pixel diameters are not unique."""
    camera_dict = {
        "pixel_size": np.array([0.5, 0.6, 0.5, 0.5]),
        "common_pixel_shape": 1,
        "pixel_x": np.array([0, 1, 0, 1]),
        "pixel_y": np.array([0, 0, 1, 1]),
    }
    with pytest.raises(ValueError, match="Pixel diameter is not unique"):
        Camera.read_pixel_list_from_dict(camera_dict)


def test_process_line_pixtype():
    """Test processing PixType line."""
    pixels = Camera.initialize_pixel_dict()
    line = 'PixType 0 0 1 1 3 0.5 5 "angle_file.dat" "wavelength_file.dat"'
    Camera.process_line(line, pixels)
    assert pixels["pixel_shape"] == 3
    assert pixels["pixel_diameter"] == 0.5
    assert pixels["lightguide_efficiency_angle_file"] == "angle_file.dat"
    assert pixels["lightguide_efficiency_wavelength_file"] == "wavelength_file.dat"


def test_process_line_rotate():
    """Test processing Rotate line."""
    pixels = Camera.initialize_pixel_dict()
    line = "Rotate 30.0"
    Camera.process_line(line, pixels)
    assert pixels["rotate_angle"] == pytest.approx(np.deg2rad(30.0))


def test_process_line_pixel():
    """Test processing Pixel line."""
    pixels = Camera.initialize_pixel_dict()
    line = "Pixel 5 1 1 10.5 20.3 0.5 0 0 1 1"
    Camera.process_line(line, pixels)
    assert len(pixels["x"]) == 1
    assert pixels["x"][0] == pytest.approx(1.0)
    assert pixels["y"][0] == pytest.approx(10.5)
    assert pixels["pix_id"] == [5]
    assert pixels["pix_on"] == [True]


def test_process_line_pixel_off():
    """Test processing Pixel line with pix_on=0."""
    pixels = Camera.initialize_pixel_dict()
    line = "Pixel 5 1 1 10.5 20.3 0 0 0 0 0"
    Camera.process_line(line, pixels)
    assert pixels["pix_on"] == [False]


def test_camera_from_dict(simple_camera_dict):
    """Test Camera initialization from dict."""
    camera = Camera(
        telescope_name="TestTel",
        camera_config_file=None,
        focal_length=5.6,
        camera_config_dict=simple_camera_dict,
    )
    assert camera.telescope_name == "TestTel"
    assert camera.focal_length == 5.6
    assert camera.get_number_of_pixels() == 4


def test_get_pixel_diameter(simple_camera_dict):
    """Test getting pixel diameter."""
    camera = Camera("TestTel", None, 5.6, camera_config_dict=simple_camera_dict)
    assert camera.get_pixel_diameter() == 0.5


def test_get_pixel_shape_hexagonal(simple_camera_dict):
    """Test getting pixel shape for hexagonal."""
    camera = Camera("TestTel", None, 5.6, camera_config_dict=simple_camera_dict)
    assert camera.get_pixel_shape() in [1, 3]


def test_get_pixel_shape_square():
    """Test getting pixel shape for square."""
    camera_dict = {
        "pixel_size": np.array([0.5, 0.5]),
        "common_pixel_shape": 2,
        "pixel_x": np.array([0, 1]),
        "pixel_y": np.array([0, 0]),
    }
    camera = Camera("TestTel", None, 5.6, camera_config_dict=camera_dict)
    assert camera.get_pixel_shape() == 2


def test_get_lightguide_efficiency_files(simple_camera_dict):
    """Test getting lightguide efficiency file names."""
    camera = Camera("TestTel", None, 5.6, camera_config_dict=simple_camera_dict)
    assert camera.get_lightguide_efficiency_angle_file_name() == "none"
    assert camera.get_lightguide_efficiency_wavelength_file_name() == "none"


def test_get_pixel_active_solid_angle_hexagonal(simple_camera_dict):
    """Test active solid angle for hexagonal pixels."""
    camera = Camera("TestTel", None, 5.6, camera_config_dict=simple_camera_dict)
    solid_angle = camera.get_pixel_active_solid_angle()
    expected = (0.5**2 * np.sqrt(3) / 2) / (5.6**2)
    assert solid_angle == pytest.approx(expected)


def test_get_pixel_active_solid_angle_square():
    """Test active solid angle for square pixels."""
    camera_dict = {
        "pixel_size": np.array([0.5, 0.5]),
        "common_pixel_shape": 2,
        "pixel_x": np.array([0, 1]),
        "pixel_y": np.array([0, 0]),
    }
    camera = Camera("TestTel", None, 5.6, camera_config_dict=camera_dict)
    solid_angle = camera.get_pixel_active_solid_angle()
    expected = (0.5**2) / (5.6**2)
    assert solid_angle == pytest.approx(expected)


def test_rotate_pixels_hexagonal_shape_1():
    """Test rotating hexagonal pixels shape 1."""
    pixels_dict = Camera.initialize_pixel_dict()
    pixels_dict["rotate_angle"] = np.deg2rad(30)
    pixels_dict["pixel_shape"] = 1
    pixels_dict["pixel_diameter"] = 0.5
    pixels_dict["x"] = [0.0, 1.0]
    pixels_dict["y"] = [0.0, 1.0]
    camera_dict = {
        "pixel_size": np.array([0.5, 0.5]),
        "common_pixel_shape": 1,
        "pixel_x": np.array([0, 1]),
        "pixel_y": np.array([0, 1]),
    }
    camera = Camera("TestTel", "dummy.dat", 5.6, camera_config_dict=camera_dict)
    rotated = camera._rotate_pixels(pixels_dict)
    assert "orientation" in rotated
    assert rotated["orientation"] != 0


def test_rotate_pixels_hexagonal_shape_3():
    """Test rotating hexagonal pixels shape 3."""
    pixels_dict = Camera.initialize_pixel_dict()
    pixels_dict["pixel_shape"] = 3
    pixels_dict["pixel_diameter"] = 0.5
    pixels_dict["rotate_angle"] = 0
    pixels_dict["x"] = [0.0, 1.0]
    pixels_dict["y"] = [0.0, 1.0]
    camera_dict = {
        "pixel_size": np.array([0.5, 0.5]),
        "common_pixel_shape": 3,
        "pixel_x": np.array([0, 1]),
        "pixel_y": np.array([0, 1]),
    }
    camera = Camera("TestTel", "dummy.dat", 5.6, camera_config_dict=camera_dict)
    rotated = camera._rotate_pixels(pixels_dict)
    assert rotated["orientation"] == pytest.approx(-60.0)


def test_calc_neighbor_pixels_hexagonal(simple_camera_dict):
    """Test neighbor calculation for hexagonal pixels."""
    camera = Camera("TestTel", None, 5.6, camera_config_dict=simple_camera_dict)
    neighbors = camera._calc_neighbor_pixels(camera.pixels)
    assert isinstance(neighbors, list)
    assert len(neighbors) == 4


def test_calc_neighbor_pixels_square():
    """Test neighbor calculation for square pixels."""
    camera_dict = {
        "pixel_size": np.array([0.5, 0.5, 0.5, 0.5]),
        "common_pixel_shape": 2,
        "pixel_x": np.array([0, 1, 0, 1]),
        "pixel_y": np.array([0, 0, 1, 1]),
    }
    camera = Camera("TestTel", None, 5.6, camera_config_dict=camera_dict)
    neighbors = camera._calc_neighbor_pixels(camera.pixels)
    assert len(neighbors) == 4


def test_get_neighbor_pixels_cached(simple_camera_dict):
    """Test that neighbor pixels are cached."""
    camera = Camera("TestTel", None, 5.6, camera_config_dict=simple_camera_dict)
    neighbors1 = camera.get_neighbor_pixels()
    neighbors2 = camera.get_neighbor_pixels()
    assert neighbors1 is neighbors2


def test_calc_edge_pixels_hexagonal():
    """Test edge pixel calculation for hexagonal."""
    camera_dict = {
        "pixel_size": np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
        "common_pixel_shape": 1,
        "pixel_x": np.array([0, 1, 0, 1, 0.5]),
        "pixel_y": np.array([0, 0, 1, 1, 0.5]),
    }
    camera = Camera("TestTel", None, 5.6, camera_config_dict=camera_dict)
    neighbors = camera.get_neighbor_pixels()
    edge_pixels = camera._calc_edge_pixels(camera.pixels, neighbors)
    assert isinstance(edge_pixels, list)


def test_calc_edge_pixels_square():
    """Test edge pixel calculation for square."""
    camera_dict = {
        "pixel_size": np.array([0.5, 0.5, 0.5]),
        "common_pixel_shape": 2,
        "pixel_x": np.array([0, 1, 0]),
        "pixel_y": np.array([0, 0, 1]),
    }
    camera = Camera("TestTel", None, 5.6, camera_config_dict=camera_dict)
    neighbors = camera.get_neighbor_pixels()
    edge_pixels = camera._calc_edge_pixels(camera.pixels, neighbors)
    assert len(edge_pixels) > 0


def test_get_edge_pixels_cached(simple_camera_dict):
    """Test that edge pixels are cached."""
    camera = Camera("TestTel", None, 5.6, camera_config_dict=simple_camera_dict)
    edge1 = camera.get_edge_pixels()
    edge2 = camera.get_edge_pixels()
    assert edge1 == edge2


def test_calc_fov(simple_camera_dict):
    """Test FOV calculation."""
    camera = Camera("TestTel", None, 5.6, camera_config_dict=simple_camera_dict)
    edge_pixels = camera.get_edge_pixels()
    fov, avg_edge_dist = camera._calc_fov(camera.pixels["x"], camera.pixels["y"], edge_pixels, 5.6)
    assert fov > 0
    assert avg_edge_dist > 0


def test_calc_fov_method(simple_camera_dict):
    """Test calc_fov method."""
    camera = Camera("TestTel", None, 5.6, camera_config_dict=simple_camera_dict)
    fov, avg_edge_dist = camera.calc_fov()
    assert fov > 0
    assert avg_edge_dist > 0


def test_get_camera_fill_factor(simple_camera_dict):
    """Test camera fill factor calculation."""
    camera = Camera("TestTel", None, 5.6, camera_config_dict=simple_camera_dict)
    fill_factor = camera.get_camera_fill_factor()
    assert 0 < fill_factor <= 1


def test_add_additional_neighbors():
    """Test adding additional neighbors for square pixels."""
    x_pos = np.array([0.0, 0.0, 1.0, 1.0, 2.0])
    y_pos = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    nn = [2]
    camera_dict = {
        "pixel_size": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        "common_pixel_shape": 2,
        "pixel_x": x_pos,
        "pixel_y": y_pos,
    }
    camera = Camera("TestTel", None, 5.6, camera_config_dict=camera_dict)
    camera._add_additional_neighbors(0, nn, x_pos, y_pos, 1.4, 0.2)
    assert len(nn) >= 1


def test_find_adjacent_neighbor_pixels():
    """Test finding adjacent neighbors for square pixels."""
    x_pos = np.array([0.0, 0.0, 1.0, 1.0])
    y_pos = np.array([0.0, 1.0, 0.0, 1.0])
    camera_dict = {
        "pixel_size": np.array([1.0, 1.0, 1.0, 1.0]),
        "common_pixel_shape": 2,
        "pixel_x": x_pos,
        "pixel_y": y_pos,
    }
    camera = Camera("TestTel", None, 5.6, camera_config_dict=camera_dict)
    neighbors = camera._find_adjacent_neighbor_pixels(x_pos, y_pos, 1.1, 0.2)
    assert len(neighbors) == 4


def test_read_pixel_list_from_file(tmp_path):
    """Test reading pixel list from file."""
    camera_file = tmp_path / "camera_test.dat"
    content = """PixType 0 0 1 1 3 0.5 5 "angle.dat" "wavelength.dat"
Rotate 30.0
Pixel 0 1 1 0.0 0.5 0.5 0 0 1 1
Pixel 1 1 1 1.0 0.5 0.5 0 0 1 1
"""
    camera_file.write_text(content)
    camera = Camera("TestTel", str(camera_file), 5.6)
    assert camera.get_number_of_pixels() == 2
    assert camera.get_pixel_diameter() == 0.5
    assert camera.get_pixel_shape() == 3
