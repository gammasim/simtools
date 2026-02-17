"""Comprehensive unit tests for plot_camera module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from simtools.visualization.camera_plot_utils import (
    create_pixel_patches_by_type,
    pixel_shape,
    setup_camera_axis_properties,
)
from simtools.visualization.plot_camera import (
    _color_normalization,
    _plot_axes_def,
    _plot_one_axis_def,
    plot_pixel_layout_with_image,
)


@pytest.fixture
def simple_camera():
    """Create a mock camera with basic properties."""
    camera = MagicMock()
    camera.telescope_name = "LST-01"
    camera.focal_length = 10.0
    camera.pixels = {
        "x": [0.0, 1.0, -1.0],
        "y": [0.0, 1.0, -1.0],
        "pix_id": [0, 1, 2],
        "pix_on": [True, True, False],
        "pixel_diameter": 0.5,
        "pixel_shape": 1,
        "rotate_angle": 0.0,
        "orientation": 0.0,
    }
    camera.calc_fov.return_value = (10.0, 5.0)
    camera.get_edge_pixels.return_value = [1]
    return camera


@pytest.fixture
def camera_hexagon():
    """Camera with hexagonal pixels."""
    camera = MagicMock()
    camera.telescope_name = "MST-01"
    camera.focal_length = 16.0
    camera.pixels = {
        "x": [0.0, 1.0, -1.0, 0.5, -0.5],
        "y": [0.0, 1.0, -1.0, 0.5, -0.5],
        "pix_id": [0, 1, 2, 3, 4],
        "pix_on": [True, True, True, False, True],
        "pixel_diameter": 0.3,
        "pixel_shape": 1,
        "rotate_angle": 0.0,
        "orientation": 0.0,
    }
    camera.calc_fov.return_value = (8.0, 4.0)
    camera.get_edge_pixels.return_value = [0, 3]
    return camera


@pytest.fixture
def camera_square():
    """Camera with square pixels."""
    camera = MagicMock()
    camera.telescope_name = "SST-01"
    camera.focal_length = 5.6
    camera.pixels = {
        "x": [0.0, 1.0, -1.0],
        "y": [0.0, 1.0, -1.0],
        "pix_id": [0, 1, 2],
        "pix_on": [True, True, False],
        "pixel_diameter": 0.5,
        "pixel_shape": 2,
        "rotate_angle": 0.0,
        "orientation": 0.0,
    }
    camera.calc_fov.return_value = (20.0, 10.0)
    camera.get_edge_pixels.return_value = [0]
    return camera


def test_pixel_shape_hexagon_type1(simple_camera):
    """Test hexagon pixel shape (type 1)."""
    shape = pixel_shape(simple_camera, 0.0, 0.0)
    assert shape is not None


def test_pixel_shape_hexagon_type3(camera_hexagon):
    """Test hexagon pixel shape (type 3)."""
    camera_hexagon.pixels["pixel_shape"] = 3
    shape = pixel_shape(camera_hexagon, 0.5, 0.5)
    assert shape is not None


def test_pixel_shape_square(camera_square):
    """Test square pixel shape."""
    shape = pixel_shape(camera_square, 0.0, 0.0)
    assert shape is not None


def test_pixel_shape_invalid(simple_camera):
    """Test invalid pixel shape returns None."""
    simple_camera.pixels["pixel_shape"] = 99
    shape = pixel_shape(simple_camera, 0.0, 0.0)
    assert shape is None


def test_pixel_shape_with_different_diameters(camera_hexagon):
    """Test pixel shape with various diameters."""
    for diameter in [0.1, 0.5, 1.0, 2.0]:
        camera_hexagon.pixels["pixel_diameter"] = diameter
        shape = pixel_shape(camera_hexagon, 0.0, 0.0)
        assert shape is not None


def test_pixel_shape_with_different_orientations(camera_hexagon):
    """Test pixel shape with various orientations."""
    for orientation in [0.0, np.pi / 6, np.pi / 4, np.pi / 2]:
        camera_hexagon.pixels["orientation"] = orientation
        shape = pixel_shape(camera_hexagon, 0.0, 0.0)
        assert shape is not None


def test_color_normalization_none_image():
    """Test normalization with None image."""
    result = _color_normalization(None, "viridis")
    assert result is None


def test_color_normalization_linear():
    """Test linear color normalization."""
    image = np.array([0.0, 0.5, 1.0])
    colors, _, _ = _color_normalization(image, "viridis", norm_type="lin")
    assert colors is not None
    assert len(colors) == 3


def test_color_normalization_log():
    """Test logarithmic color normalization."""
    image = np.array([1.0, 10.0, 100.0])
    colors, _, _ = _color_normalization(image, "viridis", norm_type="log")
    assert colors is not None


def test_color_normalization_with_vmin_vmax():
    """Test normalization with custom min/max."""
    image = np.array([0.0, 50.0, 100.0])
    colors, _, _ = _color_normalization(image, "plasma", norm_type="lin", vmin=0, vmax=100)
    assert colors is not None


def test_color_normalization_log_with_range():
    """Test log normalization with range."""
    image = np.array([0.1, 1.0, 10.0])
    colors, _, _ = _color_normalization(image, "viridis", norm_type="log", vmin=0.1, vmax=100)
    assert colors is not None


def test_color_normalization_different_colormaps():
    """Test normalization with different colormaps."""
    image = np.array([0.0, 0.5, 1.0])
    for cmap in ["plasma", "hot", "cool"]:
        colors, _, _ = _color_normalization(image, cmap, norm_type="lin")
        assert colors is not None


def test_create_pixel_patches_by_type_hexagon(camera_hexagon):
    """Test pixel patch creation with hexagonal pixels."""
    on, edge, off = create_pixel_patches_by_type(camera_hexagon)
    assert len(on) + len(edge) + len(off) == 5


def test_create_pixel_patches_by_type_square(camera_square):
    """Test pixel patch creation with square pixels."""
    on, edge, off = create_pixel_patches_by_type(camera_square)
    assert len(on) + len(edge) + len(off) == 3


def test_create_pixel_patches_by_type_edge_hex(camera_hexagon):
    """Test edge detection for hexagon."""
    camera_hexagon.get_edge_pixels.return_value = [0, 1]
    _, edge, _ = create_pixel_patches_by_type(camera_hexagon)
    assert len(edge) >= 1


def test_create_pixel_patches_by_type_edge_square(camera_square):
    """Test edge detection for square."""
    camera_square.get_edge_pixels.return_value = [0]
    _, edge, _ = create_pixel_patches_by_type(camera_square)
    assert len(edge) >= 1


def test_setup_camera_axis_with_scale_factor(camera_hexagon):
    """Test axis setup with y_scale_factor > 1.0."""
    ax = MagicMock()
    setup_camera_axis_properties(ax, camera_hexagon, y_scale_factor=1.42)
    ax.axis.assert_called_once()


def test_setup_camera_axis_with_padding(camera_hexagon):
    """Test axis setup with padding."""
    ax = MagicMock()
    setup_camera_axis_properties(ax, camera_hexagon, padding=0.5)
    ax.set_xlim.assert_called_once()
    ax.set_ylim.assert_called_once()


def test_setup_camera_axis_with_grid(camera_hexagon):
    """Test axis setup with grid and alpha."""
    ax = MagicMock()
    setup_camera_axis_properties(ax, camera_hexagon, grid=True, grid_alpha=0.3)
    ax.grid.assert_called_with(True, alpha=0.3)


def test_setup_camera_axis_grid_no_alpha(camera_hexagon):
    """Test axis setup with grid but no alpha."""
    ax = MagicMock()
    setup_camera_axis_properties(ax, camera_hexagon, grid=True)
    ax.grid.assert_called_with(True)


def test_setup_camera_axis_below(camera_hexagon):
    """Test axis below flag."""
    ax = MagicMock()
    setup_camera_axis_properties(ax, camera_hexagon, axis_below=True)
    ax.set_axisbelow.assert_called_with(True)


def test_setup_camera_axis_no_grid(camera_hexagon):
    """Test axis setup without grid."""
    ax = MagicMock()
    setup_camera_axis_properties(ax, camera_hexagon, grid=False)
    ax.grid.assert_not_called()


def test_plot_layout_with_image_values(simple_camera):
    """Test pixel layout with image data."""
    image = np.array([0.0, 0.5, 1.0])
    fig = plot_pixel_layout_with_image(simple_camera, image=image)
    assert fig is not None


def test_plot_layout_with_image_label(simple_camera):
    """Test pixel layout with custom colorbar label."""
    image = np.array([0.0, 0.5, 1.0])
    fig = plot_pixel_layout_with_image(simple_camera, image=image, color_bar_label="Custom")
    assert fig is not None


def test_plot_layout_with_image_no_bar(camera_square):
    """Test pixel layout without colorbar."""
    image = np.array([0.0, 0.5, 1.0])
    fig = plot_pixel_layout_with_image(camera_square, image=image, add_color_bar=False)
    assert fig is not None


def test_plot_layout_with_image_figsize(camera_hexagon):
    """Test pixel layout with custom figsize."""
    image = np.array([0.0, 0.5, 1.0, 0.7, 0.3])
    fig = plot_pixel_layout_with_image(camera_hexagon, image=image, figsize=(10, 10))
    assert fig is not None


def test_plot_layout_with_image_axes(camera_square):
    """Test pixel layout on existing axes."""
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    image = np.array([0.0, 0.5, 1.0])
    fig = plot_pixel_layout_with_image(camera_square, image=image, ax=ax)
    assert fig is not None
    plt.close("all")


def test_plot_layout_with_image_hex(camera_hexagon):
    """Test pixel layout with hexagonal pixels."""
    image = np.array([0.0, 0.5, 1.0, 0.7, 0.3])
    fig = plot_pixel_layout_with_image(camera_hexagon, image=image)
    assert fig is not None


def test_plot_layout_with_image_square(camera_square):
    """Test pixel layout with square pixels."""
    image = np.array([0.0, 0.5, 1.0])
    fig = plot_pixel_layout_with_image(camera_square, image=image)
    assert fig is not None


def test_plot_layout_with_image_log(camera_hexagon):
    """Test pixel layout with log normalization."""
    image = np.array([1.0, 10.0, 100.0, 5.0, 50.0])
    fig = plot_pixel_layout_with_image(camera_hexagon, image=image, norm="log")
    assert fig is not None


def test_plot_one_axis_no_invert():
    """Test single axis definition without inversion."""
    plt_mock = MagicMock()
    _plot_one_axis_def(
        plt_mock,
        x_title="X",
        y_title="Y",
        x_pos=0.7,
        y_pos=0.12,
        rotate_angle=0.0,
        fc="black",
        ec="black",
        invert_yaxis=False,
    )
    assert plt_mock.gca.return_value.annotate.call_count == 2


def test_plot_one_axis_with_invert():
    """Test single axis definition with inversion."""
    plt_mock = MagicMock()
    _plot_one_axis_def(
        plt_mock,
        x_title="X",
        y_title="Y",
        x_pos=0.8,
        y_pos=0.12,
        rotate_angle=np.pi / 2,
        fc="blue",
        ec="blue",
        invert_yaxis=True,
    )
    assert plt_mock.gca.return_value.annotate.call_count == 2


def test_plot_axes_dual_mirror():
    """Test axes definition for dual mirror telescope."""
    camera = MagicMock()
    camera.telescope_name = "LST-01"
    camera.pixels = {"rotate_angle": 0.5}
    plt_mock = MagicMock()

    with patch("simtools.visualization.plot_camera.is_two_mirror_telescope", return_value=True):
        _plot_axes_def(camera, plt_mock, 0.5)

    assert plt_mock.gca.return_value.annotate.call_count >= 2


def test_plot_axes_single_mirror():
    """Test axes definition for single mirror telescope."""
    camera = MagicMock()
    camera.telescope_name = "MST-01"
    camera.pixels = {"rotate_angle": 0.5}
    plt_mock = MagicMock()

    with patch("simtools.visualization.plot_camera.is_two_mirror_telescope", return_value=False):
        _plot_axes_def(camera, plt_mock, 0.5)

    assert plt_mock.gca.return_value.annotate.call_count >= 2


def test_plot_axes_large_rotation():
    """Test axes definition with large rotation angle."""
    camera = MagicMock()
    camera.telescope_name = "LST-01"
    camera.pixels = {"rotate_angle": 2.0}
    plt_mock = MagicMock()

    with patch("simtools.visualization.plot_camera.is_two_mirror_telescope", return_value=False):
        _plot_axes_def(camera, plt_mock, np.deg2rad(120))

    assert plt_mock.gca.return_value.annotate.call_count >= 2
