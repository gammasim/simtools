"""Comprehensive unit tests for plot_camera module."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from simtools.visualization.camera_plot_utils import (
    setup_camera_axis_properties,
)
from simtools.visualization.plot_camera import (
    _color_normalization,
    _parse_pixel_ids_to_print,
    _plot_axes_def,
    _plot_one_axis_def,
    plot_camera_pixel_layout_from_args,
    plot_pixel_layout,
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
    camera.effective_focal_length = [2923.7, 0.0, 0.0, 0.0, 0.0]
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
    camera.effective_focal_length = [2923.7, 0.0, 0.0, 0.0, 0.0]
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
    camera.effective_focal_length = [2923.7, 0.0, 0.0, 0.0, 0.0]
    camera.get_edge_pixels.return_value = [0]
    return camera


def test_color_normalization_none_image():
    result = _color_normalization(None, "viridis")
    assert result == (None, None, None)


def test_plot_layout_with_no_image(simple_camera):
    fig = plot_pixel_layout_with_image(simple_camera, image=None)
    assert fig is not None


def test_color_normalization_linear():
    image = np.array([0.0, 0.5, 1.0])
    colors, _, _ = _color_normalization(image, "viridis", norm_type="lin")
    assert colors is not None
    assert len(colors) == 3


def test_color_normalization_log():
    image = np.array([1.0, 10.0, 100.0])
    colors, _, _ = _color_normalization(image, "viridis", norm_type="log")
    assert colors is not None


def test_setup_camera_axis_with_scale_factor(camera_hexagon):
    ax = MagicMock()
    setup_camera_axis_properties(ax, camera_hexagon, y_scale_factor=1.42)
    ax.axis.assert_called_once()


def test_setup_camera_axis_grid_no_alpha(camera_hexagon):
    ax = MagicMock()
    setup_camera_axis_properties(ax, camera_hexagon, grid=True)
    ax.grid.assert_called_with(True)


def test_setup_camera_axis_below(camera_hexagon):
    ax = MagicMock()
    setup_camera_axis_properties(ax, camera_hexagon, axis_below=True)
    ax.set_axisbelow.assert_called_with(True)


def test_plot_layout_with_image_values(simple_camera):
    image = np.array([0.0, 0.5, 1.0])
    fig = plot_pixel_layout_with_image(simple_camera, image=image)
    assert fig is not None


def test_plot_layout_with_image_no_bar(camera_square):
    image = np.array([0.0, 0.5, 1.0])
    fig = plot_pixel_layout_with_image(camera_square, image=image, add_color_bar=False)
    assert fig is not None


def test_plot_layout_with_image_axes(camera_square):
    _, ax = plt.subplots()
    image = np.array([0.0, 0.5, 1.0])
    fig = plot_pixel_layout_with_image(camera_square, image=image, ax=ax)
    assert fig is not None


def test_plot_one_axis_no_invert():
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


def _axis_direction(annotation_call):
    """Return the vector from the axis origin to its labelled arrow tip."""
    origin = np.asarray(annotation_call.kwargs["xy"])
    arrow_tip = np.asarray(annotation_call.kwargs["xytext"])
    return (arrow_tip - origin).tolist()


def test_plot_axes_dual_mirror_directions():
    camera = MagicMock()
    camera.telescope_name = "LST-01"
    plt_mock = MagicMock()

    with patch("simtools.visualization.plot_camera.is_two_mirror_telescope", return_value=True):
        _plot_axes_def(camera, plt_mock, 0.0)

    calls = plt_mock.gca.return_value.annotate.call_args_list
    assert _axis_direction(calls[2]) == pytest.approx([0.0, -0.1])  # x_cam: down
    assert _axis_direction(calls[3]) == pytest.approx([0.1, 0.0])  # y_cam: right


def test_plot_axes_single_mirror_directions():
    camera = MagicMock()
    camera.telescope_name = "MST-01"
    plt_mock = MagicMock()
    rotate_angle = np.deg2rad(30.0)

    with patch("simtools.visualization.plot_camera.is_two_mirror_telescope", return_value=False):
        _plot_axes_def(camera, plt_mock, rotate_angle)

    calls = plt_mock.gca.return_value.annotate.call_args_list
    assert _axis_direction(calls[0]) == pytest.approx([-0.05, -np.sqrt(3) * 0.05])
    assert _axis_direction(calls[1]) == pytest.approx([-np.sqrt(3) * 0.05, 0.05])
    assert _axis_direction(calls[2]) == pytest.approx([0.0, -0.1])  # x_cam: down
    assert _axis_direction(calls[3]) == pytest.approx([-0.1, 0.0])  # y_cam: left


def test_plot_axes_large_rotation():
    camera = MagicMock()
    camera.telescope_name = "LST-01"
    camera.pixels = {"rotate_angle": 2.0}
    plt_mock = MagicMock()

    with patch("simtools.visualization.plot_camera.is_two_mirror_telescope", return_value=False):
        _plot_axes_def(camera, plt_mock, np.deg2rad(120))

    assert plt_mock.gca.return_value.annotate.call_count >= 2


def test_plot_pixel_layout_does_not_mutate_camera_coordinates(simple_camera):
    simple_camera.telescope_name = "LSTN-01"
    original_y = list(simple_camera.pixels["y"])

    with (
        patch("simtools.visualization.plot_camera.is_two_mirror_telescope", return_value=False),
        patch(
            "simtools.visualization.plot_camera.create_pixel_patches_by_type",
            return_value=([], [], []),
        ) as mock_create_patches,
    ):
        figure = plot_pixel_layout(simple_camera, camera_in_sky_coor=False)

    plotted_camera = mock_create_patches.call_args.args[0]
    assert plotted_camera.pixels["y"] == [-value for value in original_y]
    assert simple_camera.pixels["y"] == original_y
    plt.close(figure)


def _mock_camera(n_pixels=1855):
    camera = MagicMock()
    camera.get_number_of_pixels.return_value = n_pixels
    camera.calc_fov.return_value = (4.5, 120.0)
    return camera


def test_parse_pixel_ids_to_print_integer():
    camera = _mock_camera()
    assert _parse_pixel_ids_to_print(50, camera) == 50


def test_parse_pixel_ids_to_print_zero_returns_minus_one():
    camera = _mock_camera()
    assert _parse_pixel_ids_to_print(0, camera) == -1


def test_parse_pixel_ids_to_print_all():
    camera = _mock_camera(n_pixels=1855)
    assert _parse_pixel_ids_to_print("All", camera) == 1855


def test_parse_pixel_ids_to_print_invalid_raises():
    camera = _mock_camera()
    with pytest.raises(ValueError, match="must be integer or 'All'"):
        _parse_pixel_ids_to_print("invalid", camera)


def test_plot_camera_pixel_layout_from_args(tmp_test_directory):
    args_dict = {
        "site": "North",
        "telescope": "LSTN-01",
        "model_version": "5.0.0",
        "camera_in_sky_coor": False,
        "print_pixels_id": 10,
    }
    io_handler = MagicMock()
    io_handler.get_output_directory.return_value = Path(str(tmp_test_directory))
    app_context = SimpleNamespace(args=args_dict, io_handler=io_handler)

    mock_camera = _mock_camera()
    mock_tel_model = MagicMock()
    mock_tel_model.name = "LSTN-01"
    mock_tel_model.camera = mock_camera
    mock_tel_model.get_telescope_effective_focal_length.return_value = 2800.0

    with (
        patch(
            "simtools.visualization.plot_camera.TelescopeModel", return_value=mock_tel_model
        ) as mock_tm,
        patch("simtools.visualization.plot_camera.plot_pixel_layout") as mock_plot,
        patch("simtools.visualization.plot_camera.save_figure") as mock_save,
    ):
        plot_camera_pixel_layout_from_args(app_context)

        mock_tm.assert_called_once_with(
            site="North",
            telescope_name="LSTN-01",
            model_version="5.0.0",
            label="plot_camera_pixel_layout",
        )
        mock_tel_model.export_model_files.assert_called_once()
        mock_camera.calc_fov.assert_called_once()
        mock_plot.assert_called_once_with(mock_camera, False, 10)
        mock_save.assert_called_once()
