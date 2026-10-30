#!/usr/bin/python3

from unittest import mock

import matplotlib.patches as mpatches
import pytest

from simtools.visualization.camera_plot_utils import (
    add_pixel_legend,
    create_pixel_patches_by_type,
    pixel_shape,
    setup_camera_axis_properties,
)
from simtools.visualization.matplotlib_backend import pyplot as plt


@pytest.mark.parametrize(
    ("shape_code", "expected_type"),
    [(1, mpatches.RegularPolygon), (2, mpatches.Rectangle), (99, type(None))],
)
def test_pixel_shape(shape_code, expected_type):
    camera = mock.MagicMock()
    camera.pixels = {
        "pixel_shape": shape_code,
        "pixel_diameter": 0.6,
        "orientation": 0.0,
    }

    shape = pixel_shape(camera, 0.0, 0.0)
    assert isinstance(shape, expected_type) if shape is not None else expected_type is type(None)


def test_create_pixel_patches_by_type():
    camera = mock.MagicMock()
    camera.pixels = {
        "x": [0.0, 1.0, 2.0],
        "y": [0.0, 0.0, 0.0],
        "pix_on": [True, True, False],
        "pixel_shape": 1,
        "pixel_diameter": 0.6,
        "orientation": 0.0,
    }
    camera.get_edge_pixels.return_value = [1]

    on_pixels, edge_pixels, off_pixels = create_pixel_patches_by_type(camera)

    assert len(on_pixels) == 1
    assert len(edge_pixels) == 1
    assert len(off_pixels) == 1


def test_add_pixel_legend():
    ax = mock.MagicMock()
    on_pixels = [mpatches.RegularPolygon((0, 0), numVertices=6, radius=0.5)]
    off_pixels = [mpatches.RegularPolygon((1, 1), numVertices=6, radius=0.5)]

    add_pixel_legend(ax, on_pixels, off_pixels)
    ax.legend.assert_called_once()


@pytest.mark.parametrize(
    ("scale", "padding", "expected_x", "expected_y"),
    [
        (1, 1, (-2, 2), (-3, 3)),
        (2, 0, (-1, 1), (-4, 4)),
    ],
)
def test_camera_axis_preserves_limits_when_drawn(scale, padding, expected_x, expected_y, caplog):
    camera = mock.MagicMock()
    camera.pixels = {"x": [-1, 1], "y": [-2, 2]}
    fig, ax = plt.subplots(figsize=(8, 3))
    try:
        setup_camera_axis_properties(ax, camera, y_scale_factor=scale, padding=padding)
        fig.canvas.draw()
        assert ax.get_xlim() == pytest.approx(expected_x)
        assert ax.get_ylim() == pytest.approx(expected_y)
        assert "Ignoring fixed" not in caplog.text
    finally:
        plt.close(fig)
