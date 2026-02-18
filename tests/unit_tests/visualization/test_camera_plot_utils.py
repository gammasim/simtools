#!/usr/bin/python3

from unittest import mock

import matplotlib.patches as mpatches

from simtools.visualization.camera_plot_utils import (
    add_pixel_legend,
    add_pixel_patch_collections,
    create_pixel_patches_by_type,
    pixel_shape,
    setup_camera_axis_properties,
)


def test_pixel_shape_hexagon():
    """Test hexagonal pixel shape."""
    camera = mock.MagicMock()
    camera.pixels = {
        "pixel_shape": 1,
        "pixel_diameter": 0.6,
        "orientation": 0.0,
    }

    shape = pixel_shape(camera, 0.0, 0.0)
    assert isinstance(shape, mpatches.RegularPolygon)


def test_pixel_shape_square():
    """Test square pixel shape."""
    camera = mock.MagicMock()
    camera.pixels = {
        "pixel_shape": 2,
        "pixel_diameter": 0.6,
        "orientation": 0.0,
    }

    shape = pixel_shape(camera, 0.0, 0.0)
    assert isinstance(shape, mpatches.Rectangle)


def test_pixel_shape_invalid():
    """Test invalid pixel shape returns None."""
    camera = mock.MagicMock()
    camera.pixels = {
        "pixel_shape": 99,
        "pixel_diameter": 0.6,
        "orientation": 0.0,
    }

    shape = pixel_shape(camera, 0.0, 0.0)
    assert shape is None


def test_create_pixel_patches_by_type():
    """Test patch creation by pixel type."""
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


def test_add_pixel_legend_empty():
    """Test add_pixel_legend with empty on pixels."""
    ax = mock.MagicMock()
    add_pixel_legend(ax, [], [])
    ax.legend.assert_not_called()


def test_add_pixel_legend_with_off_pixels():
    """Test add_pixel_legend with off pixels."""
    ax = mock.MagicMock()
    on_pixels = [mpatches.RegularPolygon((0, 0), numVertices=6, radius=0.5)]
    off_pixels = [mpatches.RegularPolygon((1, 1), numVertices=6, radius=0.5)]

    add_pixel_legend(ax, on_pixels, off_pixels)
    ax.legend.assert_called_once()


def test_add_pixel_patch_collections():
    """Test patch collection addition."""
    ax = mock.MagicMock()
    on_pixels = [mpatches.RegularPolygon((0, 0), numVertices=6, radius=0.5)]
    edge_pixels = [mpatches.RegularPolygon((1, 1), numVertices=6, radius=0.5)]
    off_pixels = [mpatches.RegularPolygon((2, 2), numVertices=6, radius=0.5)]

    add_pixel_patch_collections(ax, on_pixels, edge_pixels, off_pixels)

    assert ax.add_collection.call_count == 3


def test_setup_camera_axis_properties_grid_alpha():
    """Test axis setup with grid alpha."""
    ax = mock.MagicMock()
    camera = mock.MagicMock()
    camera.pixels = {"x": [0.0, 1.0], "y": [0.0, 1.0]}

    setup_camera_axis_properties(ax, camera, grid=True, grid_alpha=0.4)

    ax.grid.assert_called_with(True, alpha=0.4)


def test_setup_camera_axis_properties_padding():
    """Test axis setup with padding."""
    ax = mock.MagicMock()
    camera = mock.MagicMock()
    camera.pixels = {"x": [0.0, 1.0], "y": [0.0, 1.0]}

    setup_camera_axis_properties(ax, camera, padding=1.0)

    ax.set_xlim.assert_called_once()
    ax.set_ylim.assert_called_once()
