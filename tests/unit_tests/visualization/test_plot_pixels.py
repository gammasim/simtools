#!/usr/bin/python3

from pathlib import Path
from unittest import mock

import astropy.units as u
import matplotlib.patches as mpatches
import numpy as np
import pytest
from astropy.tests.helper import assert_quantity_allclose
from matplotlib.figure import Figure

from simtools.visualization import plot_pixels

# Constants
DUMMY_DAT_PATH = "tests/resources/pixel_layout.dat"


@mock.patch("simtools.visualization.plot_pixels.plot_pixel_layout_from_file")
@mock.patch("simtools.visualization.plot_pixels.visualize.save_figure")
@mock.patch("simtools.visualization.plot_pixels.db_handler.DatabaseHandler")
def test_plot(mock_db_handler, mock_save, mock_plot_layout):
    """Test the main plot function."""
    config = {
        "parameter": "pixel_layout",
        "site": "North",
        "telescope": "LSTN-01",
        "parameter_version": "1.0.0",
        "model_version": "6.0.0",
        "file_name": "test.dat",
    }

    # Mock database handler
    mock_db_instance = mock.MagicMock()
    mock_db_handler.return_value = mock_db_instance

    # Mock the plot layout function
    mock_fig = mock.MagicMock()
    mock_plot_layout.return_value = mock_fig

    # Mock IO handler
    with mock.patch("simtools.visualization.plot_pixels.io_handler.IOHandler") as mock_io:
        mock_io_instance = mock.MagicMock()
        mock_io.return_value = mock_io_instance
        mock_io_instance.get_output_directory.return_value = Path("/test/path")

        # Call function
        plot_pixels.plot(config, "test.png")

        # Verify database calls
        mock_db_instance.export_model_file.assert_called_once_with(
            parameter="pixel_layout",
            site="North",
            array_element_name="LSTN-01",
            parameter_version="1.0.0",
            model_version="6.0.0",
            export_file_as_table=False,
        )

        # Verify plot calls
        expected_path = Path("/test/path/test.dat")
        mock_plot_layout.assert_called_once_with(expected_path, "LSTN-01", pixels_id_to_print=80)
        mock_save.assert_called_once_with(mock_fig, "test.png")


def test__create_patch():
    """Test patch creation for different pixel shapes."""
    x, y = 1.0, 1.0
    diameter = 2.0

    # Test circular pixel (shape=0)
    patch = plot_pixels._create_patch(x, y, diameter, 0)
    assert isinstance(patch, mpatches.Circle)
    assert patch.center == (x, y)
    assert patch.radius == pytest.approx(diameter / 2)

    # Test hexagonal pixel flat-x (shape=1)
    patch = plot_pixels._create_patch(x, y, diameter, 1)
    assert isinstance(patch, mpatches.RegularPolygon)
    assert patch.xy == (x, y)
    assert np.isclose(patch.radius, diameter / np.sqrt(3))
    assert patch.orientation == 0

    # Test square pixel (shape=2)
    patch = plot_pixels._create_patch(x, y, diameter, 2)
    assert isinstance(patch, mpatches.Rectangle)
    assert patch.get_xy() == (x - diameter / 2, y - diameter / 2)
    assert patch.get_width() == pytest.approx(diameter)
    assert patch.get_height() == pytest.approx(diameter)

    # Test hexagonal pixel flat-y (shape=3)
    patch = plot_pixels._create_patch(x, y, diameter, 3)
    assert isinstance(patch, mpatches.RegularPolygon)
    assert patch.xy == (x, y)
    assert np.isclose(patch.radius, diameter / np.sqrt(3))
    assert np.isclose(patch.orientation, np.deg2rad(30))


def test__is_edge_pixel():
    """Test edge pixel detection."""
    # Create a 3x3 grid with circular pixels
    # Distance between adjacent pixels = 1.0
    # Distance between diagonal pixels = sqrt(2)
    x_pos = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])
    y_pos = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1])
    module_ids = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
    shape = 0  # circle shape
    pixel_spacing = 1.42  # max distance between pixels
    module_gap = 0.5
    current_module_id = 1

    # Test center pixel (should not be edge)
    assert not plot_pixels._is_edge_pixel(
        0, 0, x_pos, y_pos, module_ids, pixel_spacing, module_gap, shape, current_module_id
    )
    # Test corner pixel (should be edge)
    assert plot_pixels._is_edge_pixel(
        -1, -1, x_pos, y_pos, module_ids, pixel_spacing, module_gap, shape, current_module_id
    )

    x_pos = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])
    y_pos = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1])
    module_ids = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
    shape = 2  # square shape
    pixel_spacing = 1.0
    # Test center pixel (should not be edge)
    assert not plot_pixels._is_edge_pixel(
        0, 0, x_pos, y_pos, module_ids, pixel_spacing, module_gap, shape, current_module_id
    )
    # Test corner pixel (should be edge)
    assert plot_pixels._is_edge_pixel(
        -1, -1, x_pos, y_pos, module_ids, pixel_spacing, module_gap, shape, current_module_id
    )

    # Hexagonal shape
    x_pos = np.array([0.0, 1.0, 0.5, -0.5, -1.0, -0.5, 0.5])
    y_pos = np.array([0.0, 0.0, 0.866, 0.866, 0.0, -0.866, -0.866])
    shape = 1
    pixel_spacing = 1.0
    # Test center pixel (should not be edge)
    assert not plot_pixels._is_edge_pixel(
        0, 0, x_pos, y_pos, module_ids, pixel_spacing, module_gap, shape, current_module_id
    )
    # Test corner pixel (should be edge)
    assert plot_pixels._is_edge_pixel(
        -1, -1, x_pos, y_pos, module_ids, pixel_spacing, module_gap, shape, current_module_id
    )

    # Test that invalid shape raises ValueError
    shape = 100
    with pytest.raises(ValueError, match="Unsupported pixel shape: 100"):
        plot_pixels._is_edge_pixel(
            0, 0, x_pos, y_pos, module_ids, pixel_spacing, module_gap, shape, current_module_id
        )


def test__count_neighbors():
    """Test neighbor counting function."""
    # square pixel positions
    x_pos = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])
    y_pos = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1])
    module_ids = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
    pixel_spacing = 1.0
    module_gap = 0.5
    current_module_id = 1

    # Test center pixel (should have 4 neighbors)
    count = plot_pixels._count_neighbors(
        0, 0, x_pos, y_pos, module_ids, pixel_spacing, module_gap, current_module_id
    )
    assert count == 4

    # Test corner pixel
    count = plot_pixels._count_neighbors(
        -1, -1, x_pos, y_pos, module_ids, pixel_spacing, module_gap, current_module_id
    )
    assert count < 4

    # Test with different modules
    module_ids = np.array([1, 1, 2, 1, 1, 2, 1, 1, 2])
    assert (
        plot_pixels._count_neighbors(1, 0, x_pos, y_pos, module_ids, pixel_spacing, module_gap, 1)
        < 4
    )

    # Hexagonal array positions (central pixel with 6 neighbors)
    x_pos = np.array(
        [
            0.0,
            1.0,
            0.5,
            -0.5,
            -1.0,
            -0.5,
            0.5,
            2.0,
            1.5,
            0.5,
            -0.5,
            -1.5,
            -2.0,
            -1.5,
            -0.5,
            0.5,
            1.5,
        ]
    )
    y_pos = np.array(
        [
            0.0,
            0.0,
            0.866,
            0.866,
            0.0,
            -0.866,
            -0.866,
            0.0,
            0.866,
            1.732,
            1.732,
            0.866,
            0.0,
            -0.866,
            -1.732,
            -1.732,
            -0.866,
        ]
    )
    module_ids = np.ones_like(x_pos)
    pixel_spacing = 1.0
    module_gap = 0.5
    current_module_id = 1

    # Test center pixel (should have 6 neighbors in hexagonal arrangement)
    count = plot_pixels._count_neighbors(
        0, 0, x_pos, y_pos, module_ids, pixel_spacing, module_gap, current_module_id
    )
    assert count == 6

    # Test edge pixel
    count = plot_pixels._count_neighbors(
        2.0, 0.0, x_pos, y_pos, module_ids, pixel_spacing, module_gap, current_module_id
    )
    assert count < 6

    # Empty list
    x_pos = np.array([])
    y_pos = np.array([])
    module_ids = np.array([])
    pixel_spacing = 1.5
    module_gap = 0.5
    current_module_id = 1

    count = plot_pixels._count_neighbors(
        0, 0, x_pos, y_pos, module_ids, pixel_spacing, module_gap, current_module_id
    )
    assert count == 0


@mock.patch("matplotlib.pyplot.subplots")
def test__create_pixel_plot(mock_subplots):
    """Test pixel plot creation."""
    mock_fig = mock.MagicMock()
    mock_ax = mock.MagicMock()
    # Configure the mock to return appropriate values
    mock_ax.get_xlim.return_value = (-10, 10)
    mock_ax.get_ylim.return_value = (-10, 10)

    mock_subplots.return_value = (mock_fig, mock_ax)

    pixel_data = plot_pixels._prepare_pixel_data(DUMMY_DAT_PATH, "LSTN-01")

    with mock.patch("simtools.visualization.plot_pixels._is_edge_pixel") as mock_is_edge:
        mock_is_edge.side_effect = lambda x, y, *args: x == -1.0

        with mock.patch("matplotlib.pyplot.text") as mock_text:
            fig = plot_pixels._create_pixel_plot(pixel_data, "LSTN-01", pixels_id_to_print=3)

            assert fig is not None
            mock_ax.set_aspect.assert_called_once_with("equal")
            assert mock_ax.add_collection.call_count == 1
            text_calls = mock_text.call_args_list
            pixel_id_calls = [call for call in text_calls if str(call[0][2]).isdigit()]
            assert len(pixel_id_calls) == 3


def test__read_pixel_config():
    """Test reading pixel configuration from file."""
    config = plot_pixels._read_pixel_config(DUMMY_DAT_PATH)

    assert_quantity_allclose(config["rotate_angle"], 10.0 * u.deg)
    assert config["pixel_shape"] == 2
    assert_quantity_allclose(config["pixel_spacing"], 0.640)
    assert_quantity_allclose(config["module_gap"], 0.02)
    assert_quantity_allclose(config["pixel_diameter"], 0.6)
    assert len(config["pixel_ids"]) == 31
    assert config["pixels_on"].count(True) == 30


@mock.patch("simtools.visualization.plot_pixels._read_pixel_config")
def test__prepare_pixel_data(mock_read):
    """Test pixel data preparation."""
    # Setup mock return value
    mock_read.return_value = {
        "x": [0.0, 1.0],
        "y": [0.0, 1.0],
        "pixel_ids": [1, 2],
        "pixels_on": [True, True],
        "pixel_shape": 1,
        "pixel_diameter": 0.5,
        "rotate_angle": 10.0 * u.deg,
        "pixel_spacing": 0.6,
        "module_gap": 0.2,
        "module_number": [1, 1],
    }

    # Call the function under test
    data = plot_pixels._prepare_pixel_data(DUMMY_DAT_PATH, "LSTN-01")

    assert "pixel_spacing" in data
    assert "module_gap" in data
    assert "module_number" in data
    assert data["pixel_spacing"] == pytest.approx(0.6)


def test__add_coordinate_axes():
    """Test coordinate axes addition."""
    mock_ax = mock.MagicMock()
    # Configure the mock to return appropriate values
    mock_ax.get_xlim.return_value = (-10, 10)
    mock_ax.get_ylim.return_value = (-10, 10)
    rotation = 90.0 * u.deg

    plot_pixels._add_coordinate_axes(mock_ax, rotation)

    assert mock_ax.arrow.call_count == 4
    assert mock_ax.text.call_count == 4


def test__add_legend():
    """Test legend addition."""
    # Test with non-empty pixel lists
    mock_ax = mock.MagicMock()
    on_pixels = [
        mpatches.RegularPolygon(xy=(0, 0), numVertices=4, radius=0.5),
        mpatches.RegularPolygon(xy=(0, 1), numVertices=4, radius=0.5),
    ]
    off_pixels = [
        mpatches.RegularPolygon(xy=(1, 1), numVertices=4, radius=0.5),
        mpatches.RegularPolygon(xy=(1, 0), numVertices=4, radius=0.5),
    ]
    assert len(on_pixels) == 2
    assert isinstance(on_pixels[0], mpatches.RegularPolygon)
    plot_pixels._add_legend(mock_ax, on_pixels, off_pixels)

    mock_ax.legend.assert_called_once()
    args = mock_ax.legend.call_args[0]
    assert len(args[0]) == 3  # 3 legend objects
    assert len(args[1]) == 3  # 3 legend labels


@mock.patch("simtools.model.model_utils.is_two_mirror_telescope")
def test_prepare_pixel_data_two_mirror(mock_is_two_mirror):
    """Test pixel data preparation for different telescope types."""
    base_config = plot_pixels._read_pixel_config(DUMMY_DAT_PATH)

    test_cases = [("MSTS-01", True), ("SSTS-01", False), ("SCTS-01", True), ("LSTN-01", False)]

    for telescope, is_two_mirror in test_cases:
        with mock.patch("simtools.visualization.plot_pixels._read_pixel_config") as mock_read:
            mock_read.return_value = base_config.copy()
            mock_is_two_mirror.return_value = is_two_mirror

            data = plot_pixels._prepare_pixel_data(DUMMY_DAT_PATH, telescope)
            assert "pixel_spacing" in data
            assert data["pixel_spacing"] == pytest.approx(0.64)


@mock.patch("simtools.utils.names.get_array_element_type_from_name")
@mock.patch("simtools.visualization.plot_pixels._is_edge_pixel")
@mock.patch("simtools.visualization.plot_pixels._create_patch")
def test__create_pixel_patches(mock_create_patch, mock_is_edge_pixel, mock_get_array_element_type):
    """Test the logic of _create_pixel_patches."""
    # Mock the return values for the patched functions
    mock_get_array_element_type.return_value = "SCT"
    mock_is_edge_pixel.side_effect = lambda x, y, *args: np.isclose(x, 0.0) and np.isclose(y, 0.0)
    mock_create_patch.side_effect = (
        lambda x, y, diameter, shape: f"Patch({x}, {y}, {diameter}, {shape})"
    )

    # Input data for the test
    x_pos = [0.0, 1.0, -1.0]
    y_pos = [0.0, 1.0, -1.0]
    diameter = 1.0
    module_number = [1, 1, 1]
    module_gap = 0.2
    spacing = 0.6
    shape = 1
    pixels_on = [True, False, True]
    pixel_ids = [1, 2, 3]
    pixels_id_to_print = 2
    telescope_model_name = "SCTS-01"

    # Call the function under test
    on_pixels, edge_pixels, off_pixels = plot_pixels._create_pixel_patches(
        x_pos,
        y_pos,
        diameter,
        module_number,
        module_gap,
        spacing,
        shape,
        pixels_on,
        pixel_ids,
        pixels_id_to_print,
        telescope_model_name,
    )

    # Assertions
    assert len(on_pixels) == 1  # Only one pixel is "on" and not an edge
    assert len(edge_pixels) == 1  # One pixel is an edge
    assert len(off_pixels) == 1  # One pixel is "off"
    mock_get_array_element_type.assert_called_once_with(telescope_model_name)
    mock_is_edge_pixel.assert_any_call(
        0.0, 0.0, x_pos, y_pos, module_number, spacing, module_gap, shape, 1
    )
    mock_create_patch.assert_any_call(0.0, 0.0, diameter, shape)


@mock.patch("matplotlib.pyplot.subplot")
@mock.patch("matplotlib.pyplot.figure", return_value=Figure())
def test_plot_pixel_layout_from_file(mock_figure, mock_subplot):
    """Test plot_pixel_layout_from_file function."""

    result = plot_pixels.plot_pixel_layout_from_file(
        "tests/resources/pixel_layout.dat",
        "SSTS-01",
        pixels_id_to_print=1,
        title="Test",
        xtitle="X",
        ytitle="Y",
    )

    assert result is not None
    assert isinstance(result, Figure)
