#!/usr/bin/python3

from pathlib import Path
from unittest import mock

import astropy.units as u
import matplotlib.patches as mpatches
import numpy as np
import pytest
from astropy.table import Table

from simtools.visualization import plot_mirrors


@mock.patch("simtools.visualization.plot_mirrors.plot_mirror_layout")
@mock.patch("simtools.visualization.plot_mirrors.visualize.save_figure")
@mock.patch("simtools.visualization.plot_mirrors.Mirrors")
@mock.patch("simtools.visualization.plot_mirrors.db_handler.DatabaseHandler")
def test_plot(mock_db_handler, mock_mirrors, mock_save, mock_plot_layout):
    """Test the main plot function."""
    config = {
        "parameter": "mirror_list",
        "site": "North",
        "telescope": "LSTN-01",
        "parameter_version": "1.0.0",
        "model_version": "6.0.0",
    }

    mock_db_instance = mock.MagicMock()
    mock_db_handler.return_value = mock_db_instance
    mock_db_instance.get_model_parameter.return_value = {
        "mirror_list": {"value": "mirror_list.ecsv"}
    }

    mock_fig = mock.MagicMock()
    mock_plot_layout.return_value = mock_fig

    with mock.patch("simtools.visualization.plot_mirrors.io_handler.IOHandler") as mock_io:
        mock_io_instance = mock.MagicMock()
        mock_io.return_value = mock_io_instance
        mock_io_instance.get_output_directory.return_value = Path("/test/path")

        plot_mirrors.plot(config, "test.png")

        mock_db_instance.get_model_parameter.assert_called_once_with(
            "mirror_list",
            "North",
            "LSTN-01",
            parameter_version="1.0.0",
            model_version="6.0.0",
        )
        mock_db_instance.export_model_files.assert_called_once()

        expected_path = Path("/test/path/mirror_list.ecsv")
        mock_mirrors.assert_called_once_with(mirror_list_file=expected_path)
        mock_save.assert_called_once_with(mock_fig, "test.png")


@mock.patch("simtools.visualization.plot_mirrors.plot_mirror_segmentation")
@mock.patch("simtools.visualization.plot_mirrors.visualize.save_figure")
@mock.patch("simtools.visualization.plot_mirrors.db_handler.DatabaseHandler")
def test_plot_segmentation(mock_db_handler, mock_save, mock_plot_segmentation, tmp_path):
    """Test the main plot function for segmentation."""
    config = {
        "parameter": "primary_mirror_segmentation",
        "site": "South",
        "telescope": "SSTS-01",
        "parameter_version": "1.0.0",
        "model_version": "6.0.0",
    }

    mock_db_instance = mock.MagicMock()
    mock_db_handler.return_value = mock_db_instance
    mock_db_instance.get_model_parameter.return_value = {
        "primary_mirror_segmentation": {"value": "primary_segmentation.dat"}
    }

    mock_fig = mock.MagicMock()
    mock_plot_segmentation.return_value = mock_fig

    seg_file = tmp_path / "primary_segmentation.dat"
    seg_file.write_text("100.0 200.0 150.0 2800.0 3\n")

    with mock.patch("simtools.visualization.plot_mirrors.io_handler.IOHandler") as mock_io:
        mock_io_instance = mock.MagicMock()
        mock_io.return_value = mock_io_instance
        mock_io_instance.get_output_directory.return_value = tmp_path

        plot_mirrors.plot(config, "test.png")

        expected_path = tmp_path / "primary_segmentation.dat"
        mock_plot_segmentation.assert_called_once_with(
            data_file_path=expected_path,
            telescope_model_name="SSTS-01",
            parameter_type="primary_mirror_segmentation",
            title=None,
        )
        mock_save.assert_called_once_with(mock_fig, "test.png")


@mock.patch("simtools.visualization.plot_mirrors.plot_mirror_ring_segmentation")
@mock.patch("simtools.visualization.plot_mirrors.visualize.save_figure")
@mock.patch("simtools.visualization.plot_mirrors.db_handler.DatabaseHandler")
def test_plot_ring_segmentation_main(mock_db_handler, mock_save, mock_plot_ring, tmp_path):
    """Test the main plot function with ring segmentation."""
    config = {
        "parameter": "primary_mirror_segmentation",
        "site": "South",
        "telescope": "SCTS-01",
        "parameter_version": "1.0.0",
        "model_version": "6.0.0",
    }

    mock_db_instance = mock.MagicMock()
    mock_db_handler.return_value = mock_db_instance
    mock_db_instance.get_model_parameter.return_value = {
        "primary_mirror_segmentation": {"value": "primary_ring_seg.dat"}
    }

    mock_fig = mock.MagicMock()
    mock_plot_ring.return_value = mock_fig

    ring_file = tmp_path / "primary_ring_seg.dat"
    ring_file.write_text("# Ring segmentation\nring 6 100.0 200.0 0 0.0\n")

    with mock.patch("simtools.visualization.plot_mirrors.io_handler.IOHandler") as mock_io:
        mock_io_instance = mock.MagicMock()
        mock_io.return_value = mock_io_instance
        mock_io_instance.get_output_directory.return_value = tmp_path

        plot_mirrors.plot(config, "test.png")

        expected_path = tmp_path / "primary_ring_seg.dat"
        mock_plot_ring.assert_called_once_with(
            data_file_path=expected_path,
            telescope_model_name="SCTS-01",
            parameter_type="primary_mirror_segmentation",
            title=None,
        )
        mock_save.assert_called_once_with(mock_fig, "test.png")


@mock.patch("simtools.visualization.plot_mirrors.plot_mirror_petal_segmentation")
@mock.patch("simtools.visualization.plot_mirrors.visualize.save_figure")
@mock.patch("simtools.visualization.plot_mirrors.db_handler.DatabaseHandler")
def test_plot_petal_segmentation_main(mock_db_handler, mock_save, mock_plot_petal, tmp_path):
    """Test the main plot function with petal segmentation."""
    config = {
        "parameter": "secondary_mirror_segmentation",
        "site": "North",
        "telescope": "MSTS-01",
        "parameter_version": "1.0.0",
        "model_version": "6.0.0",
    }

    mock_db_instance = mock.MagicMock()
    mock_db_handler.return_value = mock_db_instance
    mock_db_instance.get_model_parameter.return_value = {
        "secondary_mirror_segmentation": {"value": "secondary_petal_seg.dat"}
    }

    mock_fig = mock.MagicMock()
    mock_plot_petal.return_value = mock_fig

    petal_file = tmp_path / "secondary_petal_seg.dat"
    petal_file.write_text("# Petal segmentation\npoly 3 0 100.0 0.0 50.0 86.6 -50.0 86.6\n")

    with mock.patch("simtools.visualization.plot_mirrors.io_handler.IOHandler") as mock_io:
        mock_io_instance = mock.MagicMock()
        mock_io.return_value = mock_io_instance
        mock_io_instance.get_output_directory.return_value = tmp_path

        plot_mirrors.plot(config, "test.png")

        expected_path = tmp_path / "secondary_petal_seg.dat"
        mock_plot_petal.assert_called_once_with(
            data_file_path=expected_path,
            telescope_model_name="MSTS-01",
            parameter_type="secondary_mirror_segmentation",
            title=None,
        )
        mock_save.assert_called_once_with(mock_fig, "test.png")


def test__create_single_mirror_patch():
    """Test patch creation for different mirror shapes."""
    x, y = 100.0, 100.0
    diameter = 150.0

    patch = plot_mirrors._create_single_mirror_patch(x, y, diameter, 0)
    assert isinstance(patch, mpatches.Circle)
    assert patch.center == (x, y)
    assert patch.radius == pytest.approx(diameter / 2)

    patch = plot_mirrors._create_single_mirror_patch(x, y, diameter, 1)
    assert isinstance(patch, mpatches.RegularPolygon)
    assert patch.xy == (x, y)
    assert np.isclose(patch.radius, diameter / np.sqrt(3))
    assert patch.orientation == 0

    patch = plot_mirrors._create_single_mirror_patch(x, y, diameter, 2)
    assert isinstance(patch, mpatches.Rectangle)
    assert patch.get_xy() == (x - diameter / 2, y - diameter / 2)
    assert patch.get_width() == pytest.approx(diameter)
    assert patch.get_height() == pytest.approx(diameter)

    patch = plot_mirrors._create_single_mirror_patch(x, y, diameter, 3)
    assert isinstance(patch, mpatches.RegularPolygon)
    assert patch.xy == (x, y)
    assert np.isclose(patch.radius, diameter / np.sqrt(3))
    assert np.isclose(patch.orientation, np.pi / 2)


def test__create_mirror_patches():
    """Test creation of mirror patches."""
    x_pos = np.array([100.0, 200.0, 300.0])
    y_pos = np.array([100.0, 200.0, 300.0])
    diameter = 150.0
    shape_type = 3
    focal_lengths = np.array([2800.0, 2850.0, 2900.0])

    patches, colors = plot_mirrors._create_mirror_patches(
        x_pos, y_pos, diameter, shape_type, focal_lengths
    )

    assert len(patches) == 3
    assert len(colors) == 3
    assert all(isinstance(p, mpatches.RegularPolygon) for p in patches)
    assert colors == list(focal_lengths)


@mock.patch("matplotlib.pyplot.subplots")
def test_plot_mirror_layout(mock_subplots):
    """Test mirror layout plotting."""
    mock_fig = mock.MagicMock()
    mock_ax = mock.MagicMock()
    mock_ax.get_xlim.return_value = (-1000, 1000)
    mock_ax.get_ylim.return_value = (-1000, 1000)
    mock_subplots.return_value = (mock_fig, mock_ax)

    mock_mirrors = mock.MagicMock()

    mirror_table = Table(
        {
            "mirror_x": [100.0, 200.0, 300.0] * u.cm,
            "mirror_y": [100.0, 200.0, 300.0] * u.cm,
            "focal_length": [2800.0, 2850.0, 2900.0] * u.cm,
            "mirror_panel_id": [1, 2, 3],
        }
    )

    mock_mirrors.mirror_table = mirror_table
    mock_mirrors.mirror_diameter = 151.0 * u.cm
    mock_mirrors.shape_type = 3
    mock_mirrors.number_of_mirrors = 3

    with mock.patch("matplotlib.pyplot.colorbar") as mock_colorbar:
        mock_cbar = mock.MagicMock()
        mock_colorbar.return_value = mock_cbar

        fig = plot_mirrors.plot_mirror_layout(
            mirrors=mock_mirrors,
            telescope_model_name="LSTN-01",
            title="Test Mirror Layout",
        )

        assert fig is not None
        mock_ax.set_aspect.assert_called_once_with("equal")
        mock_ax.add_collection.assert_called_once()
        mock_colorbar.assert_called_once()
        mock_cbar.set_label.assert_called_once()


def test__add_mirror_labels():
    """Test adding mirror labels to plot."""
    mock_ax = mock.MagicMock()
    x_pos = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    y_pos = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    mirror_ids = [1, 2, 3, 4, 5]

    plot_mirrors._add_mirror_labels(mock_ax, x_pos, y_pos, mirror_ids, max_labels=3)

    assert mock_ax.text.call_count == 3


def test__configure_mirror_plot():
    """Test mirror plot configuration."""
    mock_ax = mock.MagicMock()
    mock_ax.get_xlim.return_value = (-1000, 1000)
    mock_ax.get_ylim.return_value = (-1000, 1000)

    x_pos = np.array([100.0, 200.0, 300.0])
    y_pos = np.array([100.0, 200.0, 300.0])

    with (
        mock.patch("matplotlib.pyplot.grid"),
        mock.patch("matplotlib.pyplot.xlabel"),
        mock.patch("matplotlib.pyplot.ylabel"),
        mock.patch("matplotlib.pyplot.tick_params"),
    ):
        plot_mirrors._configure_mirror_plot(mock_ax, x_pos, y_pos, "Test Title", "LSTN-01")

        mock_ax.set_aspect.assert_called_once_with("equal")
        mock_ax.set_xlim.assert_called_once()
        mock_ax.set_ylim.assert_called_once()
        mock_ax.set_title.assert_called_once()


def test__add_camera_frame_indicator():
    """Test camera frame coordinate system indicator addition."""
    mock_ax = mock.MagicMock()
    mock_ax.get_xlim.return_value = (-1000, 1000)
    mock_ax.get_ylim.return_value = (-1000, 1000)

    plot_mirrors._add_camera_frame_indicator(mock_ax)

    assert mock_ax.annotate.call_count == 2
    assert mock_ax.text.call_count == 2


def test__add_mirror_statistics():
    """Test mirror statistics addition."""
    mock_ax = mock.MagicMock()
    mock_mirrors = mock.MagicMock()
    mock_mirrors.number_of_mirrors = 198
    mock_mirrors.shape_type = 3

    x_pos = np.array([1000.0, -1000.0, 0.0])
    y_pos = np.array([1000.0, -1000.0, 0.0])
    diameter = 151.0

    plot_mirrors._add_mirror_statistics(mock_ax, mock_mirrors, x_pos, y_pos, diameter)

    mock_ax.text.assert_called_once()
    call_args = mock_ax.text.call_args
    assert "Number of mirrors: 198" in call_args[0][2]


def test__add_mirror_statistics_hexagonal():
    """Test mirror statistics with hexagonal shape."""
    mock_ax = mock.MagicMock()
    mock_mirrors = mock.MagicMock()
    mock_mirrors.number_of_mirrors = 100
    mock_mirrors.shape_type = 1  # Hexagonal shape

    x_pos = np.array([500.0, -500.0, 0.0])
    y_pos = np.array([500.0, -500.0, 0.0])
    diameter = 120.0

    plot_mirrors._add_mirror_statistics(mock_ax, mock_mirrors, x_pos, y_pos, diameter)

    mock_ax.text.assert_called_once()
    call_args = mock_ax.text.call_args
    assert "Number of mirrors: 100" in call_args[0][2]
    assert "Mirror diameter: 120.0 cm" in call_args[0][2]


def test__add_mirror_statistics_square():
    """Test mirror statistics with square shape."""
    mock_ax = mock.MagicMock()
    mock_mirrors = mock.MagicMock()
    mock_mirrors.number_of_mirrors = 50
    mock_mirrors.shape_type = 2  # Square shape

    x_pos = np.array([300.0, -300.0, 0.0])
    y_pos = np.array([300.0, -300.0, 0.0])
    diameter = 100.0

    plot_mirrors._add_mirror_statistics(mock_ax, mock_mirrors, x_pos, y_pos, diameter)

    mock_ax.text.assert_called_once()
    call_args = mock_ax.text.call_args
    assert "Number of mirrors: 50" in call_args[0][2]
    assert "Mirror diameter: 100.0 cm" in call_args[0][2]


def test__read_segmentation_file(tmp_path):
    """Test reading segmentation file."""
    seg_file = tmp_path / "test_segmentation.dat"
    seg_file.write_text(
        "# Test segmentation file\n"
        "100.0  200.0  150.0  2800.0  3  0.0  #%  id=1\n"
        "200.0  300.0  150.0  2850.0  3  0.0  #%  id=1\n"
        "300.0  400.0  150.0  2900.0  3  0.0  #%  id=2\n"
    )

    data = plot_mirrors._read_segmentation_file(seg_file)

    assert len(data["x"]) == 3
    assert len(data["y"]) == 3
    assert data["diameter"] == pytest.approx(150.0)
    assert data["shape_type"] == 3
    assert len(data["segment_ids"]) == 3
    assert data["segment_ids"][0] == 1
    assert data["segment_ids"][2] == 2


def test__read_segmentation_file_with_short_lines(tmp_path):
    """Test reading segmentation file with lines that have fewer than 5 parts."""
    seg_file = tmp_path / "test_segmentation.dat"
    seg_file.write_text(
        "# Test segmentation file\n"
        "100.0  200.0\n"  # Only 2 parts - should be skipped
        "200.0  300.0  150.0  2850.0  3  0.0  #%  id=1\n"
        "# Another comment\n"
        "\n"  # Empty line
        "300.0\n"  # Only 1 part - should be skipped
        "400.0  500.0  150.0  2900.0  3  0.0  #%  id=2\n"
    )

    data = plot_mirrors._read_segmentation_file(seg_file)

    assert len(data["x"]) == 2
    assert len(data["y"]) == 2
    assert data["diameter"] == pytest.approx(150.0)
    assert data["shape_type"] == 3
    assert len(data["segment_ids"]) == 2


def test__create_segmentation_patches():
    """Test creation of segmentation patches."""
    x_pos = np.array([100.0, 200.0, 300.0])
    y_pos = np.array([100.0, 200.0, 300.0])
    diameter = 150.0
    shape_type = 3
    segment_ids = [1, 1, 2]

    patches, colors = plot_mirrors._create_segmentation_patches(
        x_pos, y_pos, diameter, shape_type, segment_ids
    )

    assert len(patches) == 3
    assert len(colors) == 3
    assert all(isinstance(p, mpatches.RegularPolygon) for p in patches)
    assert colors == segment_ids


@mock.patch("matplotlib.pyplot.subplots")
def test_plot_mirror_segmentation(mock_subplots, tmp_path):
    """Test mirror segmentation plotting."""
    mock_fig = mock.MagicMock()
    mock_ax = mock.MagicMock()
    mock_ax.get_xlim.return_value = (-1000, 1000)
    mock_ax.get_ylim.return_value = (-1000, 1000)
    mock_subplots.return_value = (mock_fig, mock_ax)

    seg_file = tmp_path / "test_segmentation.dat"
    seg_file.write_text(
        "# Test segmentation file\n"
        "100.0  200.0  150.0  2800.0  3  0.0  #%  id=1\n"
        "200.0  300.0  150.0  2850.0  3  0.0  #%  id=1\n"
        "300.0  400.0  150.0  2900.0  3  0.0  #%  id=2\n"
    )

    with mock.patch("matplotlib.pyplot.colorbar") as mock_colorbar:
        mock_cbar = mock.MagicMock()
        mock_colorbar.return_value = mock_cbar

        fig = plot_mirrors.plot_mirror_segmentation(
            data_file_path=seg_file,
            telescope_model_name="SSTS-01",
            parameter_type="primary_mirror_segmentation",
            title="Test Segmentation",
        )

        assert fig is not None
        mock_ax.set_aspect.assert_called_once_with("equal")
        mock_ax.add_collection.assert_called_once()
        mock_colorbar.assert_called_once()


def test__extract_diameter():
    """Test diameter extraction."""
    parts = ["100.0", "200.0", "150.0", "2800.0", "3"]
    assert plot_mirrors._extract_diameter(parts, None) == pytest.approx(150.0)
    assert plot_mirrors._extract_diameter(parts, 120.0) == pytest.approx(120.0)


def test__extract_shape_type():
    """Test shape type extraction."""
    parts = ["100.0", "200.0", "150.0", "2800.0", "3"]
    assert plot_mirrors._extract_shape_type(parts, None) == 3
    assert plot_mirrors._extract_shape_type(parts, 1) == 1


def test__extract_segment_id():
    """Test segment ID extraction."""
    parts_with_id = ["100.0", "200.0", "150.0", "2800.0", "3", "0.0", "#%", "id=5"]
    assert plot_mirrors._extract_segment_id(parts_with_id, 0) == 5

    parts_without_id = ["100.0", "200.0", "150.0", "2800.0", "3"]
    assert plot_mirrors._extract_segment_id(parts_without_id, 10) == 10


def test__detect_segmentation_type(tmp_path):
    """Test detection of segmentation file types."""
    ring_file = tmp_path / "ring_seg.dat"
    ring_file.write_text("# Ring segmentation\nring 6 100.0 200.0 0 0.0\n")
    assert plot_mirrors._detect_segmentation_type(ring_file) == "ring"

    petal_file = tmp_path / "petal_seg.dat"
    petal_file.write_text("# Petal segmentation\npoly 3 0 10.0 20.0 30.0 40.0\n")
    assert plot_mirrors._detect_segmentation_type(petal_file) == "petal"

    standard_file = tmp_path / "standard_seg.dat"
    standard_file.write_text("100.0 200.0 150.0 2800.0 3\n")
    assert plot_mirrors._detect_segmentation_type(standard_file) == "standard"


def test__calculate_mean_outer_edge_radius():
    """Test calculation of mean outer edge radius."""
    x_pos = np.array([100.0, -100.0, 0.0])
    y_pos = np.array([0.0, 0.0, 100.0])
    diameter = 50.0

    radius_circular = plot_mirrors._calculate_mean_outer_edge_radius(x_pos, y_pos, diameter, 0)
    assert radius_circular == pytest.approx(125.0)

    radius_hexagonal = plot_mirrors._calculate_mean_outer_edge_radius(x_pos, y_pos, diameter, 1)
    expected_hex = np.mean(
        [
            100.0 + diameter / np.sqrt(3),
            100.0 + diameter / np.sqrt(3),
            100.0 + diameter / np.sqrt(3),
        ]
    )
    assert radius_hexagonal == pytest.approx(expected_hex)

    radius_square = plot_mirrors._calculate_mean_outer_edge_radius(x_pos, y_pos, diameter, 2)
    expected_square = np.mean(
        [
            100.0 + diameter / np.sqrt(2),
            100.0 + diameter / np.sqrt(2),
            100.0 + diameter / np.sqrt(2),
        ]
    )
    assert radius_square == pytest.approx(expected_square)


def test__read_ring_segmentation_data(tmp_path):
    """Test reading ring segmentation data."""
    ring_file = tmp_path / "ring_seg.dat"
    ring_file.write_text("# Ring segmentation file\nring 6 100.0 200.0 0 15.0\n")

    radii, phi0, nseg = plot_mirrors._read_ring_segmentation_data(ring_file)

    assert len(radii) == 2
    assert radii[0] == pytest.approx(100.0)
    assert radii[1] == pytest.approx(200.0)
    assert phi0 == pytest.approx(15.0)
    assert nseg == 6


@mock.patch("matplotlib.pyplot.subplots")
def test_plot_mirror_ring_segmentation(mock_subplots, tmp_path):
    """Test ring segmentation plotting."""
    mock_fig = mock.MagicMock()
    mock_ax = mock.MagicMock()
    mock_subplots.return_value = (mock_fig, mock_ax)

    ring_file = tmp_path / "ring_seg.dat"
    ring_file.write_text("# Ring segmentation file\nring 6 100.0 200.0 0 0.0\n")

    with (
        mock.patch("matplotlib.pyplot.text"),
        mock.patch("matplotlib.pyplot.subplots_adjust"),
        mock.patch("matplotlib.pyplot.tight_layout"),
    ):
        fig = plot_mirrors.plot_mirror_ring_segmentation(
            data_file_path=ring_file,
            telescope_model_name="SCTS-01",
            parameter_type="primary_mirror_segmentation",
            title="Test Ring",
        )

        assert fig is not None
        mock_ax.set_ylim.assert_called_once()
        mock_ax.set_title.assert_called_once()


@mock.patch("matplotlib.pyplot.subplots")
def test_plot_mirror_ring_segmentation_small_inner_radius(mock_subplots, tmp_path):
    """Test ring segmentation plotting with small inner radius."""
    mock_fig = mock.MagicMock()
    mock_ax = mock.MagicMock()
    mock_subplots.return_value = (mock_fig, mock_ax)

    ring_file = tmp_path / "ring_seg_small.dat"
    ring_file.write_text(
        "# Ring segmentation file with small inner radius\nring 6 10.0 200.0 0 0.0\n"
    )

    with (
        mock.patch("matplotlib.pyplot.text") as mock_text,
        mock.patch("matplotlib.pyplot.subplots_adjust"),
        mock.patch("matplotlib.pyplot.tight_layout"),
    ):
        fig = plot_mirrors.plot_mirror_ring_segmentation(
            data_file_path=ring_file,
            telescope_model_name="SCTS-01",
            parameter_type="secondary_mirror_segmentation",
            title="Test Ring Small",
        )

        assert fig is not None
        mock_ax.set_ylim.assert_called_once()
        mock_ax.set_title.assert_called_once()
        assert mock_text.call_count >= 2


@mock.patch("matplotlib.pyplot.subplots")
def test_plot_mirror_petal_segmentation(mock_subplots, tmp_path):
    """Test petal segmentation plotting."""
    mock_fig = mock.MagicMock()
    mock_ax = mock.MagicMock()
    mock_subplots.return_value = (mock_fig, mock_ax)

    petal_file = tmp_path / "petal_seg.dat"
    petal_file.write_text(
        "# Petal segmentation file\n"
        "poly 3 0 100.0 0.0 50.0 86.6 -50.0 86.6\n"
        "poly 3 120 100.0 0.0 50.0 86.6 -50.0 86.6\n"
    )

    with (
        mock.patch("matplotlib.pyplot.grid"),
        mock.patch("matplotlib.pyplot.xlabel"),
        mock.patch("matplotlib.pyplot.ylabel"),
        mock.patch("matplotlib.pyplot.tick_params"),
        mock.patch("matplotlib.pyplot.tight_layout"),
    ):
        fig = plot_mirrors.plot_mirror_petal_segmentation(
            data_file_path=petal_file,
            telescope_model_name="SCTS-01",
            parameter_type="primary_mirror_segmentation",
            title="Test Petal",
        )

        assert fig is not None
        mock_ax.set_aspect.assert_called_once_with("equal")
        mock_ax.set_title.assert_called_once()
        mock_ax.text.assert_called()
