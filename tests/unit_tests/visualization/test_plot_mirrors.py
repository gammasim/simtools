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


@pytest.mark.parametrize(
    ("shape_type", "n_mirrors", "diameter"),
    [
        (1, 100, 120.0),  # Hexagonal
        (2, 50, 100.0),  # Square
        (3, 198, 151.0),  # Y-hexagonal
    ],
)
def test__add_mirror_statistics_various_shapes(shape_type, n_mirrors, diameter):
    """Test mirror statistics with different shape types."""
    mock_ax = mock.MagicMock()
    mock_mirrors = mock.MagicMock()
    mock_mirrors.number_of_mirrors = n_mirrors
    mock_mirrors.shape_type = shape_type

    x_pos = np.array([300.0, -300.0, 0.0])
    y_pos = np.array([300.0, -300.0, 0.0])

    plot_mirrors._add_mirror_statistics(mock_ax, mock_mirrors, x_pos, y_pos, diameter)

    mock_ax.text.assert_called_once()
    call_args = mock_ax.text.call_args
    assert f"Number of mirrors: {n_mirrors}" in call_args[0][2]
    assert f"Mirror diameter: {diameter:.1f} cm" in call_args[0][2]


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

    shape_file = tmp_path / "shape_seg.dat"
    shape_file.write_text("# Shape segmentation\nhex 1 10.0 20.0 30.0 40.0\n")
    assert plot_mirrors._detect_segmentation_type(shape_file) == "shape"

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


def test__read_ring_segmentation_data(tmp_path):
    """Test reading ring segmentation data."""
    ring_file = tmp_path / "ring_seg.dat"
    ring_file.write_text(
        "# Ring segmentation file\nring 6 100.0 200.0 60.0 15.0\nring 12 200.0 300.0 30.0 0.0\n"
    )

    rings = plot_mirrors._read_ring_segmentation_data(ring_file)

    assert len(rings) == 2
    assert rings[0]["nseg"] == 6
    assert rings[0]["rmin"] == pytest.approx(100.0)
    assert rings[0]["rmax"] == pytest.approx(200.0)
    assert rings[0]["dphi"] == pytest.approx(60.0)
    assert rings[0]["phi0"] == pytest.approx(15.0)
    assert rings[1]["nseg"] == 12
    assert rings[1]["rmin"] == pytest.approx(200.0)
    assert rings[1]["rmax"] == pytest.approx(300.0)


@mock.patch("matplotlib.pyplot.subplots")
def test_plot_mirror_ring_segmentation(mock_subplots, tmp_path):
    """Test ring segmentation plotting."""
    mock_fig = mock.MagicMock()
    mock_ax = mock.MagicMock()
    mock_subplots.return_value = (mock_fig, mock_ax)

    ring_file = tmp_path / "ring_seg.dat"
    ring_file.write_text("# Ring segmentation file\nring 6 100.0 200.0 60.0 0.0\n")

    with (
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
def test_plot_mirror_shape_segmentation(mock_subplots, tmp_path):
    """Test shape segmentation plotting."""
    mock_fig = mock.MagicMock()
    mock_ax = mock.MagicMock()
    mock_subplots.return_value = (mock_fig, mock_ax)

    shape_file = tmp_path / "shape_seg.dat"
    shape_file.write_text(
        "# Shape segmentation file\nhex 1 100.0 0.0 50.0 0.0\nhex 2 -50.0 86.6 50.0 0.0\n"
    )

    with (
        mock.patch("matplotlib.pyplot.grid"),
        mock.patch("matplotlib.pyplot.xlabel"),
        mock.patch("matplotlib.pyplot.ylabel"),
        mock.patch("matplotlib.pyplot.tick_params"),
    ):
        fig = plot_mirrors.plot_mirror_shape_segmentation(
            data_file_path=shape_file,
            telescope_model_name="SCTS-01",
            parameter_type="primary_mirror_segmentation",
            title="Test Shape",
        )

        assert fig is not None
        mock_ax.set_aspect.assert_called_once_with("equal")
        mock_ax.set_title.assert_called_once()
        mock_ax.text.assert_called()


def test__read_segmentation_file_with_invalid_data(tmp_path):
    """Test reading segmentation file with invalid numeric data (ValueError handling)."""
    data_file = tmp_path / "invalid_seg.dat"
    data_file.write_text("invalid_x invalid_y 100.0 50.0\n")

    result = plot_mirrors._read_segmentation_file(data_file)

    assert len(result["x"]) == 0
    assert len(result["y"]) == 0


def test__read_segmentation_file_empty_warning(tmp_path, caplog):
    """Test warning when segmentation file has no valid numeric data."""
    data_file = tmp_path / "empty_seg.dat"
    data_file.write_text("# All comments\n% More comments\n")

    plot_mirrors._read_segmentation_file(data_file)

    assert "No valid numeric data found" in caplog.text


def test_plot_mirror_ring_segmentation_empty_data(tmp_path, caplog):
    """Test ring segmentation with empty data file."""
    data_file = tmp_path / "empty_ring.dat"
    data_file.write_text("# No data\n")

    result = plot_mirrors.plot_mirror_ring_segmentation(
        data_file_path=data_file,
        telescope_model_name="SCTS-01",
        parameter_type="primary_mirror_segmentation",
    )

    assert result is None
    assert "No ring data found" in caplog.text


def test__parse_segment_id_line_invalid():
    """Test _parse_segment_id_line with invalid input."""
    assert plot_mirrors._parse_segment_id_line("invalid line") == 0
    assert plot_mirrors._parse_segment_id_line("") == 0


def test__is_skippable_line():
    """Test _is_skippable_line function."""
    assert plot_mirrors._is_skippable_line("")
    assert plot_mirrors._is_skippable_line("# comment")
    assert plot_mirrors._is_skippable_line("% comment")
    assert not plot_mirrors._is_skippable_line("valid data")


def test__read_shape_segmentation_file_with_segment_id(tmp_path):
    """Test reading shape segmentation file with segment ID marker."""
    data_file = tmp_path / "shape_with_id.dat"
    data_file.write_text("# segment id 5\nhex 1 100.0 0.0 50.0 0.0\nhex 2 200.0 0.0 50.0 0.0\n")

    shape_segments, segment_ids = plot_mirrors._read_shape_segmentation_file(data_file)

    assert len(shape_segments) == 2
    assert all(sid == 5 for sid in segment_ids)


def test__create_shape_patches_circle():
    """Test creating circular shape patches."""
    mock_ax = mock.MagicMock()
    shape_segments = [
        {"shape": "circular", "x": 0.0, "y": 0.0, "diameter": 100.0, "rotation": 0.0},
    ]
    segment_ids = [1]

    patches, _ = plot_mirrors._create_shape_patches(mock_ax, shape_segments, segment_ids)

    assert len(patches) == 1
    assert isinstance(patches[0], mpatches.Circle)


def test_plot_mirror_shape_segmentation_stats_variations(tmp_path):
    """Test shape segmentation plot with different stats text variations."""
    data_file = tmp_path / "shape_stats.dat"

    data_file.write_text("hex 1 100.0 0.0 50.0 0.0\n")
    fig1 = plot_mirrors.plot_mirror_shape_segmentation(
        data_file_path=data_file,
        telescope_model_name="SCTS-01",
        parameter_type="primary_mirror_segmentation",
    )
    assert fig1 is not None

    data_file.write_text("# No data\n")
    fig2 = plot_mirrors.plot_mirror_shape_segmentation(
        data_file_path=data_file,
        telescope_model_name="SCTS-01",
        parameter_type="primary_mirror_segmentation",
    )
    assert fig2 is not None


def test__add_camera_frame_indicator_mst():
    """Test camera frame indicator for MST telescope."""
    mock_ax = mock.MagicMock()
    mock_ax.get_xlim.return_value = (-1000, 1000)
    mock_ax.get_ylim.return_value = (-1000, 1000)

    plot_mirrors._add_camera_frame_indicator(mock_ax, "MSTN-01")

    assert mock_ax.annotate.call_count == 2
    assert mock_ax.text.call_count == 2


def test__configure_mirror_plot_empty_data():
    """Test mirror plot configuration with empty data."""
    mock_ax = mock.MagicMock()

    x_pos = np.array([])
    y_pos = np.array([])

    with (
        mock.patch("matplotlib.pyplot.grid"),
        mock.patch("matplotlib.pyplot.xlabel"),
        mock.patch("matplotlib.pyplot.ylabel"),
        mock.patch("matplotlib.pyplot.tick_params"),
    ):
        plot_mirrors._configure_mirror_plot(mock_ax, x_pos, y_pos, "Test", "LSTN-01")

        mock_ax.set_xlim.assert_called_once_with(-1000, 1000)
        mock_ax.set_ylim.assert_called_once_with(-1000, 1000)
        mock_ax.text.assert_called_once()


def test__get_radius_offset_square():
    """Test radius offset calculation for square shape."""
    diameter = 100.0
    offset = plot_mirrors._get_radius_offset(diameter, 2)
    assert offset == pytest.approx(diameter / np.sqrt(2))


@pytest.mark.parametrize(
    ("ring_count", "ring_data"),
    [
        (1, "ring 8 100.0 200.0 45.0 0.0\n"),
        (2, "ring 16 217.174 338.505 22.5 -11.25\nring 32 339.908 481.178 11.25 0\n"),
    ],
)
def test_plot_mirror_ring_segmentation_various_ring_counts(tmp_path, ring_count, ring_data):
    """Test ring segmentation with 1 ring (general case) and 2 rings (SCT inner/outer case)."""
    ring_file = tmp_path / f"ring_seg_{ring_count}.dat"
    ring_file.write_text(ring_data)

    with mock.patch("matplotlib.pyplot.tight_layout"):
        fig = plot_mirrors.plot_mirror_ring_segmentation(
            data_file_path=ring_file,
            telescope_model_name="SCTS-01",
            parameter_type="primary_mirror_segmentation",
        )

        assert fig is not None


def test_plot_mirror_shape_segmentation_no_segment_ids(tmp_path):
    """Test shape segmentation with segments but no segment IDs."""
    data_file = tmp_path / "shape_no_ids.dat"
    data_file.write_text(
        "# Shape file without segment IDs\nhex 1 100.0 0.0 50.0 0.0\nhex 2 200.0 0.0 50.0 0.0\n"
    )

    with mock.patch(
        "simtools.visualization.plot_mirrors._read_shape_segmentation_file"
    ) as mock_read:
        mock_read.return_value = (
            [
                {"shape": "hex", "x": 100.0, "y": 0.0, "diameter": 50.0, "rotation": 0.0},
                {"shape": "hex", "x": 200.0, "y": 0.0, "diameter": 50.0, "rotation": 0.0},
            ],
            [],  # Empty segment_ids
        )

        fig = plot_mirrors.plot_mirror_shape_segmentation(
            data_file_path=data_file,
            telescope_model_name="SCTS-01",
            parameter_type="primary_mirror_segmentation",
        )

        assert fig is not None


@mock.patch("simtools.visualization.plot_mirrors.plot_mirror_shape_segmentation")
@mock.patch("simtools.visualization.plot_mirrors.visualize.save_figure")
@mock.patch("simtools.visualization.plot_mirrors.db_handler.DatabaseHandler")
def test_plot_shape_segmentation_main(mock_db_handler, mock_save, mock_plot_shape, tmp_path):
    """Test the main plot function with shape segmentation."""
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
        "primary_mirror_segmentation": {"value": "primary_shape_seg.dat"}
    }

    mock_fig = mock.MagicMock()
    mock_plot_shape.return_value = mock_fig

    shape_file = tmp_path / "primary_shape_seg.dat"
    shape_file.write_text("# Shape segmentation\nhex 1 100.0 0.0 50.0 0.0\n")

    with mock.patch("simtools.visualization.plot_mirrors.io_handler.IOHandler") as mock_io:
        mock_io_instance = mock.MagicMock()
        mock_io.return_value = mock_io_instance
        mock_io_instance.get_output_directory.return_value = tmp_path

        plot_mirrors.plot(config, "test.png")

        expected_path = tmp_path / "primary_shape_seg.dat"
        mock_plot_shape.assert_called_once_with(
            data_file_path=expected_path,
            telescope_model_name="SCTS-01",
            parameter_type="primary_mirror_segmentation",
            title=None,
        )
        mock_save.assert_called_once_with(mock_fig, "test.png")


def test__read_segmentation_file_with_invalid_x_y(tmp_path):
    """Test reading segmentation file where x,y are invalid but other parts are valid."""
    seg_file = tmp_path / "test_segmentation_invalid_xy.dat"
    seg_file.write_text(
        "# Test segmentation file\n"
        "invalid_x  invalid_y  150.0  2800.0  3  0.0  #%  id=1\n"
        "200.0  300.0  150.0  2850.0  3  0.0  #%  id=1\n"
    )

    data = plot_mirrors._read_segmentation_file(seg_file)

    assert len(data["x"]) == 1
    assert len(data["y"]) == 1
    assert data["x"][0] == pytest.approx(200.0)
    assert data["y"][0] == pytest.approx(300.0)
