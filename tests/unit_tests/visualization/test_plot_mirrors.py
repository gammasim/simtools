#!/usr/bin/python3

import contextlib
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
@mock.patch("simtools.visualization.plot_mirrors.TelescopeModel")
def test_plot(mock_telescope_model, mock_mirrors, mock_save, mock_plot_layout, tmp_path):
    """Test the main plot function."""
    config = {
        "parameter": "mirror_list",
        "site": "North",
        "telescope": "LSTN-01",
        "parameter_version": "1.0.0",
        "model_version": "6.0.0",
    }

    mock_tel_instance = mock.MagicMock()
    mock_telescope_model.return_value = mock_tel_instance
    mock_tel_instance.get_parameter_value.return_value = "mirror_list.ecsv"

    mock_fig = mock.MagicMock()
    mock_plot_layout.return_value = mock_fig

    with mock.patch("simtools.visualization.plot_mirrors.io_handler.IOHandler") as mock_io:
        mock_io_instance = mock.MagicMock()
        mock_io.return_value = mock_io_instance
        mock_io_instance.get_output_directory.return_value = tmp_path

        plot_mirrors.plot(config, "test.png")

        mock_telescope_model.assert_called_once_with(
            site="North",
            telescope_name="LSTN-01",
            model_version="6.0.0",
            db_config=None,
            ignore_software_version=True,
        )
        mock_tel_instance.get_parameter_value.assert_called_once_with("mirror_list")
        mock_tel_instance.export_model_files.assert_called_once_with(destination_path=tmp_path)

        expected_path = tmp_path / "mirror_list.ecsv"
        mock_mirrors.assert_called_once_with(mirror_list_file=expected_path)
        mock_save.assert_called_once_with(mock_fig, "test.png")


@pytest.mark.parametrize(
    ("telescope", "file_name", "file_content", "plot_function"),
    [
        (
            "SSTS-01",
            "primary_segmentation.dat",
            "100.0 200.0 150.0 2800.0 3\n",
            "plot_mirror_segmentation",
        ),
        (
            "SCTS-01",
            "primary_ring_seg.dat",
            "# Ring segmentation\nring 6 100.0 200.0 0 0.0\n",
            "plot_mirror_ring_segmentation",
        ),
        (
            "SCTS-01",
            "primary_shape_seg.dat",
            "# Shape segmentation\nhex 1 100.0 0.0 50.0 0.0\n",
            "plot_mirror_shape_segmentation",
        ),
    ],
)
@mock.patch("simtools.visualization.plot_mirrors.visualize.save_figure")
@mock.patch("simtools.visualization.plot_mirrors.TelescopeModel")
def test_plot_segmentation_types(
    mock_telescope_model, mock_save, tmp_path, telescope, file_name, file_content, plot_function
):
    """Test the main plot function for different segmentation types."""
    config = {
        "parameter": "primary_mirror_segmentation",
        "site": "South",
        "telescope": telescope,
        "parameter_version": "1.0.0",
        "model_version": "6.0.0",
    }

    mock_tel_instance = mock.MagicMock()
    mock_telescope_model.return_value = mock_tel_instance
    mock_tel_instance.get_parameter_value.return_value = file_name

    seg_file = tmp_path / file_name
    seg_file.write_text(file_content)

    mock_fig = mock.MagicMock()
    mock_io_instance = mock.MagicMock()
    mock_io_instance.get_output_directory.return_value = tmp_path

    with (
        mock.patch(
            "simtools.visualization.plot_mirrors.io_handler.IOHandler",
            return_value=mock_io_instance,
        ),
        mock.patch(
            f"simtools.visualization.plot_mirrors.{plot_function}", return_value=mock_fig
        ) as mock_plot_func,
    ):
        plot_mirrors.plot(config, "test.png")

        expected_path = tmp_path / file_name
        mock_plot_func.assert_called_once_with(
            data_file_path=expected_path,
            telescope_model_name=telescope,
            parameter_type="primary_mirror_segmentation",
        )
        mock_save.assert_called_once_with(mock_fig, "test.png")


def test__create_single_mirror_patch():
    """Test patch creation for different mirror shapes."""
    x, y = 100.0, 100.0
    diameter = 150.0

    patch = plot_mirrors._create_single_mirror_patch(x, y, diameter, 1)
    assert isinstance(patch, mpatches.RegularPolygon)
    assert patch.xy == (x, y)
    assert np.isclose(patch.radius, diameter / np.sqrt(3))
    # Rotation compensation: base (0) - pi/2 = -pi/2
    assert np.isclose(patch.orientation, -np.pi / 2)

    patch = plot_mirrors._create_single_mirror_patch(x, y, diameter, 3)
    assert isinstance(patch, mpatches.RegularPolygon)
    assert patch.xy == (x, y)
    assert np.isclose(patch.radius, diameter / np.sqrt(3))
    # Rotation compensation: base (pi/2) - pi/2 = 0
    assert np.isclose(patch.orientation, 0)


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
def test_plot_mirror_layout(mock_subplots, tmp_path):
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
            mirror_file_path=tmp_path / "test_mirror_list.dat",
            telescope_model_name="LSTN-01",
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
        mock.patch("matplotlib.pyplot.grid") as mock_grid,
        mock.patch("matplotlib.pyplot.xlabel") as mock_xlabel,
        mock.patch("matplotlib.pyplot.ylabel") as mock_ylabel,
        mock.patch("matplotlib.pyplot.tick_params") as mock_tick_params,
    ):
        plot_mirrors._configure_mirror_plot(mock_ax, x_pos, y_pos)

        mock_ax.set_aspect.assert_called_once_with("equal")
        mock_ax.set_xlim.assert_called_once()
        mock_ax.set_ylim.assert_called_once()
        mock_grid.assert_called_once()
        mock_xlabel.assert_called_once()
        mock_ylabel.assert_called_once()
        mock_tick_params.assert_called_once()


@pytest.mark.parametrize(
    ("shape_type", "n_mirrors", "diameter"),
    [
        (1, 100, 120.0),  # Hexagonal
        (3, 198, 151.0),  # Y-hexagonal
    ],
)
def test__add_mirror_statistics_various_shapes(shape_type, n_mirrors, diameter, tmp_path):
    """Test mirror statistics with different shape types."""
    mock_ax = mock.MagicMock()
    mock_mirrors = mock.MagicMock()
    mock_mirrors.number_of_mirrors = n_mirrors
    mock_mirrors.shape_type = shape_type

    mirror_file = tmp_path / "test_mirror.dat"
    mirror_file.write_text("# Test file\n100.0 200.0 150.0 2800.0 3\n")

    x_pos = np.array([300.0, -300.0, 0.0])
    y_pos = np.array([300.0, -300.0, 0.0])

    plot_mirrors._add_mirror_statistics(mock_ax, mock_mirrors, mirror_file, x_pos, y_pos, diameter)

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
        )

        assert fig is not None
        mock_ax.set_ylim.assert_called_once()


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
        )

        assert fig is not None
        mock_ax.set_aspect.assert_called_once_with("equal")
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


def test__configure_mirror_plot_empty_data():
    """Test mirror plot configuration with empty data."""
    mock_ax = mock.MagicMock()
    plot_mirrors._configure_mirror_plot(mock_ax, np.array([]), np.array([]))
    mock_ax.text.assert_called_once()
    assert "No valid mirror data" in mock_ax.text.call_args[0][2]


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


def test__extract_float_after_keyword():
    """Test extracting float values after keywords."""
    # Parse after colon
    line = "# Total surface area: 107.2 m**2"
    assert plot_mirrors._extract_float_after_keyword(line, ":") == pytest.approx(107.2)

    # Parse after equals
    line = "# Rmax = 6.90 m (Dmax = 13.80 m)"
    assert plot_mirrors._extract_float_after_keyword(line, "=") == pytest.approx(6.9)

    # Parse after 'of'
    line = "# 198 mirrors are inside a radius of 12.393 m"
    assert plot_mirrors._extract_float_after_keyword(line, "of") == pytest.approx(12.393)

    # Invalid value
    assert plot_mirrors._extract_float_after_keyword("# value: invalid", ":") is None

    # No keyword found
    assert plot_mirrors._extract_float_after_keyword("# no match", ":") is None

    # Empty after keyword
    assert plot_mirrors._extract_float_after_keyword("# Total surface area:", ":") is None


def test__read_mirror_file_metadata(tmp_path):
    """Test reading mirror file metadata."""
    # MST format
    mst_file = tmp_path / "mst_mirror.dat"
    mst_file.write_text(
        "# Command line: ...\n"
        "# Total mirror surface area: 107.2 m**2\n"
        "# Rmax = 6.90 m (Dmax = 13.80 m)\n"
        "# Data starts here\n"
        "100.0 200.0 150.0 2800.0 3\n"
    )

    metadata = plot_mirrors._read_mirror_file_metadata(mst_file)
    assert metadata["total_surface_area"] == pytest.approx(107.2)
    assert metadata["rmax"] == pytest.approx(6.9)

    # LST format
    lst_file = tmp_path / "lst_mirror.dat"
    lst_file.write_text(
        "# Mirror positions\n"
        "# 198 mirrors are inside a radius of 12.393 m\n"
        "# Total surface area: 390.98 m**2\n"
        "# Data starts here\n"
        "100.0 200.0 150.0 2800.0 3\n"
    )

    metadata = plot_mirrors._read_mirror_file_metadata(lst_file)
    assert metadata["total_surface_area"] == pytest.approx(390.98)
    assert metadata["rmax"] == pytest.approx(12.393)

    metadata = plot_mirrors._read_mirror_file_metadata(tmp_path / "nonexistent.dat")
    assert metadata == {}


def test_plot_mirror_layout_mst_mirror_ids(tmp_path):
    """Test that MST mirror IDs are reversed (N to 1) with mirror #1 at the bottom."""
    mock_fig = mock.MagicMock()
    mock_ax = mock.MagicMock()
    mock_ax.get_xlim.return_value = (-1000, 1000)
    mock_ax.get_ylim.return_value = (-1000, 1000)

    mock_mirrors = mock.MagicMock()
    mirror_table = Table(
        {
            "mirror_x": [100.0, 200.0, 300.0] * u.cm,
            "mirror_y": [100.0, 200.0, 300.0] * u.cm,
            "focal_length": [2800.0, 2850.0, 2900.0] * u.cm,
            "mirror_panel_id": [0, 1, 2],
        }
    )

    mock_mirrors.mirror_table = mirror_table
    mock_mirrors.mirror_diameter = 120.0 * u.cm
    mock_mirrors.shape_type = 1
    mock_mirrors.number_of_mirrors = 3

    mirror_file = tmp_path / "mst_mirror.dat"
    mirror_file.write_text("# MST mirror file\n100.0 200.0 120.0 2800.0 1\n")
    mock_mirrors._mirror_list_file = mirror_file

    with (
        mock.patch("matplotlib.pyplot.subplots", return_value=(mock_fig, mock_ax)),
        mock.patch("matplotlib.pyplot.colorbar") as mock_colorbar,
        mock.patch("simtools.visualization.plot_mirrors._add_mirror_labels") as mock_labels,
    ):
        mock_cbar = mock.MagicMock()
        mock_colorbar.return_value = mock_cbar

        plot_mirrors.plot_mirror_layout(
            mirrors=mock_mirrors,
            mirror_file_path=tmp_path / "test_mirror_list.dat",
            telescope_model_name="MSTN-01",
        )

        call_args = mock_labels.call_args
        mirror_ids = call_args[0][3]
        assert list(mirror_ids) == [3, 2, 1]


@pytest.mark.parametrize(
    ("telescope", "plot_function", "file_content"),
    [
        (
            "MSTN-01",
            "plot_mirror_segmentation",
            "# MST segmentation file\n100.0  0.0  150.0  2800.0  3  0.0  #%  id=1\n",
        ),
        (
            "LSTN-01",
            "plot_mirror_segmentation",
            "# LST segmentation file\n100.0  0.0  150.0  2800.0  3  0.0  #%  id=1\n",
        ),
        (
            "MSTN-01",
            "plot_mirror_shape_segmentation",
            "# MST shape segmentation\nhex 1 100.0 0.0 50.0 30.0\n",
        ),
        (
            "LSTN-01",
            "plot_mirror_shape_segmentation",
            "# LST shape segmentation\nhex 1 100.0 0.0 50.0 30.0\n",
        ),
    ],
)
@mock.patch("matplotlib.pyplot.subplots")
def test_telescope_rotation(mock_subplots, tmp_path, telescope, plot_function, file_content):
    """Test telescope-specific rotation for different plot types."""
    mock_fig = mock.MagicMock()
    mock_ax = mock.MagicMock()
    mock_ax.get_xlim.return_value = (-1000, 1000)
    mock_ax.get_ylim.return_value = (-1000, 1000)
    mock_subplots.return_value = (mock_fig, mock_ax)

    seg_file = tmp_path / "test_seg.dat"
    seg_file.write_text(file_content)

    patches_to_apply = []
    if plot_function == "plot_mirror_segmentation":
        patches_to_apply.append(mock.patch("matplotlib.pyplot.colorbar"))
    else:
        patches_to_apply.extend(
            [
                mock.patch("matplotlib.pyplot.grid"),
                mock.patch("matplotlib.pyplot.xlabel"),
                mock.patch("matplotlib.pyplot.ylabel"),
                mock.patch("matplotlib.pyplot.tick_params"),
            ]
        )

    with contextlib.ExitStack() as stack:
        for patch in patches_to_apply:
            stack.enter_context(patch)

        func = getattr(plot_mirrors, plot_function)
        fig = func(
            data_file_path=seg_file,
            telescope_model_name=telescope,
            parameter_type="primary_mirror_segmentation",
        )

        assert fig is not None
