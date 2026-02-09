#!/usr/bin/python3

"""
Unit tests for plot_simtel_events module.

Tests cover:
- PlotSimtelEvent initialization and setup
- Plot generation methods (time_traces, waveforms, step_traces)
- Helper methods (_make_title, _histogram_edges, _plot_histogram, _lines_and_ranges)
- Plot routing and selection (_plots_to_run, _plot_definitions)
- Output path generation
- Constants validation (PLOT_CHOICES)
"""

# pylint: disable=protected-access,redefined-outer-name,unused-argument

from pathlib import Path
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np
import pytest

from simtools.visualization import plot_simtel_events


@pytest.fixture
def mock_event_data():
    """Create minimal mock event data."""
    rng = np.random.default_rng(42)
    n_pixels, n_samples = 100, 50
    adc_samples = rng.uniform(50, 100, (2, n_pixels, n_samples))  # dual gain
    adc_samples[:, :, -10:] = 60  # pedestals
    adc_samples[:, 10, 20:30] = 200  # signal in pixel 10

    tel_desc = {
        "pixel_settings": {"time_slice": 1.0},
        "camera_settings": {
            "focal_length": 5.6,
            "pixel_shape": "hexagonal",
            "pixel_spacing": 0.05,
            "pixel_diameter": 0.05,
        },
    }

    event = {"adc_samples": adc_samples}

    return [0], tel_desc, [event]


@pytest.fixture
def mock_camera():
    """Create a mock camera with minimal required attributes."""
    rng = np.random.default_rng(42)
    camera = mock.Mock()
    camera.n_pixels = 100
    camera.pixel_x_pos = rng.uniform(-1, 1, 100)
    camera.pixel_y_pos = rng.uniform(-1, 1, 100)
    camera.camera_name = "TestCamera"
    return camera


@pytest.fixture
def mock_plotter(mock_event_data, mock_camera):
    """Create PlotSimtelEvent with mocked data."""
    with mock.patch("simtools.visualization.plot_simtel_events.read_events") as mock_read:
        mock_read.return_value = mock_event_data
        with mock.patch("simtools.visualization.plot_simtel_events.Camera") as mock_cam_class:
            mock_cam_class.return_value = mock_camera
            plotter = plot_simtel_events.PlotSimtelEvent("fake.simtel", "LSTN-01", 0)
    return plotter


def test_plot_simtel_event_init(mock_plotter):
    assert mock_plotter.telescope == "LSTN-01"
    assert mock_plotter.event_index == 0
    assert mock_plotter.n_pixels == 100
    assert mock_plotter.n_samples == 50
    assert mock_plotter.adc_samples.shape == (100, 50)
    assert mock_plotter.pedestals.shape == (100,)
    assert mock_plotter.image.shape == (100,)


def test_make_title(mock_plotter):
    title = mock_plotter._make_title("test plot")
    assert "LSTN-01" in title
    assert "test plot" in title
    assert "event 0" in title


def test_plot_definitions_property(mock_plotter):
    defs = mock_plotter._plot_definitions
    assert isinstance(defs, dict)
    assert "pedestals" in defs
    assert "signals" in defs
    assert "peak_timing" in defs
    assert "time_traces" in defs
    assert "waveforms" in defs
    assert "step_traces" in defs


def test_plots_to_run_all(mock_plotter):
    plots = mock_plotter._plots_to_run(["all"])
    assert "pedestals" in plots
    assert "signals" in plots
    assert len(plots) == 6


def test_plots_to_run_specific(mock_plotter):
    plots = mock_plotter._plots_to_run(["pedestals", "signals"])
    assert plots == ["pedestals", "signals"]


def test_plot_time_traces(mock_plotter):
    fig = mock_plotter.plot_time_traces(n_pixels=3)
    assert fig is not None
    assert len(fig.axes) == 1
    plt.close(fig)


def test_plot_waveforms(mock_plotter):
    fig = mock_plotter.plot_waveforms()
    assert fig is not None
    plt.close(fig)


def test_plot_waveforms_with_vmax(mock_plotter):
    fig = mock_plotter.plot_waveforms(vmax=150)
    assert fig is not None
    plt.close(fig)


def test_plot_waveforms_with_pixel_step(mock_plotter):
    fig = mock_plotter.plot_waveforms(pixel_step=5)
    assert fig is not None
    plt.close(fig)


def test_plot_step_traces(mock_plotter):
    fig = mock_plotter.plot_step_traces(pixel_step=50)
    assert fig is not None
    plt.close(fig)


def test_plot_step_traces_with_max_pixels(mock_plotter):
    fig = mock_plotter.plot_step_traces(pixel_step=10, max_pixels=3)
    assert fig is not None
    plt.close(fig)


def test_histogram_edges_default(mock_plotter):
    edges = mock_plotter._histogram_edges(None)
    assert len(edges) == mock_plotter.n_samples + 1


def test_histogram_edges_with_bins(mock_plotter):
    edges = mock_plotter._histogram_edges(20)
    assert len(edges) == 21


def test_plot_histogram(mock_plotter):
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots()
    values = rng.uniform(0, 100, 100)
    edges = np.linspace(0, 100, 20)
    stats = {"considered": 100, "found": 95, "mean": 50.0, "median": 51.0, "std": 10.0}

    mock_plotter._plot_histogram(ax, values, edges, stats, "x", "y")
    assert len(ax.patches) > 0
    plt.close(fig)


def test_lines_and_ranges(mock_plotter):
    fig, ax = plt.subplots()
    stats = {"median": 50.0, "std": 10.0}

    mock_plotter._lines_and_ranges(ax, stats)
    assert len(ax.lines) > 0
    plt.close(fig)


def test_make_output_paths(mock_plotter, io_handler):
    output_path = mock_plotter.make_output_paths(io_handler, "test_output")
    assert output_path.suffix == ".pdf"
    assert "test_output" in output_path.name


def test_plot_method_unknown_plot(mock_plotter):
    mock_plotter.plot(["unknown_plot"], {}, Path("test.pdf"))
    assert len(mock_plotter.figures) == 0


def test_plot_choices_constant():
    """Verify PLOT_CHOICES matches available plots."""
    assert "pedestals" in plot_simtel_events.PLOT_CHOICES
    assert "signals" in plot_simtel_events.PLOT_CHOICES
    assert "peak_timing" in plot_simtel_events.PLOT_CHOICES
    assert "all" in plot_simtel_events.PLOT_CHOICES
    assert len(plot_simtel_events.PLOT_CHOICES) == 7


def test_plot_pedestals(mock_plotter):
    """Test pedestal plot generation."""
    rng = np.random.default_rng(42)
    # Ensure pedestals have variation to avoid xlim warning
    mock_plotter.pedestals = rng.uniform(50, 70, mock_plotter.n_pixels)
    with mock.patch("simtools.visualization.plot_simtel_events.plot_pixel_layout_with_image"):
        fig = mock_plotter.plot_pedestals()
        assert fig is not None
        plt.close(fig)


def test_plot_signals(mock_plotter):
    """Test signal plot generation."""
    rng = np.random.default_rng(42)
    # Ensure signals have variation
    mock_plotter.image = rng.uniform(100, 500, mock_plotter.n_pixels)
    with mock.patch("simtools.visualization.plot_simtel_events.plot_pixel_layout_with_image"):
        fig = mock_plotter.plot_signals()
        assert fig is not None
        plt.close(fig)


def test_plot_peak_timing_with_timing_bins(mock_plotter):
    """Test peak timing with custom timing bins."""
    with mock.patch("simtools.visualization.plot_simtel_events.plot_pixel_layout_with_image"):
        fig = mock_plotter.plot_peak_timing(sum_threshold=5.0, timing_bins=25)
        assert fig is not None
        plt.close(fig)


def test_plot_camera_image_and_histogram(mock_plotter):
    """Test camera image and histogram plotting."""
    rng = np.random.default_rng(42)
    with mock.patch("simtools.visualization.plot_simtel_events.plot_pixel_layout_with_image"):
        values = rng.uniform(0, 100, mock_plotter.n_pixels)
        pix_ids = np.arange(mock_plotter.n_pixels)
        edges = np.linspace(0, 100, 50)

        fig = mock_plotter._plot_camera_image_and_histogram(
            values, pix_ids, mock_plotter.n_pixels, edges, "test", "count"
        )
        assert fig is not None
        assert len(fig.axes) == 2
        plt.close(fig)


def test_plot_method_with_plots(mock_plotter):
    """Test plot method generates figures."""
    with mock.patch("simtools.visualization.plot_simtel_events.plot_pixel_layout_with_image"):
        mock_plotter.plot(["signals"], {}, Path("test.pdf"))
        assert len(mock_plotter.figures) == 1


def test_plot_method_with_save_png(mock_plotter):
    """Test plot method with PNG saving."""
    with mock.patch("simtools.visualization.plot_simtel_events.plot_pixel_layout_with_image"):
        with mock.patch("simtools.visualization.plot_simtel_events.save_figure"):
            mock_plotter.plot(["time_traces"], {}, Path("test.pdf"), save_png=True, dpi=150)
            assert len(mock_plotter.figures) == 1


def test_save_method(mock_plotter, io_handler, tmp_path):
    """Test save method."""
    mock_plotter.figures = [plt.figure(), plt.figure()]
    output_file = tmp_path / "test_output.pdf"

    with mock.patch(
        "simtools.visualization.plot_simtel_events.save_figures_to_single_document"
    ) as mock_save:
        with mock.patch(
            "simtools.visualization.plot_simtel_events.MetadataCollector.dump"
        ) as mock_dump:
            mock_plotter.save({}, output_file)
            mock_save.assert_called_once()
            mock_dump.assert_called_once()
            assert mock_save.call_args[0][0] == mock_plotter.figures

    for fig in mock_plotter.figures:
        plt.close(fig)


def test_save_method_no_figures(mock_plotter, tmp_path):
    """Test save method with no figures."""
    output_file = tmp_path / "test_output.pdf"

    with mock.patch(
        "simtools.visualization.plot_simtel_events.save_figures_to_single_document"
    ) as mock_save:
        with mock.patch(
            "simtools.visualization.plot_simtel_events.MetadataCollector.dump"
        ) as mock_dump:
            mock_plotter.save({}, output_file)
            mock_save.assert_not_called()
            mock_dump.assert_not_called()


def test_read_and_init_event_no_events():
    """Test initialization with no events."""
    with mock.patch("simtools.visualization.plot_simtel_events.read_events") as mock_read:
        mock_read.return_value = ([], {}, [])
        with mock.patch("simtools.visualization.plot_simtel_events.Camera"):
            with pytest.raises(ValueError, match="No events read from file"):
                plot_simtel_events.PlotSimtelEvent("fake.simtel", "LSTN-01", 0)


def test_generate_and_save_plots(mock_event_data, mock_camera, io_handler, tmp_path):
    """Test generate_and_save_plots function."""
    simtel_files = [tmp_path / "test.simtel"]
    simtel_files[0].touch()

    args = {
        "telescope": "LSTN-01",
        "event_index": 0,
        "output_file": "test_output",
        "save_pngs": False,
        "dpi": 300,
    }

    with mock.patch("simtools.visualization.plot_simtel_events.read_events") as mock_read:
        mock_read.return_value = mock_event_data
        with mock.patch("simtools.visualization.plot_simtel_events.Camera") as mock_cam:
            mock_cam.return_value = mock_camera
            with mock.patch(
                "simtools.visualization.plot_simtel_events.save_figures_to_single_document"
            ) as mock_save:
                with mock.patch(
                    "simtools.visualization.plot_simtel_events.MetadataCollector.dump"
                ) as mock_dump:
                    plot_simtel_events.generate_and_save_plots(
                        simtel_files, ["time_traces"], args, io_handler
                    )
                    mock_save.assert_called_once()
                    mock_dump.assert_called_once()


def test_generate_and_save_plots_multiple_events(
    mock_event_data, mock_camera, io_handler, tmp_path
):
    """Test generate_and_save_plots with multiple events."""
    simtel_files = [tmp_path / "test.simtel"]
    simtel_files[0].touch()

    args = {
        "telescope": "LSTN-01",
        "event_index": [0, 1],
        "output_file": "test_output",
        "save_pngs": True,
        "dpi": 150,
    }

    with mock.patch("simtools.visualization.plot_simtel_events.read_events") as mock_read:
        mock_read.return_value = mock_event_data
        with mock.patch("simtools.visualization.plot_simtel_events.Camera") as mock_cam:
            mock_cam.return_value = mock_camera
            with mock.patch("simtools.visualization.plot_simtel_events.save_figure") as _:
                with mock.patch(
                    "simtools.visualization.plot_simtel_events.save_figures_to_single_document"
                ) as mock_save_doc:
                    with mock.patch(
                        "simtools.visualization.plot_simtel_events.MetadataCollector.dump"
                    ) as mock_dump:
                        plot_simtel_events.generate_and_save_plots(
                            simtel_files, ["waveforms"], args, io_handler
                        )
                        assert mock_save_doc.call_count == 2
                        assert mock_dump.call_count == 2
