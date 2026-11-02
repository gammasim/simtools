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

from simtools.camera import event_images
from simtools.visualization import plot_simtel_events

SIGNALS = "signals"
PDF_SAVE_FAILED = "PDF save failed"
EVENT_ID_KEY = "event_id"
PEDESTAL_KEY = "pedestal"
N_PEDESTAL_SLICES_KEY = "n_ped_slices"
TELESCOPE_MONITORING = "camera_monitorings"
PEDESTAL_SOURCE_KEY = "pedestal_source"


@pytest.fixture
def mock_event_data():
    """Create minimal mock event data."""
    rng = np.random.default_rng(42)
    number_of_pixels, n_samples = 100, 50
    adc_samples = rng.uniform(50, 100, (2, number_of_pixels, n_samples))  # dual gain
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
    camera.number_of_pixels = 100
    camera.pixel_x_pos = rng.uniform(-1, 1, 100)
    camera.pixel_y_pos = rng.uniform(-1, 1, 100)
    camera.camera_name = "TestCamera"
    return camera


@pytest.fixture
def mock_plotter(mock_event_data, mock_camera):
    """Create PlotSimtelEvent with mocked data."""
    rng = np.random.default_rng(42)
    number_of_pixels, n_samples = 100, 50
    samples = rng.uniform(50, 100, (number_of_pixels, n_samples))
    samples[:, -10:] = 60
    samples[10, 20:30] = 200
    pedestals = np.full(number_of_pixels, 60.0)
    prepared = event_images.EventImages(
        event_data={
            0: event_images.EventData(
                event_id=0,
                adc_samples=samples,
                number_of_pixels=number_of_pixels,
                n_samples=n_samples,
                pedestals=pedestals,
                image=np.sum(samples[:, 4:] - pedestals[:, None], axis=1),
            )
        },
        time_axis=np.arange(n_samples),
        camera=mock_camera,
        trace_maxima=event_images.trace_maxima,
    )
    plotter = plot_simtel_events._StreamingPlotter(
        "fake.simtel", "LSTN-01", 0, prepared_events=prepared
    )
    plotter.prepared_events = prepared
    return plotter


def _mock_event_images(plotter):
    """Return a preparation record around plotter fixture data."""
    return event_images.EventImages(
        event_data=plotter.event_data,
        time_axis=plotter.time_axis,
        camera=plotter.camera,
        trace_maxima=event_images.trace_maxima,
    )


def test_plot_simtel_event_init(mock_plotter):
    assert mock_plotter.telescope == "LSTN-01"
    assert mock_plotter.event_ids == 0
    assert 0 in mock_plotter.event_data
    event = mock_plotter.event_data[0]
    assert event.number_of_pixels == 100
    assert event.n_samples == 50
    assert event.adc_samples.shape == (100, 50)
    assert event.pedestals.shape == (100,)
    assert event.image.shape == (100,)


def test_monitoring_pedestal_is_used_for_signal(mock_plotter, mocker):
    event = mock_plotter.event_data[0]
    second_event = event_images.EventData(
        event_id=1,
        adc_samples=event.adc_samples.copy(),
        number_of_pixels=event.number_of_pixels,
        n_samples=event.n_samples,
        pedestals=event.pedestals.copy(),
        image=event.image.copy(),
    )
    mock_plotter.event_data[1] = second_event
    prepared = _mock_event_images(mock_plotter)
    prepared.event_data = mock_plotter.event_data
    monitoring = {
        N_PEDESTAL_SLICES_KEY: 40,
        PEDESTAL_KEY: np.stack((np.full(100, 400 * 40), np.full(100, 600 * 40))),
    }
    simtel = mocker.patch.object(event_images, "SimTelFile")
    simtel.return_value.__iter__.return_value = iter(
        [
            {EVENT_ID_KEY: 42, TELESCOPE_MONITORING: {}},
            {EVENT_ID_KEY: 0, TELESCOPE_MONITORING: {1: monitoring}},
            {EVENT_ID_KEY: 1, TELESCOPE_MONITORING: {1: monitoring}},
        ]
    )
    mocker.patch.object(
        event_images,
        "get_sim_telarray_telescope_id_to_telescope_name_mapping",
        return_value={1: "LSTN-01"},
    )
    event_images.apply_monitoring_pedestals(
        prepared, mock_plotter.file_name, mock_plotter.telescope
    )

    assert np.all(prepared.event_data[0].pedestals == 400)
    expected = np.sum(prepared.event_data[0].adc_samples[:, 4:] - 400, axis=1)
    np.testing.assert_allclose(prepared.event_data[0].image, expected)
    assert np.all(prepared.event_data[1].pedestals == 400)


@pytest.mark.parametrize(
    "monitoring",
    [
        None,
        {N_PEDESTAL_SLICES_KEY: 0, PEDESTAL_KEY: [[400]]},
        {N_PEDESTAL_SLICES_KEY: 40, PEDESTAL_KEY: np.full((2, 99), 400 * 40)},
        {N_PEDESTAL_SLICES_KEY: 40, PEDESTAL_KEY: np.full((2, 100), np.nan)},
        {N_PEDESTAL_SLICES_KEY: 40, PEDESTAL_KEY: np.full((2, 100), np.inf)},
    ],
)
def test_missing_monitoring_pedestal_is_an_error(mock_plotter, mocker, monitoring):
    simtel = mocker.patch.object(event_images, "SimTelFile")
    simtel.return_value.__enter__.return_value = iter(
        [{EVENT_ID_KEY: 0, TELESCOPE_MONITORING: {1: monitoring}}]
    )
    mocker.patch.object(
        event_images,
        "get_sim_telarray_telescope_id_to_telescope_name_mapping",
        return_value={1: "LSTN-01"},
    )
    with pytest.raises(ValueError, match="monitoring pedestal"):
        event_images.apply_monitoring_pedestals(
            _mock_event_images(mock_plotter), mock_plotter.file_name, mock_plotter.telescope
        )


def test_monitoring_pedestal_rejects_unknown_telescope(mock_plotter, mocker):
    mocker.patch.object(
        event_images,
        "get_sim_telarray_telescope_id_to_telescope_name_mapping",
        return_value={},
    )
    with pytest.raises(ValueError, match="not present in the sim_telarray file"):
        event_images.apply_monitoring_pedestals(
            _mock_event_images(mock_plotter), mock_plotter.file_name, mock_plotter.telescope
        )


def test_monitoring_pedestal_rejects_unavailable_selected_event(mock_plotter, mocker):
    simtel = mocker.patch.object(event_images, "SimTelFile")
    simtel.return_value.__enter__.return_value = iter([])
    mocker.patch.object(
        event_images,
        "get_sim_telarray_telescope_id_to_telescope_name_mapping",
        return_value={1: "LSTN-01"},
    )
    with pytest.raises(ValueError, match="No monitoring pedestal found for event IDs \\[0\\]"):
        event_images.apply_monitoring_pedestals(
            _mock_event_images(mock_plotter), mock_plotter.file_name, mock_plotter.telescope
        )


def test_monitoring_pedestal_accepts_no_selected_events(mock_plotter, mocker):
    mock_plotter.prepared_events.event_data = {}
    mock_plotter.event_data = {}
    simtel = mocker.patch.object(event_images, "SimTelFile")
    simtel.return_value.__enter__.return_value = iter([])
    mocker.patch.object(
        event_images,
        "get_sim_telarray_telescope_id_to_telescope_name_mapping",
        return_value={1: "LSTN-01"},
    )

    event_images.apply_monitoring_pedestals(
        _mock_event_images(mock_plotter), mock_plotter.file_name, mock_plotter.telescope
    )


def test_make_title(mock_plotter):
    title = mock_plotter._make_title("test plot", 0)
    assert "LSTN-01" in title
    assert "test plot" in title
    assert "event 0" in title


def test_plot_definitions_property(mock_plotter):
    defs = mock_plotter._plot_definitions
    assert isinstance(defs, dict)
    assert "pedestals" in defs
    assert SIGNALS in defs
    assert "peak_timing" in defs
    assert "time_traces" in defs
    assert "waveforms" in defs
    assert "step_traces" in defs


def test_plots_to_run_all(mock_plotter):
    plots = mock_plotter._plots_to_run(["all"])
    assert "pedestals" in plots
    assert SIGNALS in plots
    assert len(plots) == 6


def test_plots_to_run_specific(mock_plotter):
    plots = mock_plotter._plots_to_run(["pedestals", SIGNALS])
    assert plots == ["pedestals", SIGNALS]


def test_plot_time_traces(mock_plotter):
    fig = mock_plotter.plot_time_traces(event_index=0, number_of_pixels=3)
    assert fig is not None
    assert len(fig.axes) == 1
    plt.close(fig)


def test_plot_waveforms(mock_plotter):
    fig = mock_plotter.plot_waveforms(event_index=0)
    assert fig is not None
    plt.close(fig)


def test_plot_waveforms_with_pixel_step(mock_plotter):
    fig = mock_plotter.plot_waveforms(event_index=0, pixel_step=5)
    assert fig is not None
    plt.close(fig)


def test_plot_step_traces(mock_plotter):
    fig = mock_plotter.plot_step_traces(event_index=0, pixel_step=50)
    assert fig is not None
    plt.close(fig)


def test_plot_step_traces_with_max_pixels(mock_plotter):
    fig = mock_plotter.plot_step_traces(event_index=0, pixel_step=10, max_pixels=3)
    assert fig is not None
    plt.close(fig)


def test_histogram_edges_default(mock_plotter):
    event = mock_plotter.event_data[0]
    edges = mock_plotter._histogram_edges(None, event.n_samples)
    assert len(edges) == event.n_samples + 1


def test_histogram_edges_with_bins(mock_plotter):
    event = mock_plotter.event_data[0]
    edges = mock_plotter._histogram_edges(20, event.n_samples)
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


def test_make_output_paths(mock_plotter, io_handler):
    output_path = mock_plotter.make_output_paths(io_handler, "test_output")
    assert output_path.suffix == ".pdf"
    assert "test_output" in output_path.name


def test_plot_method_unknown_plot(mock_plotter):
    mock_plotter.plot(["unknown_plot"], {}, Path("test.pdf"))
    assert len(mock_plotter.figures) == 0


def test_plot_pedestals(mock_plotter):
    rng = np.random.default_rng(42)
    event = mock_plotter.event_data[0]
    event.pedestals = rng.uniform(50, 70, event.number_of_pixels)
    with mock.patch("simtools.visualization.plot_simtel_events.plot_pixel_layout_with_image"):
        fig = mock_plotter.plot_pedestals(event_index=0)
        assert fig is not None
        plt.close(fig)


def test_plot_signals(mock_plotter):
    rng = np.random.default_rng(42)
    event = mock_plotter.event_data[0]
    event.image = rng.uniform(100, 500, event.number_of_pixels)
    with mock.patch("simtools.visualization.plot_simtel_events.plot_pixel_layout_with_image"):
        fig = mock_plotter.plot_signals(event_index=0)
        assert fig is not None
        plt.close(fig)


def test_plot_peak_timing_with_timing_bins(mock_plotter):
    with mock.patch("simtools.visualization.plot_simtel_events.plot_pixel_layout_with_image"):
        fig = mock_plotter.plot_peak_timing(event_index=0, sum_threshold=5.0, timing_bins=25)
        assert fig is not None
        plt.close(fig)


def test_plot_camera_image_and_histogram(mock_plotter):
    """Test camera image and histogram plotting."""
    rng = np.random.default_rng(42)
    event = mock_plotter.event_data[0]
    with mock.patch("simtools.visualization.plot_simtel_events.plot_pixel_layout_with_image"):
        values = rng.uniform(0, 100, event.number_of_pixels)
        pix_ids = np.arange(event.number_of_pixels)
        edges = np.linspace(0, 100, 50)

        fig = mock_plotter._plot_camera_image_and_histogram(
            values,
            pix_ids,
            event.number_of_pixels,
            edges,
            "test",
            "count",
            event_index=0,
        )
        assert fig is not None
        assert len(fig.axes) == 2
        plt.close(fig)


def test_plot_method_with_plots(mock_plotter):
    with mock.patch("simtools.visualization.plot_simtel_events.plot_pixel_layout_with_image"):
        mock_plotter.plot([SIGNALS], {}, Path("test.pdf"))
        assert len(mock_plotter.figures) == 1
        for fig in mock_plotter.figures:
            plt.close(fig)


def test_plot_method_with_save_png(mock_plotter):
    with mock.patch("simtools.visualization.plot_simtel_events.plot_pixel_layout_with_image"):
        with mock.patch("simtools.visualization.plot_simtel_events.save_figure"):
            mock_plotter.plot(["time_traces"], {}, Path("test.pdf"), save_png=True, dpi=150)
            assert len(mock_plotter.figures) == 1
            for fig in mock_plotter.figures:
                plt.close(fig)


def test_save_method(mock_plotter, io_handler, tmp_path):
    fig1, fig2 = plt.figure(), plt.figure()
    mock_plotter.figures = [fig1, fig2]
    output_file = tmp_path / "test_output.pdf"

    with mock.patch(
        "simtools.visualization.plot_simtel_events.MetadataCollector.dump"
    ) as mock_dump:
        mock_plotter.save({}, output_file)
        mock_dump.assert_called_once()
    assert output_file.exists()
    assert Path(output_file).stat().st_size > 0

    plt.close(fig1)
    plt.close(fig2)


def test_save_method_no_figures(mock_plotter, tmp_path):
    output_file = tmp_path / "test_output.pdf"

    with mock.patch("simtools.visualization.plot_simtel_events.pdf_report") as mock_save:
        with mock.patch(
            "simtools.visualization.plot_simtel_events.MetadataCollector.dump"
        ) as mock_dump:
            mock_plotter.save({}, output_file)
            mock_save.assert_not_called()
            mock_dump.assert_not_called()


def test_generate_and_save_plots(mock_camera, io_handler, tmp_path):
    simtel_file = tmp_path / "test.simtel"
    simtel_file.touch()

    args = {
        "simtel_file": str(simtel_file),
        "telescope": "LSTN-01",
        EVENT_ID_KEY: 0,
        "output_file": "test_output",
        "save_pngs": False,
        "figure_dpi": 300,
    }

    rng = np.random.default_rng(42)
    samples = rng.uniform(50, 100, (100, 50))
    prepared = event_images.EventImages(
        event_data={
            0: event_images.EventData(
                0, samples, 100, 50, np.full(100, 60), np.sum(samples[:, 4:] - 60, axis=1)
            )
        },
        time_axis=np.arange(50),
        camera=mock_camera,
        trace_maxima=event_images.trace_maxima,
    )
    args["_prepared_events"] = prepared
    with mock.patch("simtools.visualization.plot_simtel_events.pdf_report") as mock_save:
        with mock.patch(
            "simtools.visualization.plot_simtel_events.MetadataCollector.dump"
        ) as mock_dump:
            plot_simtel_events.generate_and_save_plots(
                ["time_traces"], args | {"_prepared_events": prepared}, io_handler
            )
            mock_save.assert_called_once()
            mock_save.return_value.__enter__.return_value.add.assert_called_once()
            mock_dump.assert_called_once()


def test_generate_and_save_plots_applies_monitoring_before_plot(
    mock_plotter, io_handler, tmp_test_directory
):
    args = {PEDESTAL_SOURCE_KEY: plot_simtel_events.PEDESTAL_SOURCE_MONITORING}
    args["_prepared_events"] = mock_plotter.prepared_events
    output_file = tmp_test_directory / "events.pdf"

    def generate_figure(_plot_name, event_index, _args):
        assert np.all(mock_plotter.event_data[event_index].pedestals == 60)
        return plt.figure()

    with (
        mock.patch.object(plot_simtel_events, "_StreamingPlotter", return_value=mock_plotter),
        mock.patch.object(mock_plotter, "make_output_paths", return_value=output_file),
        mock.patch.object(mock_plotter, "_generate_single_plot", side_effect=generate_figure),
        mock.patch.object(plot_simtel_events.MetadataCollector, "dump"),
    ):
        plot_simtel_events.generate_and_save_plots(
            [SIGNALS], args | {"_prepared_events": mock_plotter.prepared_events}, io_handler
        )


def test_generate_many_plots_closes_figures(mock_plotter, io_handler, tmp_test_directory, caplog):
    mock_plotter.event_data = dict.fromkeys(range(25))
    mock_plotter.prepared_events.event_data = mock_plotter.event_data
    output_file = tmp_test_directory / "events.pdf"

    def generate_figure(_plot_name, event_index, _args):
        fig, ax = plt.subplots()
        ax.plot([event_index])
        return fig

    with (
        mock.patch.object(plot_simtel_events, "_StreamingPlotter", return_value=mock_plotter),
        mock.patch.object(mock_plotter, "make_output_paths", return_value=output_file),
        mock.patch.object(mock_plotter, "_generate_single_plot", side_effect=generate_figure),
        mock.patch.object(plot_simtel_events.MetadataCollector, "dump"),
    ):
        plot_simtel_events.generate_and_save_plots(
            [SIGNALS], {"_prepared_events": mock_plotter.prepared_events}, io_handler
        )

    assert output_file.exists()
    assert Path(output_file).stat().st_size > 0
    assert not plt.get_fignums()
    assert "More than 20 figures" not in caplog.text


def test_streamed_plot_saves_png_after_pdf_page(mock_plotter, tmp_test_directory):
    output_file = tmp_test_directory / "events.pdf"
    fig, ax = plt.subplots()
    ax.plot([1, 2])
    args = {"save_pngs": True, "figure_dpi": 150}
    with (
        mock.patch.object(mock_plotter, "_generate_single_plot", return_value=fig),
    ):
        figures = mock_plotter.stream_figures([SIGNALS], args, output_file)
        assert next(figures) is fig
        assert list(figures) == []
    assert (tmp_test_directory / f"{SIGNALS}_0.png").exists()
    assert fig.number not in plt.get_fignums()


def test_streamed_plot_closes_figure_on_pdf_error(mock_plotter, io_handler, tmp_test_directory):
    fig = plt.figure()

    class FailedReport:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def add(self, figure):
            plt.close(figure)
            raise RuntimeError(PDF_SAVE_FAILED)

    with (
        mock.patch.object(plot_simtel_events, "_StreamingPlotter", return_value=mock_plotter),
        mock.patch.object(mock_plotter, "make_output_paths", return_value=tmp_test_directory / "x"),
        mock.patch.object(mock_plotter, "_generate_single_plot", return_value=fig),
        mock.patch.object(plot_simtel_events, "pdf_report", return_value=FailedReport()),
        pytest.raises(RuntimeError, match=PDF_SAVE_FAILED),
    ):
        plot_simtel_events.generate_and_save_plots(
            [SIGNALS], {"_prepared_events": mock_plotter.prepared_events}, io_handler
        )
    assert fig.number not in plt.get_fignums()


def test_streamed_plots_skip_missing_figures(mock_plotter, io_handler, tmp_test_directory, caplog):
    with (
        mock.patch.object(plot_simtel_events, "_StreamingPlotter", return_value=mock_plotter),
        mock.patch.object(mock_plotter, "make_output_paths", return_value=tmp_test_directory / "x"),
        mock.patch.object(mock_plotter, "_generate_single_plot", return_value=None),
        mock.patch.object(plot_simtel_events, "pdf_report") as save_document,
        mock.patch.object(plot_simtel_events.MetadataCollector, "dump") as dump,
    ):
        plot_simtel_events.generate_and_save_plots(
            [SIGNALS], {"_prepared_events": mock_plotter.prepared_events}, io_handler
        )
    save_document.assert_not_called()
    dump.assert_not_called()
    assert "No figures produced" in caplog.text
