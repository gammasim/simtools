#!/usr/bin/python3
"""Plot sim_telarray events."""

import logging
from contextlib import closing
from pathlib import Path

import numpy as np

from simtools.data_model.metadata_collector import MetadataCollector
from simtools.utils import value_conversion
from simtools.visualization.matplotlib_backend import pyplot as plt
from simtools.visualization.plot_camera import plot_pixel_layout_with_image
from simtools.visualization.visualize import pdf_report, save_figure

_logger = logging.getLogger(__name__)

TIME_NS_LABEL = "time [ns]"
R1_SAMPLES_LABEL = "R1 samples [d.c.]"
PIXEL_LABEL = "N pixels"
NO_FIGURES_MESSAGE = "No figures produced for %s"
SAVED_PDF_MESSAGE = "Saved PDF: %s"
PEDESTAL_SOURCE_WAVEFORM = "waveform"
PEDESTAL_SOURCE_MONITORING = "monitoring"
EVENT_ID_KEY = "event_id"

DEFAULT_SUM_THRESHOLD = 10.0
DEFAULT_HISTOGRAM_BINS = 50
DEFAULT_TIME_TRACES_PIXELS = 3
DEFAULT_STEP_TRACES_INTERVAL = 10

PLOT_CHOICES = [
    "pedestals",
    "signals",
    "peak_timing",
    "time_traces",
    "waveforms",
    "step_traces",
    "all",
]


def generate_and_save_plots(plots, args, ioh, prepared_events=None):
    """
    Generate events plots from a sim-telarray file and save a multi-page PDF per input.

    Writes additionally metadata.

    Parameters
    ----------
    plots : list of str
        List of plot names to generate.
    args : dict
        Additional arguments for plot functions.
    ioh : OutputPaths
        Paths used for output files and metadata.
    prepared_events : PreparedEvents
        Events prepared by ``camera.event_images``. Required to generate plots.
    """
    if prepared_events is None:
        raise ValueError("prepared_events must be supplied before generating plots.")
    plotter = _StreamingPlotter(
        file_name=args.get("simtel_file", None),
        telescope=args.get("telescope", None),
        event_ids=value_conversion.ensure_list(args.get(EVENT_ID_KEY, None)),
        max_events=args.get("max_events", None),
        prepared_events=prepared_events,
    )
    output_file = plotter.make_output_paths(ioh, args.get("output_file"))
    with closing(plotter.stream_figures(plots, args, output_file)) as figures:
        first = next(figures, None)
        if first is None:
            _logger.warning(NO_FIGURES_MESSAGE, plotter.file_name)
            return
        with pdf_report(output_file) as report:
            report.add(first)
            for figure in figures:
                report.add(figure)
    _logger.info(SAVED_PDF_MESSAGE, output_file)
    MetadataCollector.dump(args, output_file, add_activity_name=True)


class PlotSimtelEvent:
    """
    Plot a single sim_telarray event.

    Parameters
    ----------
    file_name : str | pathlib.Path
        Path to the sim_telarray file.
    telescope : str
        Telescope name or ID to process.
    event_ids : int or list of int, optional
        IDs of the event(s) to process.
    max_events : int, optional
        Maximum number of events to process.
    """

    def __init__(self, file_name, telescope, event_ids=None, max_events=None, prepared_events=None):
        """Initialize plotter for one or more events."""
        self.file_name = Path(file_name) if file_name else None
        self.telescope = telescope
        self.event_ids = event_ids
        self.max_events = max_events
        self.figures = []
        self.camera = None

        if prepared_events is None:
            raise ValueError("prepared_events must be supplied when constructing a plotter.")
        self.prepared_events = prepared_events
        self.event_data = prepared_events.event_data
        self.time_axis = prepared_events.time_axis
        self.camera = prepared_events.camera
        self._trace_maxima = prepared_events.trace_maxima

    def make_output_paths(self, ioh, base):
        """Return output file path based on base name and input file."""
        out_dir = ioh.get_output_directory()
        pdf_path = ioh.get_output_file(
            f"{base}_{self.file_name.stem}" if base else self.file_name.stem
        )
        pdf_path = Path(f"{pdf_path}.pdf") if Path(pdf_path).suffix != ".pdf" else Path(pdf_path)

        return out_dir / pdf_path.name

    def plot(self, plots, args, output_file, save_png=False, dpi=300):
        """
        Generate all requested plots for the event.

        Parameters
        ----------
        plots : list of str
            List of plot names to generate.
        args : dict
            Additional arguments for plot functions.
        output_file : Path
            Base output file path for saving plots.
        save_png : bool
            Whether to save individual PNG files per plot.
        dpi : int
            DPI for saved PNG files.
        """
        output_file = Path(output_file)
        plots = self._plots_to_run(plots)

        for plot_name in plots:
            for event_index in self.event_data:
                fig = self._generate_single_plot(plot_name, event_index, args)
                if fig is not None:
                    self.figures.append(fig)
                    if save_png:
                        png_path = output_file.with_name(f"{plot_name}_{event_index}.png")
                        save_figure(fig, png_path, figure_format=["png"], dpi=int(dpi))

    def _generate_single_plot(self, plot_name, event_index, args):
        """
        Generate a single plot.

        Parameters
        ----------
        plot_name : str
            Name of the plot to generate.
        event_index : int
            Event index.
        args : dict
            Additional arguments for plot function.

        Returns
        -------
        matplotlib.figure.Figure | None
            Generated figure or None if plot unknown.
        """
        entry = self._plot_definitions.get(plot_name)
        if entry is None:
            _logger.warning("Unknown plot selection '%s'", plot_name)
            return None

        func, defaults = entry
        kwargs = {k: args.get(k, v) for k, v in defaults.items()}
        kwargs["event_index"] = event_index
        return func(**kwargs)

    def save(self, args, output_file):
        """Save generated plots to files."""
        if not self.figures:
            _logger.warning(NO_FIGURES_MESSAGE, self.file_name)
            return
        with pdf_report(output_file) as report:
            for figure in self.figures:
                report.add(figure)
        _logger.info(SAVED_PDF_MESSAGE, output_file)
        MetadataCollector.dump(args, output_file, add_activity_name=True)

    def _plots_to_run(self, plots):
        """Generate list of plots to run based on user input."""
        if "all" in plots:
            return list(self._plot_definitions.keys())
        return value_conversion.ensure_list(plots)

    @property
    def _plot_definitions(self):
        """Return mapping of plot names to methods and their default arguments."""
        return {
            "pedestals": (self.plot_pedestals, {}),
            "signals": (self.plot_signals, {}),
            "peak_timing": (
                self.plot_peak_timing,
                {"sum_threshold": DEFAULT_SUM_THRESHOLD, "timing_bins": None},
            ),
            "time_traces": (
                self.plot_time_traces,
                {"number_of_pixels": DEFAULT_TIME_TRACES_PIXELS},
            ),
            "waveforms": (self.plot_waveforms, {"vmax": None, "pixel_step": None}),
            "step_traces": (
                self.plot_step_traces,
                {"pixel_step": DEFAULT_STEP_TRACES_INTERVAL, "max_pixels": None},
            ),
        }

    def _make_title(self, subject, event_index):
        """Generate consistent plot title."""
        return f"{self.telescope} {subject} (event {event_index})"

    def trace_maxima(self, samples, sum_threshold):
        """Call the trace analysis function supplied with prepared event data."""
        return self._trace_maxima(samples, sum_threshold)

    def plot_time_traces(self, event_index, number_of_pixels=DEFAULT_TIME_TRACES_PIXELS):
        """
        Plot R1 time traces for pixels with highest signal.

        Parameters
        ----------
        event_index : int
            Event index.
        number_of_pixels : int, optional
            Number of pixels with highest signal to plot.

        Returns
        -------
        matplotlib.figure.Figure
            The created figure.
        """
        image_flat = np.asarray(self.event_data[event_index].image).ravel()
        pix_ids = np.argsort(image_flat)[-number_of_pixels:][::-1]

        fig, ax = plt.subplots(dpi=300)
        for pid in pix_ids:
            (line,) = ax.plot(
                self.time_axis,
                self.event_data[event_index].adc_samples[pid],
                label=f"pix {int(pid)}",
                drawstyle="steps-mid",
            )
            plt.axhline(
                y=self.event_data[event_index].pedestals[pid],
                color=line.get_color(),
                linestyle="--",
                linewidth=0.5,
            )
        ax.set_xlabel(TIME_NS_LABEL)
        ax.set_ylabel(R1_SAMPLES_LABEL)
        ax.set_title(self._make_title("waveforms", event_index))
        ax.legend(loc="best", fontsize=7)
        fig.tight_layout()
        return fig

    def plot_waveforms(self, event_index, vmax=None, pixel_step=None):
        """
        Create a pseudocolor image of R1 waveforms (sample index vs. pixel id).

        Parameters
        ----------
        event_index : int
            Event index.
        vmax : float | None, optional
            Upper limit for color normalization. If None, determined automatically.
        pixel_step : int | None, optional
            Step between plotted pixel ids (e.g., 1 plots all, 2 plots every second pixel).

        Returns
        -------
        matplotlib.figure.Figure
            The created figure.
        """
        step = self._ensure_positive_int(pixel_step, default=1)
        pix_idx = np.arange(self.event_data[event_index].number_of_pixels)[::step]
        w_sel = self.event_data[event_index].adc_samples[pix_idx]

        fig, ax = plt.subplots(dpi=300)
        mesh = ax.pcolormesh(self.time_axis, pix_idx, w_sel, shading="auto", vmax=vmax)
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label(R1_SAMPLES_LABEL)
        ax.set_title(self._make_title("waveform matrix", event_index))
        ax.set_xlabel(TIME_NS_LABEL)
        ax.set_ylabel("pixel id")
        fig.tight_layout()
        return fig

    def plot_step_traces(
        self, event_index, pixel_step=DEFAULT_STEP_TRACES_INTERVAL, max_pixels=None
    ):
        """
        Plot step-style R1 traces for regularly sampled pixels (0, N, 2N, ...).

        Parameters
        ----------
        event_index : int
            Event index.
        pixel_step : int, optional
            Interval between pixel indices to plot.
        max_pixels : int | None, optional
            Maximum number of pixels to plot. If None, plot all selected by ``pixel_step``.

        Returns
        -------
        matplotlib.figure.Figure
            The created figure.
        """
        step = self._ensure_positive_int(pixel_step, default=DEFAULT_STEP_TRACES_INTERVAL)
        pix_ids = np.arange(0, self.event_data[event_index].number_of_pixels, step)
        if max_pixels is not None:
            pix_ids = pix_ids[:max_pixels]

        fig, ax = plt.subplots(dpi=300)
        for pid in pix_ids:
            ax.plot(
                self.time_axis,
                self.event_data[event_index].adc_samples[int(pid)],
                label=f"pix {int(pid)}",
                drawstyle="steps-mid",
            )
        ax.set_xlabel(TIME_NS_LABEL)
        ax.set_ylabel(R1_SAMPLES_LABEL)
        ax.set_title(self._make_title("step traces", event_index))
        ax.legend(loc="best", fontsize=7, ncol=2)
        fig.tight_layout()
        return fig

    def plot_peak_timing(self, event_index, sum_threshold=DEFAULT_SUM_THRESHOLD, timing_bins=None):
        """
        Peak finding per pixel; plot histogram of peak sample times.

        Parameters
        ----------
        event_index : int
            Event index.
        sum_threshold : float, optional
            Minimum sum over samples for a pixel to be considered.
        timing_bins : int | None, optional
            Number of histogram bins. If None, use unit-width bins.

        Returns
        -------
        matplotlib.figure.Figure | None
            The created figure, or None if no pixels exceed threshold.
        """
        trace_max_time, pix_ids, found_count = self.trace_maxima(
            self.event_data[event_index].adc_samples, sum_threshold=sum_threshold
        )

        if trace_max_time is None or pix_ids is None:
            _logger.warning("No pixels exceeded sum_threshold for peak timing")
            return None

        return self._plot_camera_image_and_histogram(
            trace_max_time,
            pix_ids,
            found_count,
            self._histogram_edges(timing_bins, self.event_data[event_index].n_samples),
            x_label="peak sample",
            y_label=PIXEL_LABEL,
            event_index=event_index,
        )

    def plot_signals(self, event_index):
        """Plot integrated trace values."""
        values = self.event_data[event_index].image
        return self._plot_camera_image_and_histogram(
            values,
            np.arange(self.event_data[event_index].number_of_pixels),
            self.event_data[event_index].number_of_pixels,
            self._create_histogram_edges(values, DEFAULT_HISTOGRAM_BINS),
            x_label="signal",
            y_label=PIXEL_LABEL,
            event_index=event_index,
        )

    def plot_pedestals(self, event_index):
        """Plot pedestal values for all pixels."""
        values = self.event_data[event_index].pedestals
        return self._plot_camera_image_and_histogram(
            values,
            np.arange(self.event_data[event_index].number_of_pixels),
            self.event_data[event_index].number_of_pixels,
            self._create_histogram_edges(values, DEFAULT_HISTOGRAM_BINS),
            x_label="pedestals",
            y_label=PIXEL_LABEL,
            event_index=event_index,
        )

    def _plot_camera_image_and_histogram(
        self, values, pix_ids, found_count, edges, x_label, y_label, event_index
    ):
        """Plot value image on camera and histogram of values."""
        stats = {
            "considered": int(pix_ids.size),
            "found": int(found_count),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
        }

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=300)

        plot_pixel_layout_with_image(self.camera, image=values, ax=ax1, color_bar_label=x_label)
        self._plot_histogram(ax2, values, edges, stats, x_label, y_label)

        fig.suptitle(self._make_title(x_label, event_index))
        fig.tight_layout()
        return fig

    def _plot_histogram(self, ax, values, edges, stats, x_label, y_label):
        """
        Draw a histogram of pixel distributions with overlays and annotations.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Target axes to draw into.
        values : numpy.ndarray
            Values to histogram.
        edges : numpy.ndarray
            Histogram bin edges.
        stats : dict
            Statistics dictionary with keys {"considered", "found", "mean", "median", "std"}.
        x_label : str
            Label for x-axis.
        y_label : str
            Label for y-axis.
        """
        counts, edges = np.histogram(values, bins=edges)
        ax.bar(edges[:-1], counts, width=np.diff(edges), color="#5B90DC", align="edge")
        ax.set_xlim(edges[0], edges[-1])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        self._add_statistics_to_histogram(ax, stats)
        ax.legend(fontsize=7)

    def _add_statistics_to_histogram(self, ax, stats):
        """
        Add statistical overlays and annotations to histogram axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Target axes.
        stats : dict
            Statistics dictionary.
        """
        self._lines_and_ranges(ax, stats)
        if "considered" in stats and "found" in stats:
            annotation_text = f"considered: {stats['considered']}\nwith data: {stats['found']}"
            ax.text(
                0.98,
                0.95,
                annotation_text,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7,
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "white",
                    "alpha": 0.6,
                    "linewidth": 0.0,
                },
            )

    def _lines_and_ranges(self, ax, stats, color="#2A9D8F"):
        """Draw median line and std range on histogram axes."""
        if "median" in stats:
            ax.axvline(
                stats["median"],
                color=color,
                linestyle="--",
                label=f"median={stats['median']:.2f}",
            )
        if "std" in stats and "median" in stats:
            ax.axvspan(
                stats["median"] - stats["std"],
                stats["median"] + stats["std"],
                color=color,
                alpha=0.2,
                label=f"std={stats['std']:.2f}",
            )

    def _histogram_edges(self, bins, n_samples):
        """
        Compute contiguous histogram bin edges for sample indices.

        Parameters
        ----------
        bins : int | None
            Number of histogram bins. If None, use unit-width bins.
        n_samples : int
            Number of samples in the data.

        Returns
        -------
        numpy.ndarray
            Array of bin edges spanning the sample index range.
        """
        if bins and bins > 0:
            return np.linspace(-0.5, n_samples - 0.5, int(bins) + 1)
        return np.arange(-0.5, n_samples + 0.5, 1.0)

    def _create_histogram_edges(self, values, n_bins):
        """
        Create histogram edges spanning the range of values.

        Parameters
        ----------
        values : numpy.ndarray
            Data values.
        n_bins : int
            Number of histogram bins.

        Returns
        -------
        numpy.ndarray
            Array of bin edges.
        """
        return np.linspace(np.min(values), np.max(values), n_bins)

    @staticmethod
    def _ensure_positive_int(value, default=1):
        """
        Ensure a value is a positive integer.

        Parameters
        ----------
        value : int | None
            Input value.
        default : int
            Default value if input is None.

        Returns
        -------
        int
            Positive integer value.
        """
        if value is None:
            return default
        return max(1, int(value))


class _StreamingPlotter(PlotSimtelEvent):
    """Generate figures one at a time for the application output."""

    def stream_figures(self, plots, args, output_file):
        """Generate and close each figure after it has been saved to the PDF."""
        output_file = Path(output_file)
        for plot_name in self._plots_to_run(plots):
            for event_index in self.event_data:
                fig = self._generate_single_plot(plot_name, event_index, args)
                if fig is None:
                    continue
                try:
                    yield fig
                    if args.get("save_pngs", False):
                        png_path = output_file.with_name(f"{plot_name}_{event_index}.png")
                        save_figure(
                            fig,
                            png_path,
                            figure_format=["png"],
                            dpi=int(args.get("figure_dpi", 300)),
                        )
                finally:
                    plt.close(fig)
