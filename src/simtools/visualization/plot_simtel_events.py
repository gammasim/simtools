#!/usr/bin/python3
"""Plot sim_telarray events."""

import logging
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from simtools.camera.trace_analysis import (
    calculate_pedestals,
    get_adc_samples_per_gain,
    get_time_axis,
    trace_integration,
    trace_maxima,
)
from simtools.data_model.metadata_collector import MetadataCollector
from simtools.model.camera import Camera
from simtools.simtel.simtel_event_reader import read_events
from simtools.utils import general as gen
from simtools.visualization.plot_camera import plot_pixel_layout_with_image
from simtools.visualization.visualize import save_figure, save_figures_to_single_document

_logger = logging.getLogger(__name__)

# Reusable literal constants (duplicated from visualize to avoid circular deps)
AXES_FRACTION = "axes fraction"
NO_R1_WAVEFORMS_MSG = "No R1 waveforms available in event"
TIME_NS_LABEL = "time [ns]"
R1_SAMPLES_LABEL = "R1 samples [d.c.]"
PIXEL_LABEL = "N pixels"

PLOT_CHOICES = [
    "pedestals",
    "signals",
    "peak_timing",
    "time_traces",
    "waveforms",
    "step_traces",
    "all",
]


def generate_and_save_plots(simtel_files, plots, args, ioh):
    """
    Generate plots for files and save a multi-page PDF per input.

    One PDF per sim_telarray file is created.
    Writes additionally metadata for each sim_telarray file.
    """
    telescope = args.get("telescope", None)
    event_index = gen.ensure_iterable(args.get("event_index", 0))
    # TODO needs attention
    if not event_index:
        event_index = [0]

    for file_name in simtel_files:
        _logger.info(f"Processing file: {file_name}")
        for event_idx in event_index:
            _logger.info(f"  Event index: {event_idx}")
            plotter = PlotSimtelEvent(file_name, telescope, event_idx)
            output_file = plotter.make_output_paths(ioh, args.get("output_file"))
            plotter.plot(
                plots,
                args,
                output_file,
                save_png=bool(args.get("save_pngs", False)),
                dpi=int(args.get("dpi", 300)),
            )
            plotter.save(args, output_file)


class PlotSimtelEvent:
    """
    Plot a single sim_telarray event.

    Parameters
    ----------
    file_name : str | pathlib.Path
        Path to the sim_telarray file.
    telescope : str
        Telescope name or ID to process.
    event_index : int
        Index of the event to process.
    """

    def __init__(self, file_name, telescope, event_index):
        """Initialize plotter for a single event."""
        self.file_name = Path(file_name) if file_name else None
        self.telescope = telescope
        self.event_index = event_index
        self.figures = []
        self.camera = None

        self._read_and_init_event()

    def _read_and_init_event(self):
        """
        Read and initialize event data.

        Calculates pedestals, integrated image, time axis, and defines camera model.
        """
        _event_index, tel_desc, _events = read_events(
            self.file_name, self.telescope, self.event_index, max_events=1
        )
        if not _event_index and not _events:
            return

        self.event_index = _event_index[0]  # read a single event
        event = _events[0]

        self.adc_samples = get_adc_samples_per_gain(event.get("adc_samples", None))
        self.n_pixels, self.n_samples = self.adc_samples.shape

        self.pedestals = calculate_pedestals(
            self.adc_samples, start=self.n_samples - 10, end=self.n_samples
        )
        self.image = trace_integration(
            self.adc_samples,
            pedestals=self.pedestals,
            window=(4, self.n_samples),
        )

        self.time_axis = get_time_axis(
            sampling_rate=tel_desc["pixel_settings"]["time_slice"] * u.ns,
            n_samples=self.n_samples,
        )

        self.camera = Camera(
            telescope_name=self.telescope,
            camera_config_file=None,
            focal_length=tel_desc["camera_settings"]["focal_length"],
            camera_config_dict=tel_desc["camera_settings"],
        )

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
        plots = self._plots_to_run(plots)

        for plot_name in plots:
            entry = self._plot_definitions.get(plot_name)
            if entry is None:
                _logger.warning("Unknown plot selection '%s'", plot_name)
                continue
            func, defaults = entry
            kwargs = {k: args.get(k, v) for k, v in defaults.items()}
            fig = func(**kwargs)
            if fig is not None:
                self.figures.append(fig)

            if save_png:
                save_figure(
                    fig,
                    output_file.with_name(f"{plot_name}.png"),
                    figure_format=["png"],
                    dpi=int(dpi),
                )

    def save(self, args, output_file):
        """Save generated plots to files."""
        if not self.figures:
            _logger.warning("No figures produced for %s", self.file_name)

        save_figures_to_single_document(self.figures, output_file)
        _logger.info("Saved PDF: %s", output_file)
        MetadataCollector.dump(args, output_file, add_activity_name=True)

    def _plots_to_run(self, plots):
        """Generate list of plots to run based on user input."""
        if "all" in plots:
            return list(self._plot_definitions.keys())
        return gen.ensure_iterable(plots)

    @property
    def _plot_definitions(self):
        """Return mapping of plot names to methods."""
        return {
            "pedestals": (self.plot_pedestals, {}),
            "signals": (self.plot_signals, {}),
            "peak_timing": (
                self.plot_peak_timing,
                {"sum_threshold": 10.0, "timing_bins": None},
            ),
            "time_traces": (self.plot_time_traces, {"n_pixels": 3}),
            "waveforms": (self.plot_waveforms, {"vmax": None}),
            "step_traces": (
                self.plot_step_traces,
                {"pixel_step": None, "max_pixels": None},
            ),
        }

    def _make_title(self, subject):
        """Generate consistent plot title."""
        return f"{self.telescope} {subject} (event {self.event_index})"

    def plot_time_traces(self, n_pixels=3):
        """
        Plot R1 time traces for a few pixels of one event.

        Parameters
        ----------
        n_pixels : int, optional
            Number of pixels with highest signal to plot.

        Returns
        -------
        matplotlib.figure.Figure | None
            The created figure, or ``None`` if R1 waveforms are unavailable.
        """
        image_flat = np.asarray(self.image).ravel()
        pix_ids = np.argsort(image_flat)[-n_pixels:][::-1]  # brightest n_pixels

        fig, ax = plt.subplots(dpi=300)
        for pid in pix_ids:
            (line,) = ax.plot(
                self.time_axis,
                self.adc_samples[pid],
                label=f"pix {int(pid)}",
                drawstyle="steps-mid",
            )
            plt.axhline(
                y=self.pedestals[pid], color=line.get_color(), linestyle="--", linewidth=0.5
            )
        ax.set_xlabel(TIME_NS_LABEL)
        ax.set_ylabel(R1_SAMPLES_LABEL)
        ax.set_title(self._make_title("waveforms"))
        ax.legend(loc="best", fontsize=7)
        fig.tight_layout()
        return fig

    def plot_waveforms(self, vmax=None, pixel_step=None):
        """
        Create a pseudocolor image of R1 waveforms (sample index vs. pixel id).

        Parameters
        ----------
        vmax : float | None, optional
            Upper limit for color normalization. If None, determined automatically.
        pixel_step : int | None, optional
            Step between plotted pixel ids (e.g., 1 plots all, 2 plots every second pixel).

        Returns
        -------
        matplotlib.figure.Figure | None
            The created figure, or ``None`` if R1 waveforms are unavailable.
        """
        step = max(1, int(pixel_step)) if pixel_step is not None else 1
        pix_idx = np.arange(self.n_pixels)[::step]
        w_sel = self.adc_samples[pix_idx]

        fig, ax = plt.subplots(dpi=300)
        mesh = ax.pcolormesh(self.time_axis, pix_idx, w_sel, shading="auto", vmax=vmax)
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label(R1_SAMPLES_LABEL)
        ax.set_title(self._make_title("waveform matrix"))
        ax.set_xlabel(TIME_NS_LABEL)
        ax.set_ylabel("pixel id")
        fig.tight_layout()
        return fig

    def plot_step_traces(self, pixel_step=100, max_pixels=None):
        """
        Plot step-style R1 traces for regularly sampled pixels (0, N, 2N, ...).

        Parameters
        ----------
        pixel_step : int, optional
            Interval between pixel indices to plot. Default is 100.
        max_pixels : int | None, optional
            Maximum number of pixels to plot. If None, plot all selected by ``pixel_step``.

        Returns
        -------
        matplotlib.figure.Figure | None
            The created figure, or ``None`` if R1 waveforms are unavailable.
        """
        pix_ids = np.arange(0, self.n_pixels, max(1, pixel_step))
        if max_pixels is not None:
            pix_ids = pix_ids[:max_pixels]

        fig, ax = plt.subplots(dpi=300)
        for pid in pix_ids:
            ax.plot(
                self.time_axis,
                self.adc_samples[int(pid)],
                label=f"pix {int(pid)}",
                drawstyle="steps-mid",
            )
        ax.set_xlabel(TIME_NS_LABEL)
        ax.set_ylabel(R1_SAMPLES_LABEL)
        ax.set_title(self._make_title("step traces"))
        ax.legend(loc="best", fontsize=7, ncol=2)
        fig.tight_layout()
        return fig

    def plot_peak_timing(self, sum_threshold=10.0, timing_bins=None):
        """
        Peak finding per pixel; report mean/std of peak sample and plot a histogram.

        Parameters
        ----------
        sum_threshold : float, optional
            Minimum sum over samples for a pixel to be considered. Default is 10.0.
        timing_bins : int | None, optional
            Number of histogram bins. If None, use unit-width bins.

        Returns
        -------
        matplotlib.figure.Figure | tuple[matplotlib.figure.Figure, dict] | None
            The created figure, or ``None`` if R1 waveforms are unavailable. If
            ``return_stats`` is True, a tuple ``(fig, stats)`` is returned, where
            ``stats`` has keys ``{"considered", "found", "mean", "std"}``.
        """
        trace_max_time, pix_ids, found_count = trace_maxima(
            self.adc_samples, sum_threshold=sum_threshold
        )

        if trace_max_time is None or pix_ids is None:
            _logger.warning("No pixels exceeded sum_threshold for peak timing")
            return None

        return self._plot_camera_image_and_histogram(
            trace_max_time,
            pix_ids,
            found_count,
            self._histogram_edges(timing_bins),
            x_label="peak sample",
            y_label=PIXEL_LABEL,
        )

    def plot_signals(self):
        """Plot integrated trace values."""
        return self._plot_camera_image_and_histogram(
            self.image,
            np.arange(self.n_pixels),
            self.n_pixels,
            np.linspace(np.min(self.image), np.max(self.image), 50),
            x_label="signal",
            y_label=PIXEL_LABEL,
        )

    def plot_pedestals(self):
        """Plot pedestal values for all pixels."""
        return self._plot_camera_image_and_histogram(
            self.pedestals,
            np.arange(self.n_pixels),
            self.n_pixels,
            np.linspace(np.min(self.pedestals), np.max(self.pedestals), 50),
            x_label="pedestals",
            y_label=PIXEL_LABEL,
        )

    def _plot_camera_image_and_histogram(
        self, values, pix_ids, found_count, edges, x_label, y_label
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

        fig.suptitle(self._make_title(x_label))
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
            Statistics dictionary with keys {"considered", "found", "mean", "std"}.
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
        self._lines_and_ranges(ax, stats)
        if "considered" in stats and "found" in stats:
            ax.text(
                0.98,
                0.95,
                f"considered: {stats['considered']}\nwith data: {stats['found']}",
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
        ax.legend(fontsize=7)

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

    def _histogram_edges(self, bins):
        """
        Compute contiguous histogram bin edges for sample indices.

        Parameters
        ----------
        timing_bins : int | None
            Number of histogram bins. If None, use unit-width bins.

        Returns
        -------
        numpy.ndarray
            Array of bin edges spanning the sample index range.
        """
        if bins and bins > 0:
            return np.linspace(-0.5, self.n_samples - 0.5, int(bins) + 1)
        return np.arange(-0.5, self.n_samples + 0.5, 1.0)
