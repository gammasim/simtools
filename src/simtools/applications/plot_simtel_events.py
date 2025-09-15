#!/usr/bin/python3

r"""
Plot simulated events.

Produces figures from one or more sim_telarray (.simtel.zst) files
It is meant to run after simulations (e.g., simtools-simulate-flasher,
simtools-simulate-illuminator).

What it does
------------
- Loads each provided sim_telarray file
- Generates selected plots (camera image, time traces, waveform matrices, peak timing, etc.)
- Optionally saves all figures to a single multi-page PDF per input file
- Optionally also saves individual PNGs

Command line arguments
----------------------
simtel_files (list, required)
    One or more sim_telarray files to visualize (.simtel.zst).
plots (list, optional)
    Which plots to generate. Choose from: event_image, time_traces, waveform_matrix,
    step_traces, integrated_signal_image, integrated_pedestal_image, peak_timing, all.
    Default: event_image.
tel_id (int, optional)
    Telescope ID to visualize. If omitted, the first available telescope will be used.
n_pixels (int, optional)
    For time_traces: number of pixel traces to draw. Default: 3.
pixel_step (int, optional)
    For step_traces and waveform_matrix: step between pixel indices. Default: 100.
max_pixels (int, optional)
    For step_traces: cap the number of plotted pixels. Default: None.
vmax (float, optional)
    For waveform_matrix: upper limit of color scale. Default: None.
half_width (int, optional)
    For integrated_*_image: half window width in samples. Default: 8.
offset (int, optional)
    For integrated_pedestal_image: offset between pedestal and peak windows. Default: 16.
sum_threshold (float, optional)
    For peak_timing: minimum pixel sum to consider a pixel. Default: 10.0.
peak_width (int, optional)
    For peak_timing: expected peak width in samples. Default: 8.
examples (int, optional)
    For peak_timing: show example traces. Default: 3.
timing_bins (int, optional)
    For peak_timing: number of histogram bins for peak sample. Default: None (contiguous bins).
distance (float, optional)
    Optional distance annotation for event_image.
output_file (str, optional)
    Base name for output. If provided, outputs will be placed under the standard IOHandler
    output directory and named ``<base>_<inputstem>.pdf``. If omitted, defaults are derived
    from each input file name.
save_pngs (flag, optional)
    Also save individual PNG files per figure.
dpi (int, optional)
    DPI for PNG outputs. Default: 300.
output_path (str, optional)
    Path to save the output files.

Examples
--------
1) Camera image and time traces for a single file, save a PDF:

   simtools-plot-simtel-events \
     --simtel_files tests/resources/ff-1m_flasher.simtel.zst \
     --plots event_image time_traces \
     --tel_id 1 \
     --output_file simulate_illuminator_inspect

2) All plots for multiple files, PNGs and PDFs:

   simtools-plot-simtel-events \
     --simtel_files f1.simtel.zst f2.simtel.zst \
     --plots all \
     --save_pngs --dpi 200

"""

import logging
from pathlib import Path

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.data_model.metadata_collector import MetadataCollector
from simtools.io import io_handler
from simtools.visualization.plot_corsika_histograms import save_figs_to_pdf
from simtools.visualization.plot_simtel_events import (
    plot_simtel_event_image,
    plot_simtel_integrated_pedestal_image,
    plot_simtel_integrated_signal_image,
    plot_simtel_peak_timing,
    plot_simtel_step_traces,
    plot_simtel_time_traces,
    plot_simtel_waveform_matrix,
)

PLOT_CHOICES = {
    "event_image": "event_image",
    "time_traces": "time_traces",
    "waveform_matrix": "waveform_matrix",
    "step_traces": "step_traces",
    "integrated_signal_image": "integrated_signal_image",
    "integrated_pedestal_image": "integrated_pedestal_image",
    "peak_timing": "peak_timing",
    "all": "all",
}


def _call_peak_timing(
    filename,
    *,
    tel_id=None,
    sum_threshold=10.0,
    peak_width=8,
    examples=3,
    timing_bins=None,
    event_index=None,
):
    """Call ``plot_simtel_peak_timing`` and support optional ``return_stats``.

    Parameters
    ----------
    filename : pathlib.Path or str
        Path to the input simtel file.
    tel_id : int, optional
        Telescope ID to visualize.
    sum_threshold : float, default 10.0
        Minimum pixel sum to consider a pixel.
    peak_width : int, default 8
        Expected peak width in samples.
    examples : int, default 3
        Number of example traces to draw.
    timing_bins : int or None, optional
        Number of bins for timing histogram (contiguous if not set).
    event_index : int or None, optional
        0-based index of the event to plot; default is the first event.

    Returns
    -------
    object or None
        The matplotlib Figure if available, otherwise ``None``.
    """
    try:
        fig_stats = plot_simtel_peak_timing(
            filename,
            tel_id=tel_id,
            sum_threshold=sum_threshold,
            peak_width=peak_width,
            examples=examples,
            timing_bins=timing_bins,
            return_stats=True,
            event_index=event_index,
        )
        return fig_stats[0] if isinstance(fig_stats, tuple) else fig_stats
    except TypeError:
        return plot_simtel_peak_timing(
            filename,
            tel_id=tel_id,
            sum_threshold=sum_threshold,
            peak_width=peak_width,
            examples=examples,
            timing_bins=timing_bins,
            event_index=event_index,
        )


def _parse(label: str):
    """Parse command line configuration."""
    config = configurator.Configurator(
        label=label,
        description=(
            "Create diagnostic plots from sim_telarray files using simtools visualization."
        ),
    )

    config.parser.add_argument(
        "--simtel_files",
        help="One or more sim_telarray files (.simtel.zst)",
        nargs="+",
        required=True,
    )
    config.parser.add_argument(
        "--plots",
        help=f"Plots to generate. Choices: {', '.join(sorted(PLOT_CHOICES))}",
        nargs="+",
        default=["event_image"],
        choices=sorted(PLOT_CHOICES),
    )
    # common plotting options
    config.parser.add_argument("--tel_id", type=int, default=None, help="Telescope ID")
    config.parser.add_argument(
        "--n_pixels", type=int, default=3, help="For time_traces: number of pixel traces"
    )
    config.parser.add_argument(
        "--pixel_step", type=int, default=100, help="Step between pixel ids for step plots"
    )
    config.parser.add_argument(
        "--max_pixels", type=int, default=None, help="Cap number of pixels for step traces"
    )
    config.parser.add_argument("--vmax", type=float, default=None, help="Color scale vmax")
    config.parser.add_argument(
        "--half_width", type=int, default=8, help="Half window width for integrated images"
    )
    config.parser.add_argument(
        "--offset",
        type=int,
        default=16,
        help="offset between pedestal and peak windows (integrated_pedestal_image)",
    )
    config.parser.add_argument(
        "--sum_threshold",
        type=float,
        default=10.0,
        help="Minimum pixel sum to consider in peak timing",
    )
    config.parser.add_argument(
        "--peak_width", type=int, default=8, help="Expected peak width in samples"
    )
    config.parser.add_argument(
        "--examples", type=int, default=3, help="Number of example traces to draw"
    )
    config.parser.add_argument(
        "--timing_bins",
        type=int,
        default=None,
        help="Number of bins for timing histogram (contiguous if not set)",
    )
    config.parser.add_argument(
        "--distance",
        type=float,
        default=None,
        help="Optional distance annotation for event_image (same units as input expects)",
    )
    config.parser.add_argument(
        "--event_index",
        type=int,
        default=None,
        help="0-based index of the event to plot; default is the first event",
    )
    # outputs
    config.parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help=(
            "Base name for output. If set, PDFs will be named '<base>_<inputstem>.pdf' "
            "in the standard output directory"
        ),
    )
    config.parser.add_argument(
        "--save_pngs",
        action="store_true",
        help="Also save individual PNG images per plot",
    )
    config.parser.add_argument("--dpi", type=int, default=300, help="PNG dpi")

    return config.initialize(db_config=False, require_command_line=True)


def _save_png(fig, out_dir: Path, stem: str, suffix: str, dpi: int):
    """Save ``fig`` as a PNG into ``out_dir`` using ``stem`` and ``suffix``.

    Errors during saving are logged as warnings and otherwise ignored.
    """
    png_path = out_dir.joinpath(f"{stem}_{suffix}.png")
    try:
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    except Exception as ex:  # pylint:disable=broad-except
        logging.getLogger(__name__).warning("Failed to save PNG %s: %s", png_path, ex)


def _make_output_paths(
    ioh: io_handler.IOHandler, base: str | None, input_file: Path
) -> tuple[Path, Path]:
    """Return (out_dir, pdf_path) based on base and input_file."""
    out_dir = ioh.get_output_directory(label=Path(__file__).stem)
    if base:
        pdf_path = ioh.get_output_file(f"{base}_{input_file.stem}")
    else:
        pdf_path = ioh.get_output_file(input_file.stem)
    pdf_path = Path(f"{pdf_path}.pdf") if pdf_path.suffix != ".pdf" else Path(pdf_path)
    return out_dir, pdf_path


def _collect_figures_for_file(
    filename: Path,
    plots: list[str],
    args: dict,
    out_dir: Path,
    base_stem: str,
    save_pngs: bool,
    dpi: int,
):
    """Generate the selected plots for a single sim_telarray file.

    Returns a list of figures. If ``save_pngs`` is True, also writes PNGs to
    ``out_dir`` using ``base_stem`` for filenames.
    """
    logger = logging.getLogger(__name__)
    figures: list[object] = []

    def add(fig, tag: str):
        if fig is not None:
            figures.append(fig)
            if save_pngs:
                _save_png(fig, out_dir, base_stem, tag, dpi)
        else:
            logger.warning("Plot '%s' returned no figure for %s", tag, filename)

    plots_to_run = (
        [
            "event_image",
            "time_traces",
            "waveform_matrix",
            "step_traces",
            "integrated_signal_image",
            "integrated_pedestal_image",
            "peak_timing",
        ]
        if "all" in plots
        else list(plots)
    )

    # function name -> (callable, defaults)
    dispatch: dict[str, tuple[object, dict[str, object]]] = {
        "event_image": (
            plot_simtel_event_image,
            {"distance": None, "event_index": None},
        ),
        "time_traces": (
            plot_simtel_time_traces,
            {"tel_id": None, "n_pixels": 3, "event_index": None},
        ),
        "waveform_matrix": (
            plot_simtel_waveform_matrix,
            {"tel_id": None, "vmax": None, "event_index": None},
        ),
        "step_traces": (
            plot_simtel_step_traces,
            {"tel_id": None, "pixel_step": None, "max_pixels": None, "event_index": None},
        ),
        "integrated_signal_image": (
            plot_simtel_integrated_signal_image,
            {"tel_id": None, "half_width": 8, "event_index": None},
        ),
        "integrated_pedestal_image": (
            plot_simtel_integrated_pedestal_image,
            {"tel_id": None, "half_width": 8, "offset": 16, "event_index": None},
        ),
        "peak_timing": (
            _call_peak_timing,
            {
                "tel_id": None,
                "sum_threshold": 10.0,
                "peak_width": 8,
                "examples": 3,
                "timing_bins": None,
                "event_index": None,
            },
        ),
    }

    for plot_name in plots_to_run:
        entry = dispatch.get(plot_name)
        if entry is None:
            logger.warning("Unknown plot selection '%s'", plot_name)
            continue
        func, defaults = entry
        # Build kwargs with user args overriding defaults
        kwargs = {k: args.get(k, v) for k, v in defaults.items()}
        fig = func(filename, **kwargs)  # type: ignore[misc]
        add(fig, plot_name)

    return figures


def main():
    """Generate plots from sim_telarray files."""
    label = Path(__file__).stem
    args, _db = _parse(label)

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args.get("log_level", "INFO")))

    ioh = io_handler.IOHandler()

    simtel_files = [Path(p).expanduser() for p in gen.ensure_iterable(args["simtel_files"])]
    plots = list(gen.ensure_iterable(args.get("plots")))

    for simtel in simtel_files:
        out_dir, pdf_path = _make_output_paths(ioh, args.get("output_file"), simtel)
        figures = _collect_figures_for_file(
            filename=simtel,
            plots=plots,
            args=args,
            out_dir=out_dir,
            base_stem=simtel.stem,
            save_pngs=bool(args.get("save_pngs", False)),
            dpi=int(args.get("dpi", 300)),
        )

        if not figures:
            logger.warning("No figures produced for %s", simtel)
            continue

        # Save a multipage PDF
        try:
            save_figs_to_pdf(figures, pdf_path)
            logger.info("Saved PDF: %s", pdf_path)
        except Exception as ex:  # pylint:disable=broad-except
            logger.error("Failed to save PDF %s: %s", pdf_path, ex)

        # Dump run metadata alongside PDF
        try:
            MetadataCollector.dump(args, pdf_path, add_activity_name=True)
        except Exception as ex:  # pylint:disable=broad-except
            logger.warning("Failed to write metadata for %s: %s", pdf_path, ex)


if __name__ == "__main__":
    main()
