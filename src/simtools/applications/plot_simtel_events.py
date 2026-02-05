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
    Optional distance annotation for signals.
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
     --plots signals time_traces \
     --tel_id 1 \
     --output_file simulate_illuminator_inspect

2) All plots for multiple files, PNGs and PDFs:

   simtools-plot-simtel-events \
     --simtel_files f1.simtel.zst f2.simtel.zst \
     --plots all \
     --save_pngs --dpi 200

"""

from pathlib import Path

import simtools.utils.general as gen
from simtools.application_control import get_application_label, startup_application
from simtools.configuration import configurator
from simtools.visualization.plot_simtel_events import PLOT_CHOICES, generate_and_save_plots


def _parse():
    """Parse command line configuration."""
    config = configurator.Configurator(
        label=get_application_label(__file__),
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
        default=["signals"],
        choices=sorted(PLOT_CHOICES),
    )
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
    config.parser.add_argument(
        "--save_pngs",
        action="store_true",
        help="Also save individual PNG images per plot",
    )
    config.parser.add_argument("--dpi", type=int, default=300, help="PNG dpi")

    return config.initialize(
        db_config=False, simulation_model=["telescope"], output=True, require_command_line=True
    )


def main():
    """Generate plots from sim_telarray files."""
    app_context = startup_application(_parse)

    simtel_files = [
        Path(p).expanduser() for p in gen.ensure_iterable(app_context.args["simtel_files"])
    ]
    plots = list(gen.ensure_iterable(app_context.args.get("plots")))

    generate_and_save_plots(
        simtel_files=simtel_files, plots=plots, args=app_context.args, ioh=app_context.io_handler
    )


if __name__ == "__main__":
    main()
