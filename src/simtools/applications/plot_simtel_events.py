#!/usr/bin/python3

r"""
Plot simulated events from sim_telarray files.

Produces diagnostic figures from sim_telarray (.simtel.zst) files.
Meant to run after simulations (e.g., simtools-simulate-flasher,
simtools-simulate-illuminator).

What it does
------------
- Loads the provided sim_telarray file
- Generates selected plots (signals, pedestals, time traces, waveforms, peak timing, etc.)
- Saves all figures to a single multi-page PDF
- Optionally also saves individual PNG files per figure

Command line arguments
----------------------
simtel_file (str, required)
    A sim_telarray file to visualize (.simtel.zst).
telescope (str, required)
    Telescope name to process (e.g., LSTN-04, MSTN-01).
plots (list, optional)
    Which plots to generate. Choices: pedestals, signals, peak_timing, time_traces,
    waveforms, step_traces, all. Default: all.
n_pixels (int, optional)
    For time_traces: number of brightest pixel traces to plot. Default: 3.
pixel_step (int, optional)
    For step_traces and waveforms: step between pixel indices. Default: 100.
max_pixels (int, optional)
    For step_traces: maximum number of pixels to plot. Default: None (no limit).
vmax (float, optional)
    For waveforms: upper limit of color scale. Default: None (auto-scale).
sum_threshold (float, optional)
    For peak_timing: minimum pixel sum to consider. Default: 10.0.
timing_bins (int, optional)
    For peak_timing: number of histogram bins. Default: None (unit-width bins).
event_id (int or list, optional)
    Specific event ID(s) to plot. Default: None (first event).
max_events (int, optional)
    Maximum number of events to process. Default: 1.
output_file (str, optional)
    Base name for output files. PDF will be named ``<base>_<inputstem>.pdf``.
    If omitted, uses input file stem.
save_pngs (flag, optional)
    Also save individual PNG files per plot.
dpi (int, optional)
    Resolution for PNG outputs. Default: 300.
output_path (str, optional)
    Directory for output files.

Examples
--------
1) Plot signals and time traces for a telescope:

   simtools-plot-simtel-events \\
     --simtel_file run000010_North_7.0.0_simulate_flasher.simtel.zst \\
     --telescope LSTN-04 \\
     --plots signals time_traces \\
     --output_file flasher_inspect

2) Generate all plots with PNG outputs:

   simtools-plot-simtel-events \\
     --simtel_file run000010.simtel.zst \\
     --telescope MSTN-01 \\
     --plots all \\
     --save_pngs --dpi 200

3) Plot specific events:

   simtools-plot-simtel-events \\
     --simtel_file run000010.simtel.zst \\
     --telescope LSTN-04 \\
     --event_id 5 10 15 \\
     --plots signals pedestals

"""

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
        "--simtel_file",
        help="Input sim_telarray file (.simtel.zst)",
        required=True,
    )
    config.parser.add_argument(
        "--plots",
        help=f"Plots to generate. Choices: {', '.join(sorted(PLOT_CHOICES))}",
        nargs="+",
        default=["all"],
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
        "--event_id",
        type=int,
        nargs="+",
        default=None,
        help="Event ID(s) of the events to be plotted",
    )
    config.parser.add_argument(
        "--max_events",
        type=int,
        default=1,
        help="Maximum number of events to process",
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
    """Generate plots from sim_telarray file."""
    app_context = startup_application(_parse)

    plots = list(gen.ensure_iterable(app_context.args.get("plots")))
    generate_and_save_plots(plots=plots, args=app_context.args, ioh=app_context.io_handler)


if __name__ == "__main__":
    main()
