#!/usr/bin/python3

"""Plot simulated events from sim_telarray files."""

import simtools.utils.general as gen
from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.visualization.plot_simtel_events import PLOT_CHOICES, generate_and_save_plots

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "simtel_file", help="Input sim_telarray file (.simtel.zst)", required=True
    ),
    cli.ArgumentDefinition(
        "plots",
        help=f"Plots to generate. Choices: {', '.join(sorted(PLOT_CHOICES))}",
        nargs="+",
        default=["all"],
        choices=sorted(PLOT_CHOICES),
    ),
    cli.ArgumentDefinition(
        "number_of_pixels",
        type=int,
        default=3,
        help="For time_traces: number of pixel traces",
    ),
    cli.ArgumentDefinition(
        "pixel_step", type=int, default=10, help="Step between pixel ids for step plots"
    ),
    cli.ArgumentDefinition(
        "max_pixels", type=int, default=None, help="Cap number of pixels for step traces"
    ),
    cli.ArgumentDefinition("vmax", type=float, default=None, help="Color scale vmax"),
    cli.ArgumentDefinition(
        "half_width", type=int, default=8, help="Half window width for integrated images"
    ),
    cli.ArgumentDefinition(
        "offset",
        type=int,
        default=16,
        help="offset between pedestal and peak windows (integrated_pedestal_image)",
    ),
    cli.ArgumentDefinition(
        "sum_threshold",
        type=float,
        default=10.0,
        help="Minimum pixel sum to consider in peak timing",
    ),
    cli.ArgumentDefinition(
        "peak_width", type=int, default=8, help="Expected peak width in samples"
    ),
    cli.ArgumentDefinition(
        "examples", type=int, default=3, help="Number of example traces to draw"
    ),
    cli.ArgumentDefinition(
        "timing_bins",
        type=int,
        default=None,
        help="Number of bins for timing histogram (contiguous if not set)",
    ),
    cli.ArgumentDefinition(
        "distance",
        type=float,
        default=None,
        help="Optional distance annotation for event_image (same units as input expects)",
    ),
    cli.ArgumentDefinition(
        "event_id",
        type=int,
        nargs="+",
        default=None,
        help="Event ID(s) of the events to be plotted",
    ),
    cli.ArgumentDefinition(
        "max_events", type=int, default=1, help="Maximum number of events to process"
    ),
    cli.ArgumentDefinition(
        "save_pngs", action="store_true", help="Also save individual PNG images per plot"
    ),
    cli.ArgumentDefinition("dpi", type=int, default=300, help="PNG dpi"),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        cli.OVERWRITE_MODEL_PARAMETERS,
        cli.SITE,
        cli.TELESCOPE,
        *cli.OUTPUT_PATH_ARGUMENTS,
        *cli.OUTPUT_ARGUMENTS,
    ),
    initialize_output=True,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

    plots = gen.ensure_list(app_context.args.get("plots"))
    generate_and_save_plots(plots=plots, args=app_context.args, ioh=app_context.io_handler)


if __name__ == "__main__":
    main()
