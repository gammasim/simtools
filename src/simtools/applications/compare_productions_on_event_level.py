#!/usr/bin/python3

r"""Compare multiple simulation productions on event level.

The application accepts repeated production descriptors and creates event-level
comparison plots using reduced event data files.

Command line arguments
----------------------
production (repeated, required)
    Production descriptor in two fields:
    1) label
    2) comma-separated event data file patterns
output_path (str, required)
    Output directory for generated comparison plots.

Examples
--------
Compare two productions:

.. code-block:: console

    simtools-compare-productions-on-event-level \
        --production baseline "data/baseline/*.h5" \
        --production candidate "data/candidate/*.h5,data/candidate_extra/*.h5" \
        --output_path simtools-output/
"""

from simtools.application_control import build_application
from simtools.sim_events.production_comparison import (
    collect_production_metrics,
    parse_production_arguments,
)
from simtools.visualization import plot_event_level_production_comparison


def _add_arguments(parser):
    """Register application-specific command line arguments."""
    parser.initialize_application_arguments(["output_path"])
    parser.add_argument(
        "--production",
        action="append",
        nargs="+",
        metavar=("LABEL", "EVENT_DATA_FILES"),
        required=True,
        help=(
            "Production descriptor. "
            "Use as: --production <label> <comma-separated file patterns>. "
            "Repeat this argument at least twice."
        ),
    )
    parser.add_argument(
        "--telescope_ids",
        nargs="+",
        default=None,
        help="Optional telescope IDs to filter triggers.",
    )


def main():
    """See CLI description."""
    app_context = build_application(
        initialization_kwargs={"db_config": False, "output": True},
    )

    production_descriptors = parse_production_arguments(app_context.args["production"])
    metrics_per_production = collect_production_metrics(
        production_descriptors,
        telescope_list=app_context.args["telescope_ids"],
    )

    output_directory = app_context.io_handler.get_output_directory()
    plot_event_level_production_comparison.plot(
        metrics_per_production, output_path=output_directory
    )


if __name__ == "__main__":
    main()
