#!/usr/bin/python3

r"""Compare simulation productions at different comparison levels.

The application accepts repeated production descriptors and dispatches to
comparison-level specific implementations.

Command line arguments
----------------------
production (repeated, required)
    Production descriptor in two fields:
    1) label
    2) comma-separated event data file patterns
comparison_level (str, optional)
    Comparison level selector. Supported values are:
    - events
    - signals
    - compute
output_path (str, required)
    Output directory for generated comparison plots.
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
            "Repeat this argument as needed for multiple productions."
        ),
    )
    parser.add_argument(
        "--comparison_level",
        choices=["events", "signals", "compute"],
        default="events",
        help="Comparison level to execute.",
    )


def main():
    """See CLI description."""
    app_context = build_application(
        initialization_kwargs={"db_config": False, "output": True},
    )

    output_directory = app_context.io_handler.get_output_directory()
    production_descriptors = parse_production_arguments(app_context.args["production"])

    comparison_level = app_context.args["comparison_level"]
    if comparison_level == "events":
        plot_event_level_production_comparison.plot(
            collect_production_metrics(production_descriptors),
            output_path=output_directory,
        )
    else:
        raise NotImplementedError(f"Comparison level '{comparison_level}' is not implemented yet.")


if __name__ == "__main__":
    main()
