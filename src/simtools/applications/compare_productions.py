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
    parser.add_argument(
        "--telescope_ids",
        nargs="+",
        default=None,
        help="Optional telescope IDs to filter triggers.",
    )


def _run_event_comparison(app_context):
    """Run event-level production comparison."""
    production_descriptors = parse_production_arguments(app_context.args["production"])
    metrics_per_production = collect_production_metrics(
        production_descriptors,
        telescope_list=app_context.args["telescope_ids"],
    )

    output_directory = app_context.io_handler.get_output_directory()
    plot_event_level_production_comparison.plot(
        metrics_per_production,
        output_path=output_directory,
    )


def _run_placeholder_comparison(comparison_level):
    """Warn for future comparison-level implementations.

    Notes
    -----
    Future work tracked in:
    - signals level: issue #2183
    - compute level: issue #2184
    """
    raise NotImplementedError(
        f"Comparison level '{comparison_level}' is not implemented yet "
        "(planned via issues #2183 and #2184)."
    )


def main():
    """See CLI description."""
    app_context = build_application(
        initialization_kwargs={"db_config": False, "output": True},
    )

    comparison_level = app_context.args["comparison_level"]
    if comparison_level == "events":
        _run_event_comparison(app_context)
        return

    _run_placeholder_comparison(comparison_level)


if __name__ == "__main__":
    main()
