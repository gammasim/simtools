#!/usr/bin/python3

r"""Compare simulation productions at different comparison levels.

The application accepts repeated production descriptors and dispatches to
comparison-level specific implementations.

Command line arguments
----------------------
production (repeated, required)
    Production descriptor in two fields:
    1) label
    2) comma-separated trigger histogram HDF5 file patterns
comparison_level (str, optional)
    Comparison level selector. Supported values are:
    - events
    - signals
    - compute
output_path (str, required)
    Output directory for generated comparison plots.
array_layout_name (str, optional)
    Restrict comparison inputs to one or more array layout names stored in the
    trigger histogram HDF5 metadata.
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
        metavar=("LABEL", "TRIGGER_HISTOGRAM_FILES"),
        required=True,
        help=(
            "Production descriptor. "
            "Use as: --production <label> <comma-separated file patterns>. "
            "Repeat this argument as needed for multiple trigger histogram files."
        ),
    )
    parser.add_argument(
        "--comparison_level",
        choices=["events", "signals", "compute"],
        default="events",
        help="Comparison level to execute.",
    )
    parser.add_argument(
        "--array_layout_name",
        nargs="+",
        help="Restrict trigger histogram references to the selected array layout name(s).",
        required=False,
    )


def main():
    """See CLI description."""
    app_context = build_application(
        initialization_kwargs={"db_config": False, "output": True},
    )

    comparison_level = app_context.args["comparison_level"]
    if comparison_level == "events":
        metrics_per_production = collect_production_metrics(
            parse_production_arguments(app_context.args["production"]),
            array_names=app_context.args.get("array_layout_name"),
        )
        plot_event_level_production_comparison.plot(
            metrics_per_production,
            output_path=app_context.io_handler.get_output_directory(),
            array_layout_name=app_context.args.get("array_layout_name"),
        )
    else:
        raise NotImplementedError(f"Comparison level '{comparison_level}' is not implemented yet.")


if __name__ == "__main__":
    main()
