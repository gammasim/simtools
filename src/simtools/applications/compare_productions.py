#!/usr/bin/python3

"""Compare trigger-histogram products from simulation productions."""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.constants import SCHEMA_PATH
from simtools.data_model.metadata_collector import MetadataCollector
from simtools.sim_events.production_comparison import (
    collect_production_metrics,
    collect_signal_metrics,
    parse_production_arguments,
)
from simtools.visualization import (
    plot_event_level_production_comparison,
    plot_signal_level_production_comparison,
)

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "production",
        action="append",
        nargs="+",
        metavar=("LABEL", "TRIGGER_HISTOGRAM_PATTERNS"),
        required=True,
        help=(
            "Production descriptor: --production <label> <comma-separated file patterns>. "
            "Repeat for each production; the first production is the baseline."
        ),
    ),
    cli.ArgumentDefinition(
        "comparison_level",
        choices=["events", "signal", "compute"],
        default="events",
        help="Comparison level to execute.",
    ),
    cli.ArgumentDefinition(
        "array_layout_name",
        nargs="+",
        help="Array layout filter, or the single layout for signal comparison.",
        required=False,
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        *cli.OUTPUT_PATH_ARGUMENTS,
    ),
    initialize_output=False,
    excluded_standard_arguments=("test", "ignore_existing_parameter_version"),
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()
    comparison_level = app_context.args["comparison_level"]
    if comparison_level == "events":
        output_files = _run_event_comparison(app_context)
    elif comparison_level == "signal":
        output_files = _run_signal_comparison(app_context)
    else:
        raise NotImplementedError(f"Comparison level '{comparison_level}' is not implemented yet.")

    for output_file, array_layout_name in output_files:
        _dump_comparison_metadata(app_context.args, output_file, array_layout_name)


def _run_event_comparison(app_context):
    """Run event-level comparison and return generated statistics files."""
    production_descriptors = parse_production_arguments(app_context.args["production"])
    array_layout_names = app_context.args.get("array_layout_name") or [None]
    output_files = []
    for array_layout_name in array_layout_names:
        output_files.append(
            (
                _compare_array_layout(production_descriptors, app_context, array_layout_name),
                array_layout_name,
            )
        )
    return output_files


def _compare_array_layout(production_descriptors, app_context, array_layout_name):
    """Compare one selected array layout and return its statistics file."""
    metrics_per_production = collect_production_metrics(
        production_descriptors,
        array_names=array_layout_name,
    )
    return plot_event_level_production_comparison.plot(
        metrics_per_production,
        output_path=app_context.io_handler.get_output_directory(),
        array_layout_name=array_layout_name,
        figure_format=app_context.args.get("figure_format"),
    )


def _run_signal_comparison(app_context):
    """Run signal-level comparison and return generated statistics files."""
    production_descriptors = parse_production_arguments(app_context.args["production"])
    metrics_by_telescope = collect_signal_metrics(
        production_descriptors,
        array_layout_name=app_context.args.get("array_layout_name"),
    )
    return [
        (statistics_file, None)
        for statistics_file in plot_signal_level_production_comparison.plot(
            metrics_by_telescope,
            output_path=app_context.io_handler.get_output_directory(),
        )
    ]


def _dump_comparison_metadata(args, output_file, array_layout_name=None):
    """Write comparison metadata for one generated statistics file."""
    metadata_args = dict(args)
    if array_layout_name is not None:
        metadata_args["array_layout_name"] = array_layout_name
    metadata_args.update(
        {
            "output_file": str(output_file),
            "output_file_format": "JSON",
            "metadata_product_data_name": "production_comparison_statistics",
            "schema_file": str(SCHEMA_PATH / "production_comparison_statistics.schema.yml"),
        }
    )
    MetadataCollector.dump(metadata_args, output_file)


if __name__ == "__main__":
    main()
