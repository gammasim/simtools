#!/usr/bin/python3

"""Compare simulation productions or summarize their resource requirements."""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.configuration.argument_helpers import telescope
from simtools.constants import SCHEMA_PATH
from simtools.data_model.metadata_collector import MetadataCollector
from simtools.production_configuration.production_comparison import write_production_comparison
from simtools.production_configuration.resource_requirements import write_resource_requirements
from simtools.sim_events.production_comparison import (
    collect_signal_metrics,
    parse_production_arguments,
)
from simtools.visualization import (
    plot_resource_requirements,
    plot_signal_level_production_comparison,
)

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "production",
        action="append",
        nargs="+",
        metavar=("LABEL", "INPUT_FILE_PATTERNS"),
        required=False,
        help=(
            "Production descriptor for event or signal comparison: --production <label> "
            "<comma-separated input file patterns>. Repeat for each production; the first "
            "production is the baseline."
        ),
    ),
    cli.ArgumentDefinition(
        "baseline_path",
        help="Production directory containing baseline metadata manifests.",
        type=str,
        required=False,
    ),
    cli.ArgumentDefinition(
        "candidate_path",
        help="Optional candidate production directory containing metadata manifests.",
        type=str,
        required=False,
    ),
    cli.ArgumentDefinition(
        "baseline_label",
        help="Display label for the baseline production in computing comparisons.",
        type=str,
        required=False,
    ),
    cli.ArgumentDefinition(
        "candidate_label",
        help="Display label for the candidate production in computing comparisons.",
        type=str,
        required=False,
    ),
    cli.ArgumentDefinition(
        "select",
        help="Selection expression as dotted.path=value. Can be repeated.",
        action="append",
        default=[],
    ),
    cli.ArgumentDefinition(
        "compare_by",
        help="Configuration field allowed to differ between baseline and candidate.",
        action="append",
        default=[],
    ),
    cli.ArgumentDefinition(
        "comparison_level",
        choices=["events", "signal", "computing"],
        default="events",
        help="Comparison level to execute.",
    ),
    cli.ArgumentDefinition(
        "array_layout_name",
        nargs="+",
        help=(
            "Restrict event-level comparison to selected array layout name(s), or signal-level "
            "comparison to selected telescope name(s)."
        ),
        required=False,
    ),
    cli.ArgumentDefinition(
        "telescope_name",
        nargs="+",
        type=telescope,
        help="Restrict signal-level comparison to the selected telescope name(s).",
        required=False,
    ),
)


def _post_parse(args_dict, _config_sources, parser):
    """Validate legacy and metadata-based production input modes."""
    if args_dict.get("comparison_level") == "computing":
        _validate_computing_arguments(args_dict, parser)
        return
    _validate_non_computing_arguments(args_dict, parser)


def _validate_computing_arguments(args_dict, parser):
    """Validate arguments specific to computing-resource comparisons."""
    if args_dict.get("production") or not args_dict.get("baseline_path"):
        parser.error(
            "Computing-resource comparison requires '--baseline_path' "
            "and does not use '--production'."
        )
    baseline_label = args_dict.get("baseline_label") or "baseline"
    candidate_label = args_dict.get("candidate_label") or "candidate"
    labels_are_explicit = args_dict.get("baseline_label") and args_dict.get("candidate_label")
    if (
        args_dict.get("candidate_path") or labels_are_explicit
    ) and baseline_label == candidate_label:
        parser.error("'--baseline_label' and '--candidate_label' must be different.")


def _validate_non_computing_arguments(args_dict, parser):
    """Validate arguments specific to event and signal comparisons."""
    if args_dict.get("baseline_label") or args_dict.get("candidate_label"):
        parser.error("Production labels can only be used with '--comparison_level computing'.")
    has_legacy_input = bool(args_dict.get("production"))
    has_metadata_input = bool(args_dict.get("baseline_path") or args_dict.get("candidate_path"))
    if has_legacy_input == has_metadata_input:
        parser.error("Use either '--production' or '--baseline_path' with '--candidate_path'.")
    if has_metadata_input and not (
        args_dict.get("baseline_path") and args_dict.get("candidate_path")
    ):
        parser.error("'--baseline_path' and '--candidate_path' must be used together.")
    if args_dict.get("comparison_level") == "signal" and not has_legacy_input:
        parser.error("Signal-level comparison requires '--production' sim_telarray inputs.")


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        *cli.OUTPUT_PATH_ARGUMENTS,
    ),
    initialize_output=False,
    initialize_model_reader=False,
    excluded_standard_arguments=("test", "ignore_existing_parameter_version"),
    post_parse=_post_parse,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()
    comparison_level = app_context.args["comparison_level"]
    if comparison_level == "events":
        write_production_comparison(
            app_context.args,
            app_context.io_handler.get_output_directory(),
        )
        return
    if comparison_level == "signal":
        output_files = _run_signal_comparison(app_context)
    elif comparison_level == "computing":
        write_resource_requirements(
            app_context.args,
            app_context.io_handler.get_output_directory(),
            plot_resource_requirements.plot,
        )
        return
    else:
        raise NotImplementedError(f"Comparison level '{comparison_level}' is not implemented yet.")

    for output_file, array_layout_name in output_files:
        _dump_comparison_metadata(app_context.args, output_file, array_layout_name)


def _run_signal_comparison(app_context):
    """Run signal-level comparison and return generated statistics files."""
    array_layout_names = app_context.args.get("array_layout_name")
    telescope_names = app_context.args.get("telescope_name")
    if array_layout_names and telescope_names:
        raise ValueError(
            "Use only one of --array_layout_name and --telescope_name for signal comparison."
        )
    production_descriptors = parse_production_arguments(app_context.args["production"])
    metrics_by_telescope = collect_signal_metrics(
        production_descriptors,
        telescope_names=array_layout_names or telescope_names,
    )
    return [
        (statistics_file, None)
        for statistics_file in plot_signal_level_production_comparison.plot(
            metrics_by_telescope,
            output_path=app_context.io_handler.get_output_directory(),
            figure_format=app_context.args.get("figure_format"),
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
