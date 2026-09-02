#!/usr/bin/python3

"""Compare trigger-histogram products from simulation productions."""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.production_configuration.production_comparison import write_production_comparison

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "production",
        action="append",
        nargs="+",
        metavar=("LABEL", "TRIGGER_HISTOGRAM_PATTERNS"),
        required=False,
        help=(
            "Production descriptor: --production <label> <comma-separated file patterns>. "
            "Repeat for each production; the first production is the baseline."
        ),
    ),
    cli.ArgumentDefinition(
        "baseline_path",
        help="Directory containing baseline trigger-histogram metadata YAML files.",
        type=str,
        required=False,
    ),
    cli.ArgumentDefinition(
        "candidate_path",
        help="Directory containing candidate trigger-histogram metadata YAML files.",
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
        choices=["events", "signals", "compute"],
        default="events",
        help="Comparison level to execute.",
    ),
    cli.ArgumentDefinition(
        "array_layout_name",
        nargs="+",
        help="Restrict trigger histogram references to the selected array layout name(s).",
        required=False,
    ),
)


def _post_parse(args_dict, _config_sources, parser):
    """Validate legacy and metadata-based production input modes."""
    has_legacy_input = bool(args_dict.get("production"))
    has_metadata_input = bool(args_dict.get("baseline_path") or args_dict.get("candidate_path"))
    if has_legacy_input == has_metadata_input:
        parser.error("Use either '--production' or '--baseline_path' with '--candidate_path'.")
    if has_metadata_input and not (
        args_dict.get("baseline_path") and args_dict.get("candidate_path")
    ):
        parser.error("'--baseline_path' and '--candidate_path' must be used together.")


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        *cli.OUTPUT_PATH_ARGUMENTS,
    ),
    initialize_output=False,
    excluded_standard_arguments=("test", "ignore_existing_parameter_version"),
    post_parse=_post_parse,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()
    comparison_level = app_context.args["comparison_level"]
    if comparison_level != "events":
        raise NotImplementedError(f"Comparison level '{comparison_level}' is not implemented yet.")

    write_production_comparison(
        app_context.args,
        app_context.io_handler.get_output_directory(),
    )


if __name__ == "__main__":
    main()
