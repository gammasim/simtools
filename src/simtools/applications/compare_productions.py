#!/usr/bin/python3

"""Compare trigger-histogram products from simulation productions."""

from pathlib import Path

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.constants import SCHEMA_PATH
from simtools.data_model.metadata_collector import MetadataCollector
from simtools.production_configuration.production_file_selection import (
    check_manifest,
    discover_product_manifests,
    filter_manifests,
    normalize_for_comparison,
    stable_configuration_hash,
)
from simtools.sim_events.production_comparison import (
    ProductionDescriptor,
    collect_production_metrics,
    parse_production_arguments,
)
from simtools.visualization import plot_event_level_production_comparison

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

    array_layout_names = app_context.args.get("array_layout_name") or [None]
    if app_context.args.get("baseline_path"):
        descriptor_pairs = _production_descriptor_pairs_from_metadata(app_context.args)
        for pairing_key, production_descriptors in descriptor_pairs:
            pair_output_directory = (
                app_context.io_handler.get_output_directory()
                / f"comparison-{stable_configuration_hash(pairing_key)}"
            )
            for array_layout_name in array_layout_names:
                _compare_array_layout(
                    production_descriptors,
                    app_context,
                    array_layout_name,
                    output_directory=pair_output_directory,
                )
        return

    production_descriptors = parse_production_arguments(app_context.args["production"])
    for array_layout_name in array_layout_names:
        _compare_array_layout(production_descriptors, app_context, array_layout_name)


def _compare_array_layout(
    production_descriptors,
    app_context,
    array_layout_name,
    output_directory=None,
):
    """Compare one selected array layout and write its statistics metadata."""
    metrics_per_production = collect_production_metrics(
        production_descriptors,
        array_names=array_layout_name,
    )
    comparison_statistics_file = plot_event_level_production_comparison.plot(
        metrics_per_production,
        output_path=output_directory or app_context.io_handler.get_output_directory(),
        array_layout_name=array_layout_name,
        figure_format=app_context.args.get("figure_format"),
    )
    metadata_args = dict(app_context.args)
    metadata_args.update(
        {
            "array_layout_name": array_layout_name,
            "output_file": str(comparison_statistics_file),
            "output_file_format": "JSON",
            "metadata_product_data_name": "production_comparison_statistics",
            "schema_file": str(SCHEMA_PATH / "production_comparison_statistics.schema.yml"),
        }
    )
    MetadataCollector.dump(metadata_args, comparison_statistics_file)


def _production_descriptor_pairs_from_metadata(args_dict):
    """Build one descriptor pair per matched trigger-histogram configuration."""
    baseline = _selected_trigger_histogram_manifests(
        args_dict["baseline_path"], args_dict.get("select")
    )
    candidate = _selected_trigger_histogram_manifests(
        args_dict["candidate_path"], args_dict.get("select")
    )
    compare_by = {field.removeprefix("configuration.") for field in args_dict.get("compare_by", [])}
    baseline_by_key = _unique_manifests_by_pairing_key(baseline, compare_by, "baseline")
    candidate_by_key = _unique_manifests_by_pairing_key(candidate, compare_by, "candidate")

    missing_candidates = sorted(set(baseline_by_key) - set(candidate_by_key))
    missing_baselines = sorted(set(candidate_by_key) - set(baseline_by_key))
    if missing_candidates or missing_baselines:
        raise ValueError(
            "Trigger-histogram metadata pairing failed: "
            f"missing candidates={len(missing_candidates)}, "
            f"missing baselines={len(missing_baselines)}."
        )

    return [
        (
            key,
            [
                ProductionDescriptor(
                    label="baseline",
                    trigger_histogram_files=[
                        str(_single_trigger_histogram_file(baseline_by_key[key]))
                    ],
                ),
                ProductionDescriptor(
                    label="candidate",
                    trigger_histogram_files=[
                        str(_single_trigger_histogram_file(candidate_by_key[key]))
                    ],
                ),
            ],
        )
        for key in sorted(baseline_by_key, key=str)
    ]


def _selected_trigger_histogram_manifests(path, selections):
    """Return checked trigger-histogram manifests matching selections."""
    manifests = discover_product_manifests(path, "trigger_histograms")
    selected = filter_manifests(manifests, selections or [])
    if not selected:
        raise ValueError(f"No trigger-histogram metadata files matched in {path}.")
    for manifest in selected:
        check_manifest(manifest)
    return selected


def _unique_manifests_by_pairing_key(manifests, compare_by, label):
    """Return manifests keyed by comparable simulation configuration."""
    keyed = {}
    for manifest in manifests:
        key = _pairing_key(manifest.data, compare_by)
        if key in keyed:
            raise ValueError(
                f"More than one {label} trigger-histogram file matches configuration {key}."
            )
        keyed[key] = manifest
    return keyed


def _pairing_key(manifest, compare_by):
    """Return normalized configuration fields used for baseline/candidate pairing."""
    configuration = manifest.get("configuration", {})
    paired_configuration = {
        key: value for key, value in configuration.items() if key not in compare_by
    }
    return normalize_for_comparison(
        {
            "configuration": paired_configuration,
            "histogram_settings": manifest.get("histogram_settings", {}),
            "array_selection": manifest.get("array_selection", []),
        }
    )


def _single_trigger_histogram_file(manifest):
    """Return the single trigger-histogram file referenced by a manifest."""
    files = manifest.data.get("files", {}).get("trigger_histograms", [])
    if len(files) != 1:
        raise ValueError(
            f"Expected exactly one trigger-histogram file in {manifest.path}, found {len(files)}."
        )
    path = Path(files[0])
    if path.is_absolute():
        return path
    return manifest.directory / path


if __name__ == "__main__":
    main()
