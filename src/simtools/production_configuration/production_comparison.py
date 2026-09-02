"""Compare trigger-histogram products from simulation productions."""

from pathlib import Path

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


def write_production_comparison(args_dict, output_directory):
    """Compare selected trigger-histogram productions and write their statistics.

    Parameters
    ----------
    args_dict : dict
        Application arguments, including either explicit production descriptors or metadata paths.
    output_directory : pathlib.Path
        Directory in which comparison products are written.
    """
    array_layout_names = args_dict.get("array_layout_name") or [None]
    if args_dict.get("baseline_path"):
        descriptor_pairs = _production_descriptor_pairs_from_metadata(args_dict)
        for pairing_key, production_descriptors in descriptor_pairs:
            pair_output_directory = (
                output_directory / f"comparison-{stable_configuration_hash(pairing_key)}"
            )
            _write_array_layout_comparisons(
                production_descriptors,
                args_dict,
                pair_output_directory,
                array_layout_names,
            )
        return

    production_descriptors = parse_production_arguments(args_dict["production"])
    _write_array_layout_comparisons(
        production_descriptors,
        args_dict,
        output_directory,
        array_layout_names,
    )


def _write_array_layout_comparisons(
    production_descriptors, args_dict, output_directory, array_layout_names
):
    """Write comparison products for all selected array layouts."""
    for array_layout_name in array_layout_names:
        _compare_array_layout(
            production_descriptors,
            args_dict,
            array_layout_name,
            output_directory,
        )


def _compare_array_layout(production_descriptors, args_dict, array_layout_name, output_directory):
    """Compare one selected array layout and write its statistics metadata."""
    metrics_per_production = collect_production_metrics(
        production_descriptors,
        array_names=array_layout_name,
    )
    comparison_statistics_file = plot_event_level_production_comparison.plot(
        metrics_per_production,
        output_path=output_directory,
        array_layout_name=array_layout_name,
        figure_format=args_dict.get("figure_format"),
    )
    metadata_args = dict(args_dict)
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
