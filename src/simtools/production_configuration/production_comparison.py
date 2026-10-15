"""Compare trigger-histogram products from simulation productions."""

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np

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
from simtools.visualization.visualize import save_figures_to_single_document

COMPARISON_CONTEXT_KEY = "comparison_context"

ARRAY_LAYOUT_KEY = "array_layout_name"

_logger = logging.getLogger(__name__)


class _ProductionPairingError(ValueError):
    """Report unmatched metadata while retaining the pairs that can be compared."""

    def __init__(self, message, descriptor_pairs):
        super().__init__(message)
        self.descriptor_pairs = descriptor_pairs


def write_production_comparison(args_dict, output_directory):
    """Compare selected trigger-histogram productions and write their statistics.

    Parameters
    ----------
    args_dict : dict
        Application arguments, including either explicit production descriptors or metadata paths.
    output_directory : pathlib.Path
        Directory in which comparison products are written.
    """
    array_layout_names = args_dict.get(ARRAY_LAYOUT_KEY) or [None]
    if args_dict.get("baseline_path"):
        _write_metadata_comparisons(args_dict, output_directory, array_layout_names)
        return

    production_descriptors = parse_production_arguments(args_dict["production"])
    statistics_files = _write_array_layout_comparisons(
        production_descriptors,
        args_dict,
        output_directory,
        array_layout_names,
    )
    _write_comparison_summary(
        statistics_files, output_directory, args_dict.get("report_mode", "diagnostic")
    )


def _write_metadata_comparisons(args_dict, output_directory, array_layout_names):
    """Write comparisons for trigger-histogram metadata pairs."""
    pairing_error = None
    try:
        descriptor_pairs = _production_descriptor_pairs_from_metadata(args_dict)
    except _ProductionPairingError as exc:
        if not exc.descriptor_pairs:
            raise
        descriptor_pairs = exc.descriptor_pairs
        pairing_error = exc
    output_stems = [
        _comparison_pair_output_stem(production_descriptors)
        for _, production_descriptors in descriptor_pairs
    ]
    stem_counts = Counter(output_stems)
    statistics_files = []
    compact_figures = [] if args_dict.get("report_mode") == "compact" else None
    for (pairing_key, production_descriptors), output_stem in zip(descriptor_pairs, output_stems):
        pair_output_directory = _metadata_pair_output_directory(
            output_directory,
            pairing_key,
            output_stem,
            stem_counts,
        )
        comparison_kwargs = {}
        if "configurations" in pairing_key:
            comparison_kwargs[COMPARISON_CONTEXT_KEY] = pairing_key
        if compact_figures is None:
            pair_statistics = _write_array_layout_comparisons(
                production_descriptors,
                args_dict,
                pair_output_directory,
                array_layout_names,
                **comparison_kwargs,
            )
        else:
            pair_statistics = _write_array_layout_comparisons(
                production_descriptors,
                args_dict,
                pair_output_directory,
                array_layout_names,
                compact_figures=compact_figures,
                **comparison_kwargs,
            )
        statistics_files.extend(pair_statistics or [])
    if compact_figures:
        save_figures_to_single_document(
            compact_figures, output_directory / "trigger-report.pdf", close=True
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "comparison_inventory.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "matched_pair_count": len(descriptor_pairs),
                "pairing_error_count": int(pairing_error is not None),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if pairing_error is not None:
        _logger.warning(str(pairing_error))
    _write_comparison_summary(
        statistics_files, output_directory, args_dict.get("report_mode", "diagnostic")
    )


def _metadata_pair_output_directory(output_directory, pairing_key, output_stem, stem_counts):
    """Return the output directory for one metadata comparison pair."""
    if stem_counts[output_stem] > 1:
        hash_key = pairing_key.get("pairing_key", pairing_key)
        output_stem = f"{output_stem}-{stable_configuration_hash(hash_key)}"
    return output_directory / output_stem


def _write_array_layout_comparisons(
    production_descriptors,
    args_dict,
    output_directory,
    array_layout_names,
    compact_figures=None,
    comparison_context=None,
):
    """Write comparison products for all selected array layouts."""
    statistics_files = []
    for array_layout_name in array_layout_names:
        statistics_files.append(
            _compare_array_layout(
                production_descriptors,
                args_dict,
                array_layout_name,
                output_directory,
                compact_figures,
                comparison_context,
            )
        )
    return statistics_files


def _compare_array_layout(
    production_descriptors,
    args_dict,
    array_layout_name,
    output_directory,
    compact_figures=None,
    comparison_context=None,
):
    """Compare one selected array layout and write its statistics file."""
    metrics_per_production = collect_production_metrics(
        production_descriptors,
        array_names=array_layout_name,
    )
    plot_args = {
        "output_path": output_directory,
        ARRAY_LAYOUT_KEY: array_layout_name,
        "figure_format": args_dict.get("figure_format"),
        "report_mode": args_dict.get("report_mode", "diagnostic"),
        "comparison_title": _comparison_title(comparison_context, array_layout_name),
    }
    if compact_figures is not None:
        plot_args["compact_figures"] = compact_figures
    if comparison_context is not None:
        plot_args[COMPARISON_CONTEXT_KEY] = comparison_context
    return plot_event_level_production_comparison.plot(
        metrics_per_production,
        **plot_args,
    )


def _comparison_title(comparison_context, array_layout_name):
    """Format pair configuration values for plot headings."""
    if not isinstance(comparison_context, dict) or "configurations" not in comparison_context:
        return f"Layout: {array_layout_name}" if array_layout_name else None

    configurations = comparison_context["configurations"]
    labels = list(configurations)
    baseline = configurations[labels[0]]
    candidate = configurations[labels[-1]]
    identity_fields = []
    for key, label in (("primary", "Primary"), ("site", "Site")):
        value = _shared_configuration_value(baseline, candidate, key)
        if value is not None:
            identity_fields.append(f"{label}: {value}")

    energy = _format_energy_range(baseline, candidate, labels)
    geometry = _format_pair_value("zenith_angle", "Zenith", baseline, candidate, labels, unit="deg")
    azimuth = _format_pair_value(
        "azimuth_angle", "Azimuth", baseline, candidate, labels, unit="deg"
    )
    interaction = _shared_configuration_value(baseline, candidate, "corsika_he_interaction")
    first_line = [f"Layout: {array_layout_name}" if array_layout_name else None]
    first_line.extend(identity_fields)
    first_line.extend(value for value in (energy, geometry) if value)
    second_line = [value for value in (azimuth,) if value]
    if interaction is not None:
        second_line.append(f"Interaction model: {interaction}")
    first_line_text = " | ".join(value for value in first_line if value)
    return first_line_text + "\n" + " | ".join(second_line)


def _format_energy_range(baseline, candidate, labels):
    """Format matching energy limits as one readable energy range."""
    baseline_min = _format_configuration_value(baseline.get("energy_min"), "energy")
    baseline_max = _format_configuration_value(baseline.get("energy_max"), "energy")
    candidate_min = _format_configuration_value(candidate.get("energy_min"), "energy")
    candidate_max = _format_configuration_value(candidate.get("energy_max"), "energy")
    if baseline_min == "None" and baseline_max == "None":
        return None
    baseline_range = _join_energy_limits(baseline_min, baseline_max)
    candidate_range = _join_energy_limits(candidate_min, candidate_max)
    if baseline_range == candidate_range:
        return f"Energy: {baseline_range}"
    return f"Energy: {labels[0]} {baseline_range}; {labels[-1]} {candidate_range}"


def _join_energy_limits(minimum, maximum):
    """Join minimum and maximum energies without repeating identical limits."""
    if minimum == maximum:
        return minimum
    return f"{minimum} to {maximum}"


def _format_pair_value(key, label, baseline, candidate, labels, unit=None):
    """Format one configuration value, labeling both sides when they differ."""
    baseline_value = _format_configuration_value(baseline.get(key), unit)
    candidate_value = _format_configuration_value(candidate.get(key), unit)
    if baseline_value == "None" and candidate_value == "None":
        return None
    if baseline_value == candidate_value:
        return f"{label}: {baseline_value}"
    return f"{label}: {labels[0]} {baseline_value} -> {labels[-1]} {candidate_value}"


def _shared_configuration_value(baseline, candidate, key):
    """Return a compact value only when both pair configurations agree."""
    baseline_value = _format_configuration_value(baseline.get(key))
    candidate_value = _format_configuration_value(candidate.get(key))
    if baseline_value == candidate_value and baseline_value != "None":
        return baseline_value
    return None


def _format_configuration_value(value, quantity=None):
    """Format manifest values, including quantities, for a compact plot heading."""
    if isinstance(value, dict) and set(value) == {"value", "unit"}:
        magnitude = value["value"]
        value_unit = value["unit"]
        if quantity == "energy" and str(value_unit).lower() == "gev" and float(magnitude) >= 1000:
            return f"{float(magnitude) / 1000:g} TeV"
        return (
            f"{magnitude:g} {value_unit}"
            if isinstance(magnitude, int | float)
            else f"{magnitude} {value_unit}"
        )
    if quantity and isinstance(value, int | float):
        return f"{value:g} {quantity}"
    return str(value)


def _write_comparison_summary(statistics_files, output_directory, report_mode="diagnostic"):
    """Combine per-pair statistics into one summary and remove the individual files."""
    comparisons = []
    baseline = None
    comparison_sets = {}
    plot_statistics = {}
    compact_plot_statistics = {}
    compact_report = report_mode == "compact"
    for statistics_file in statistics_files:
        statistics = json.loads(statistics_file.read_text(encoding="utf-8"))
        comparison, pair_name, array_layout = _comparison_summary_entry(
            statistics_file, output_directory, statistics, compact_report
        )
        comparisons.append(comparison)
        baseline = _add_production_summary(baseline, statistics["baseline"])
        comparison_sets = _add_comparison_sets(comparison_sets, statistics["comparison_sets"])
        _collect_plot_statistics(
            statistics["plot_statistics"],
            pair_name,
            array_layout,
            compact_report,
            plot_statistics,
            compact_plot_statistics,
        )
        statistics_file.unlink()
        _remove_empty_comparison_directories(statistics_file.parent, output_directory)

    if compact_report:
        plot_statistics = _finalize_compact_plot_statistics(compact_plot_statistics)
    summary = {
        "format_version": 2,
        "baseline": baseline
        or {
            "label": None,
            "simulated_event_count": 0,
            "triggered_event_count": 0,
        },
        "comparison_sets": list(comparison_sets.values()),
        "plot_statistics": plot_statistics,
        "comparisons": comparisons,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "comparison_statistics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _remove_empty_comparison_directories(directory, output_directory):
    """Remove emptied per-pair directories without touching plots or the report root."""
    while directory != output_directory and directory.is_relative_to(output_directory):
        try:
            directory.rmdir()
        except OSError:
            break
        directory = directory.parent


def _comparison_summary_entry(statistics_file, output_directory, statistics, compact_report):
    """Build one comparison entry, omitting duplicated plot data in compact mode."""
    path_parts = statistics_file.relative_to(output_directory).parts
    pair_name = path_parts[0] if len(path_parts) > 2 else ""
    array_layout = path_parts[1] if len(path_parts) > 2 else path_parts[0]
    if array_layout == statistics_file.name:
        array_layout = ""
    pair_statistics = statistics
    if compact_report:
        pair_statistics = {
            "format_version": statistics.get("format_version", 1),
            "baseline": statistics["baseline"],
            "comparison_sets": statistics["comparison_sets"],
            "plot_statistics": {
                name: _scalar_plot_statistics(value)
                for name, value in statistics["plot_statistics"].items()
            },
        }
    return (
        {
            "pair": pair_name,
            "array_layout": array_layout,
            "statistics": pair_statistics,
        },
        pair_name,
        array_layout,
    )


def _add_comparison_sets(comparison_sets, productions):
    """Merge event counts for each candidate production label."""
    for production in productions:
        label = production["label"]
        comparison_sets[label] = _add_production_summary(comparison_sets.get(label), production)
    return comparison_sets


def _collect_plot_statistics(
    statistics, pair_name, array_layout, compact_report, plot_statistics, compact_statistics
):
    """Collect full per-pair statistics or compact per-plot metric summaries."""
    for name, values in statistics.items():
        if compact_report:
            plot_name = values.get("plot_name", name)
            _add_compact_plot_statistics(compact_statistics, plot_name, values)
        else:
            plot_statistics[f"{pair_name}/{array_layout}/{name}"] = values


def _add_compact_plot_statistics(compact_statistics, plot_name, statistics):
    """Accumulate plot metric results without retaining histogram arrays."""
    entry = compact_statistics.setdefault(
        plot_name,
        {
            "baseline_label": statistics.get("baseline_label"),
            "metric_type": statistics["metric_type"],
            "comparisons": [],
            "metadata": {"compact": True, "pair_count": 0, "results": {}},
        },
    )
    for key in ("metric", "quantity_name", "cumulative"):
        if key in statistics:
            entry[key] = statistics[key]
    entry["metadata"]["pair_count"] += 1
    for comparison in statistics.get("comparisons", []):
        candidate_label = comparison.get("candidate_label", "candidate")
        result_groups = {
            key: comparison[key] for key in ("simulated", "triggered") if key in comparison
        }
        if not result_groups:
            result_groups = {"result": comparison}
        for group_name, result in result_groups.items():
            result_key = f"{candidate_label}:{group_name}"
            _record_compact_metric_result(entry["metadata"]["results"], result_key, result)


def _record_compact_metric_result(results, group_name, result):
    """Accumulate counts and scalar metric values for one distribution result."""
    summary = results.setdefault(
        group_name,
        {"result_count": 0, "valid_count": 0, "invalid_reasons": {}, "values": []},
    )
    summary["result_count"] += 1
    if result.get("valid"):
        summary["valid_count"] += 1
        value = result.get("value")
        if isinstance(value, int | float):
            summary["values"].append(value)
    else:
        reason = result.get("reason", "unspecified")
        summary["invalid_reasons"][reason] = summary["invalid_reasons"].get(reason, 0) + 1


def _finalize_compact_plot_statistics(compact_statistics):
    """Return small per-plot metric summaries for the release report."""
    for entry in compact_statistics.values():
        results = entry["metadata"]["results"]
        for summary in results.values():
            values = summary.pop("values")
            summary["invalid_count"] = summary["result_count"] - summary["valid_count"]
            if values:
                summary["minimum"] = min(values)
                summary["median"] = float(np.median(values))
                summary["maximum"] = max(values)
            else:
                summary["minimum"] = None
                summary["median"] = None
                summary["maximum"] = None
    return compact_statistics


def _add_production_summary(total, production):
    """Add event counts from one production summary."""
    if total is None:
        return dict(production)
    total["simulated_event_count"] += production["simulated_event_count"]
    total["triggered_event_count"] += production["triggered_event_count"]
    return total


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
    descriptor_pairs = _matched_descriptor_pairs(
        baseline_by_key,
        candidate_by_key,
        args_dict.get("baseline_label", "baseline"),
        args_dict.get("candidate_label", "candidate"),
    )
    if missing_candidates or missing_baselines:
        raise _ProductionPairingError(
            "Trigger-histogram metadata pairing failed: "
            f"missing candidates={len(missing_candidates)}, "
            f"missing baselines={len(missing_baselines)}.",
            descriptor_pairs,
        )

    return descriptor_pairs


def _matched_descriptor_pairs(baseline_by_key, candidate_by_key, baseline_label, candidate_label):
    """Build descriptor pairs for configurations present in both productions."""
    return [
        (
            {
                "pairing_key": key,
                "configurations": {
                    baseline_label: baseline_by_key[key].data.get("configuration", {}),
                    candidate_label: candidate_by_key[key].data.get("configuration", {}),
                },
            },
            [
                ProductionDescriptor(
                    label=baseline_label,
                    input_files=[str(_single_trigger_histogram_file(baseline_by_key[key]))],
                ),
                ProductionDescriptor(
                    label=candidate_label,
                    input_files=[str(_single_trigger_histogram_file(candidate_by_key[key]))],
                ),
            ],
        )
        for key in sorted(set(baseline_by_key) & set(candidate_by_key), key=str)
    ]


def _comparison_pair_output_stem(production_descriptors):
    """Return a readable directory stem from the paired histogram filenames."""
    input_stems = [Path(descriptor.input_files[0]).stem for descriptor in production_descriptors]
    if len(set(input_stems)) == 1:
        return input_stems[0]
    return "-vs-".join(input_stems)


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
    layout = paired_configuration.get(ARRAY_LAYOUT_KEY)
    if isinstance(layout, (list, tuple)) and len(layout) == 1:
        paired_configuration[ARRAY_LAYOUT_KEY] = layout[0]
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


def _scalar_plot_statistics(statistics):
    """Retain metric results and validity without distribution arrays."""
    summary = {
        key: value
        for key, value in statistics.items()
        if key
        in {"baseline_label", "metric_type", "metric", "quantity_name", "cumulative", "rendering"}
    }
    summary["comparisons"] = [
        _scalar_metric_result(value) for value in statistics.get("comparisons", [])
    ]
    return summary


def _scalar_metric_result(result):
    """Keep small scalar results, including nested simulated/triggered metrics."""
    summary = {}
    for key, value in result.items():
        if key in {"baseline", "candidate", "metadata"}:
            continue
        if isinstance(value, dict):
            summary[key] = _scalar_metric_result(value)
        elif isinstance(value, (str, int, float, bool, type(None))):
            summary[key] = value
    return summary
