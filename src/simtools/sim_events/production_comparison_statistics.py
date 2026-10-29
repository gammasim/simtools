"""Compute statistics for comparisons of simulation productions."""

import logging
from collections import Counter

import numpy as np

from simtools.io import ascii_handler
from simtools.io.file_type import product_filename
from simtools.statistics import compare_histogram_counts, compare_samples_with_statistics
from simtools.utils import names

_JENSEN_SHANNON_DISTANCE = "jensen_shannon_distance"
_VALID = "valid"
_WASSERSTEIN_DISTANCE = "wasserstein_distance"

BIN_EDGES_KEY = "bin_edges"
CATEGORY_COUNT_KEY = "category_count"
JENSEN_SHANNON_METRIC = "jensen_shannon"
PLOT_NAME_KEY = "plot_name"
REASON_KEY = "reason"
SIMULATED_POPULATION = "simulated"
TELESCOPE_TYPES_KEY = "telescope_types"
TRIGGERED_POPULATION = "triggered"
WASSERSTEIN_METRIC = "wasserstein"

PLOT_STATISTICS_KEY = "plot_statistics"

ENERGY = "energy"
ANGULAR_DISTANCE = "angular_distance"
CORE_DISTANCE = "core_distance"
COMPACT_REPORT = "compact"
DIAGNOSTIC_REPORT = "diagnostic"
QUANTITY_NAME_KEY = "quantity_name"
CUMULATIVE_KEY = "cumulative"
VALUE_KEY = "value"
_COMPACT_STATISTIC_FIELDS = (
    "metric",
    "value",
    "normalized_value",
    _VALID,
    REASON_KEY,
    "pvalue",
    "ks_statistic",
    "ks_pvalue",
    _JENSEN_SHANNON_DISTANCE,
    _WASSERSTEIN_DISTANCE,
    "baseline_total",
    "candidate_total",
)

_logger = logging.getLogger(__name__)


def compact_plot_statistics(plot_statistics):
    """Keep JSON scalar-only so diagnostic histogram arrays do not create huge files."""
    compact = {}
    for plot_name, statistics in plot_statistics.items():
        item = {
            key: statistics[key]
            for key in (
                PLOT_NAME_KEY,
                "baseline_label",
                "metric_type",
                TELESCOPE_TYPES_KEY,
                "metric",
                QUANTITY_NAME_KEY,
                CUMULATIVE_KEY,
            )
            if key in statistics
        }
        item["comparisons"] = [
            _compact_comparison_statistics(comparison)
            for comparison in statistics.get("comparisons", [])
        ]
        compact[plot_name] = item
    return compact


def _compact_comparison_statistics(comparison):
    """Retain labels and scalar metric fields from one pair comparison."""
    compact = {key: value for key, value in comparison.items() if key == "candidate_label"}
    for population in (SIMULATED_POPULATION, TRIGGERED_POPULATION):
        if population in comparison:
            compact[population] = {
                key: comparison[population][key]
                for key in _COMPACT_STATISTIC_FIELDS
                if key in comparison[population]
            }
    if "metric" in comparison:
        compact.update(
            {key: comparison[key] for key in _COMPACT_STATISTIC_FIELDS if key in comparison}
        )
    return compact


def get_trigger_multiplicity_counts(metrics, bin_edges):
    """Return trigger multiplicity counts for the requested bin edges."""
    if metrics.trigger_multiplicity_histogram:
        counts, source_edges = metrics.trigger_multiplicity_histogram
        if np.array_equal(source_edges, bin_edges):
            return np.asarray(counts, dtype=float)
        centers = 0.5 * (source_edges[:-1] + source_edges[1:])
        rebinned, _ = np.histogram(centers, bins=bin_edges, weights=counts)
        return rebinned
    if metrics.trigger_multiplicity.size == 0:
        return None
    counts, _ = np.histogram(metrics.trigger_multiplicity, bins=bin_edges)
    return counts


def aggregate_mixed_trigger_combinations(trigger_combinations):
    """Aggregate mixed trigger combinations by display label in one pass."""
    counts = Counter()
    for combination, count in trigger_combinations.items():
        label = _format_mixed_combination_label(combination)
        if label is not None:
            counts[label] += count
    return counts


def fractions_per_triggered_events(counts, metrics):
    """Normalize counts by triggered-event count and return Poisson errors."""
    event_norm = metrics.triggered_event_count if metrics.triggered_event_count > 0 else 1
    return counts / event_norm, np.sqrt(counts) / event_norm


def _format_mixed_combination_label(combination):
    """Format mixed trigger combination as '<signature> | <tel1> + <tel2> + ...'."""
    type_counts, telescope_names = _type_counts_from_combination(combination)
    if not _is_mixed_type_combination_counts(type_counts):
        return None
    signature = "+".join(str(count) for _, count in sorted(type_counts.items()))
    telescopes_label = " + ".join(telescope_names)
    return f"{signature} | {telescopes_label}"


def _is_mixed_type_combination_counts(type_counts):
    """Return whether type counts represent a supported mixed trigger signature."""
    if len(type_counts) < 2:
        return False
    return tuple(sorted(type_counts.values())) in {(1, 1), (1, 2)}


def _type_counts_from_combination(combination):
    """Return per-type multiplicities and original telescope names for a combination."""
    telescope_names = [name for name in combination.split(",") if name]
    type_counts = {}
    for telescope_name in telescope_names:
        try:
            tel_type = names.get_array_element_type_from_name(telescope_name)
        except ValueError:
            continue
        type_counts[tel_type] = type_counts.get(tel_type, 0) + 1
    return type_counts, telescope_names


def record_telescope_type_participation(comparison_statistics, metrics_per_production):
    """Record participation-fraction comparisons for each telescope type."""
    telescope_names = sorted(
        {
            telescope
            for metrics in metrics_per_production
            for telescope in metrics.telescope_participation
        }
    )
    telescope_types = sorted(
        {names.get_array_element_type_from_name(telescope) for telescope in telescope_names}
    )
    for telescope_type in telescope_types:
        categories = [
            telescope
            for telescope in telescope_names
            if names.get_array_element_type_from_name(telescope) == telescope_type
        ]
        counts_per_label = {
            metrics.label: np.asarray(
                [metrics.telescope_participation.get(telescope, 0) for telescope in categories],
                dtype=float,
            )
            for metrics in metrics_per_production
        }
        statistics = build_aligned_count_statistics(
            metrics_per_production,
            counts_per_label,
            metadata={CATEGORY_COUNT_KEY: len(categories)},
            metric=JENSEN_SHANNON_METRIC,
        )
        statistics[PLOT_NAME_KEY] = f"telescope_participation_fraction_{telescope_type}"
        record_plot_statistics(comparison_statistics, statistics[PLOT_NAME_KEY], statistics)


def build_aligned_count_statistics(
    metrics_per_production,
    counts_per_label,
    metadata,
    metric=JENSEN_SHANNON_METRIC,
    bin_edges=None,
):
    """Build comparison statistics for plots represented by aligned count arrays."""
    baseline_label = metrics_per_production[0].label if metrics_per_production else None
    baseline_counts = counts_per_label.get(baseline_label)
    stats_summary = {
        "baseline_label": baseline_label,
        "metric_type": "aligned_counts",
        "metric": metric,
        "metadata": metadata,
        "comparisons": [],
    }

    if baseline_label is None or baseline_counts is None:
        return stats_summary

    for metrics in metrics_per_production[1:]:
        candidate_counts = counts_per_label.get(metrics.label)
        if candidate_counts is None:
            continue
        result = compare_histogram_counts(
            baseline_counts,
            candidate_counts,
            metric=metric,
            bin_edges=bin_edges,
        )
        result = {
            **result,
            "candidate_label": metrics.label,
            "baseline_total": float(np.sum(baseline_counts)),
            "candidate_total": float(np.sum(candidate_counts)),
            "baseline_counts": np.asarray(baseline_counts, dtype=int).tolist(),
            "candidate_counts": np.asarray(candidate_counts, dtype=int).tolist(),
        }
        if metric == WASSERSTEIN_METRIC and metadata.get(BIN_EDGES_KEY) and result.get(_VALID):
            centers = 0.5 * (
                np.asarray(metadata[BIN_EDGES_KEY], dtype=float)[:-1]
                + np.asarray(metadata[BIN_EDGES_KEY], dtype=float)[1:]
            )
            pooled_counts = np.asarray(baseline_counts) + np.asarray(candidate_counts)
            pooled_total = float(np.sum(pooled_counts))
            pooled_mean = (
                float(np.average(centers, weights=pooled_counts)) if pooled_total > 0 else 0.0
            )
            result["normalized_value"] = (
                float(result["value"]) / pooled_mean if pooled_mean > 0 else None
            )
        stats_summary["comparisons"].append(result)
    return stats_summary


def build_quantity_distribution_statistics(
    metrics_per_production,
    quantity_name,
    counts_by_label,
    cumulative,
):
    """Build comparison statistics for quantity distribution plots."""
    baseline_label = metrics_per_production[0].label if metrics_per_production else None
    stats_summary = {
        "baseline_label": baseline_label,
        "metric_type": "quantity_distribution",
        "metric": "ks",
        QUANTITY_NAME_KEY: quantity_name,
        CUMULATIVE_KEY: bool(cumulative),
        "comparisons": [],
    }
    if baseline_label is None or baseline_label not in counts_by_label:
        return stats_summary

    baseline_data = counts_by_label[baseline_label]

    for metrics in metrics_per_production[1:]:
        if metrics.label not in counts_by_label:
            continue
        candidate_data = counts_by_label[metrics.label]

        sim_stats = _compare_distribution_data(
            baseline_data,
            candidate_data,
            SIMULATED_POPULATION,
        )
        trig_stats = _compare_distribution_data(
            baseline_data,
            candidate_data,
            TRIGGERED_POPULATION,
        )

        comparison_record = {
            "candidate_label": metrics.label,
            SIMULATED_POPULATION: sim_stats,
            TRIGGERED_POPULATION: trig_stats,
        }
        stats_summary["comparisons"].append(comparison_record)
    return stats_summary


def _compare_distribution_data(baseline_data, candidate_data, event_kind):
    """Compare either raw sample arrays or pre-binned count arrays."""
    sample_key = f"{event_kind}_samples"
    if baseline_data[sample_key] is not None and candidate_data[sample_key] is not None:
        result = compare_samples_with_statistics(
            baseline_data[sample_key],
            candidate_data[sample_key],
            baseline_data[BIN_EDGES_KEY],
        )
        result["baseline_total"] = len(baseline_data[sample_key])
        result["candidate_total"] = len(candidate_data[sample_key])
        return result
    result = compare_histogram_counts(
        baseline_data[event_kind],
        candidate_data[event_kind],
        metric="ks",
    )
    result["baseline_counts"] = np.asarray(baseline_data[event_kind], dtype=int).tolist()
    result["candidate_counts"] = np.asarray(candidate_data[event_kind], dtype=int).tolist()
    result[BIN_EDGES_KEY] = np.asarray(baseline_data[BIN_EDGES_KEY], dtype=float).tolist()
    result["baseline_total"] = float(np.sum(baseline_data[event_kind]))
    result["candidate_total"] = float(np.sum(candidate_data[event_kind]))
    return result


def _format_statistic_value(value):
    """Format statistic values for plot text annotations."""
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _format_metric_value(statistics):
    """Return a display label and formatted value for one metric result."""
    metric_labels = {
        "ks": "KS",
        JENSEN_SHANNON_METRIC: "JS",
        WASSERSTEIN_METRIC: "W1",
    }
    metric = statistics.get("metric", "ks")
    value_key = {
        "ks": "ks_statistic",
        JENSEN_SHANNON_METRIC: _JENSEN_SHANNON_DISTANCE,
        WASSERSTEIN_METRIC: _WASSERSTEIN_DISTANCE,
    }.get(metric, VALUE_KEY)
    value = statistics.get(value_key, statistics.get(VALUE_KEY))
    return metric_labels.get(metric, metric), _format_statistic_value(value)


def initialize_comparison_statistics(metrics_per_production):
    """Initialize top-level comparison statistics structure."""
    baseline = metrics_per_production[0] if metrics_per_production else None
    return {
        "format_version": 1,
        "baseline": {
            "label": baseline.label if baseline else None,
            "simulated_event_count": baseline.simulated_event_count if baseline else 0,
            "triggered_event_count": baseline.triggered_event_count if baseline else 0,
        },
        "comparison_sets": [
            {
                "label": metrics.label,
                "simulated_event_count": metrics.simulated_event_count,
                "triggered_event_count": metrics.triggered_event_count,
            }
            for metrics in metrics_per_production[1:]
        ],
        PLOT_STATISTICS_KEY: {},
    }


def record_plot_statistics(comparison_statistics, plot_name, statistics):
    """Store plot-level statistics if available."""
    if statistics is None:
        return
    comparison_statistics[PLOT_STATISTICS_KEY][plot_name] = statistics


def write_comparison_statistics_json(comparison_statistics, output_path):
    """Write comparison statistics JSON file in the output directory."""
    output_file = output_path / product_filename("comparison_statistics")
    _logger.info(f"Writing comparison statistics: {output_file}")
    ascii_handler.write_data_to_file(comparison_statistics, output_file, sort_keys=True)
    return output_file


def fraction_with_poisson_errors(counts, _metrics=None):
    """Return normalized bin fractions and Poisson errors for counts."""
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        return np.zeros_like(counts, dtype=float), np.zeros_like(counts, dtype=float)
    fractions = counts / total
    errors = np.sqrt(counts) / total
    return fractions, errors


def comparison_statistics_annotation(statistics):
    """Return comparison metric text for baseline-vs-candidate comparisons."""
    if statistics is None or not statistics.get("comparisons"):
        return None

    metric_type = statistics.get("metric_type")
    lines = [f"Baseline: {statistics.get('baseline_label', 'n/a')}"]
    for comparison in statistics["comparisons"]:
        label = comparison.get("candidate_label", "candidate")
        if metric_type == "quantity_distribution":
            simulated_label, simulated_value = _format_metric_value(
                comparison.get(SIMULATED_POPULATION, {})
            )
            triggered_label, triggered_value = _format_metric_value(
                comparison.get(TRIGGERED_POPULATION, {})
            )
            metric_label = (
                simulated_label
                if simulated_label == triggered_label
                else f"{simulated_label}/{triggered_label}"
            )
            lines.append(f"{label}: {metric_label} s/t={simulated_value}/{triggered_value}")
            continue
        metric_label, metric_value = _format_metric_value(comparison)
        lines.append(f"{label}: {metric_label}={metric_value}")

    return "\n".join(lines)
