"""Plot telescope-level signal comparisons across simulation productions."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from simtools.io import ascii_handler
from simtools.sim_events.production_comparison import ensure_single_array_layout_name
from simtools.statistics import compare_samples_with_statistics

_logger = logging.getLogger(__name__)

_OBSERVABLES = (
    ("pedestals", "Pedestal", "pedestals.png"),
    ("signals", "Integrated signal", "signals.png"),
    ("peak_timing", "Peak sample", "peak_timing.png"),
    ("triggered_pixels", "Triggered pixels per event", "triggered_pixels.png"),
)


def plot(metrics_by_telescope, output_path, array_layout_name, bins=40):
    """Create signal comparison plots for every telescope in a layout.

    Parameters
    ----------
    metrics_by_telescope : dict[str, list[ProductionSignalMetrics]]
        Metrics grouped by telescope name and ordered by production.
    output_path : pathlib.Path
        Base output directory.
    array_layout_name : str
        Array layout name used to validate the signal comparison selection.
    bins : int, optional
        Number of histogram bins.

    Returns
    -------
    list[pathlib.Path]
        Paths to the comparison statistics JSON files.
    """
    ensure_single_array_layout_name(array_layout_name)
    statistics_files = []
    for telescope_name, metrics in metrics_by_telescope.items():
        telescope_path = Path(output_path) / telescope_name
        telescope_path.mkdir(parents=True, exist_ok=True)
        comparison_statistics = _plot_telescope_comparisons(metrics, telescope_path, bins)
        statistics_file = telescope_path / "comparison_statistics.json"
        ascii_handler.write_data_to_file(comparison_statistics, statistics_file, sort_keys=True)
        statistics_files.append(statistics_file)
    return statistics_files


def _plot_telescope_comparisons(metrics, output_path, bins):
    """Plot all observables for one telescope and return their statistics."""
    if not metrics:
        raise ValueError("At least one production is required for signal comparison.")
    statistics = {
        "format_version": 1,
        "baseline": _production_summary(metrics[0]),
        "comparison_sets": [_production_summary(item) for item in metrics[1:]],
        "plot_statistics": {},
    }
    for observable, x_label, filename in _OBSERVABLES:
        plot_statistics = _plot_observable(
            metrics,
            output_path,
            observable,
            x_label,
            filename,
            bins,
        )
        if plot_statistics is not None:
            statistics["plot_statistics"][observable] = plot_statistics
    return statistics


def _production_summary(metrics):
    """Return the common comparison summary for one telescope production."""
    event_count = int(metrics.triggered_pixels.size)
    triggered_event_count = int(np.count_nonzero(metrics.triggered_pixels > 0))
    return {
        "label": metrics.label,
        "simulated_event_count": event_count,
        "triggered_event_count": triggered_event_count,
    }


def _plot_observable(metrics, output_path, observable, x_label, filename, bins):
    """Plot one observable and calculate candidate statistics."""
    samples = [np.asarray(getattr(item, observable)) for item in metrics]
    non_empty = [values for values in samples if values.size]
    if not non_empty:
        _logger.warning("Skipping %s plot because no values are available.", observable)
        return None

    bin_edges = np.histogram_bin_edges(np.concatenate(non_empty), bins=bins)
    fig, ax = plt.subplots(figsize=(8, 5))
    for item, values in zip(metrics, samples):
        if not values.size:
            continue
        counts, _ = np.histogram(values, bins=bin_edges)
        fractions = counts / counts.sum()
        ax.stairs(fractions, bin_edges, label=item.label)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Fraction")
    ax.set_title(f"{x_label} comparison")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output_path / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    comparisons = []
    for item, values in zip(metrics[1:], samples[1:]):
        if not values.size or not samples[0].size:
            continue
        comparison = compare_samples_with_statistics(samples[0], values, bin_edges)
        comparison["candidate_label"] = item.label
        comparisons.append(comparison)
    return {
        "baseline_label": metrics[0].label,
        "metric_type": "aligned_counts",
        "metric": "ks",
        "metadata": {"bin_edges": bin_edges.tolist()},
        "comparisons": comparisons,
    }
