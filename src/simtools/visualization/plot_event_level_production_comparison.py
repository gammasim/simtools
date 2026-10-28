"""Plot event-level comparisons across multiple simulation productions."""

import functools
import logging
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import numpy as np
from astropy import units as u

from simtools.io import ascii_handler, io_handler
from simtools.io.file_type import product_filename
from simtools.statistics import (
    compare_histogram_counts,
    compare_samples_with_statistics,
)
from simtools.utils import names
from simtools.visualization.matplotlib_backend import pyplot as plt
from simtools.visualization.visualize import pdf_report, save_figure

_JENSEN_SHANNON_DISTANCE = "jensen_shannon_distance"
_MIXED_TRIGGER_COMBINATIONS = "mixed_trigger_combinations"
_SINGLE_TELESCOPE_TRIGGER_DISTRIBUTION = "single_telescope_trigger_distribution"
_TRIGGER_COMBINATION = "trigger_combination"
_VALID = "valid"
_WASSERSTEIN_DISTANCE = "wasserstein_distance"

_CUMULATIVE_SUFFIX = "_cumulative"
_BIN_EDGES_KEY = "bin_edges"
_CATEGORY_COUNT_KEY = "category_count"
_JENSEN_SHANNON_METRIC = "jensen_shannon"
_PLOT_NAME_KEY = "plot_name"
_REASON_KEY = "reason"
_SIMULATED_POPULATION = "simulated"
_PARTICIPATION_PLOT = "telescope_participation_fraction"
_TELESCOPE_TYPES_KEY = "telescope_types"
_TRIGGERED_POPULATION = "triggered"
_WASSERSTEIN_METRIC = "wasserstein"

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
    _REASON_KEY,
    "pvalue",
    "ks_statistic",
    "ks_pvalue",
    _JENSEN_SHANNON_DISTANCE,
    _WASSERSTEIN_DISTANCE,
    "baseline_total",
    "candidate_total",
)

_logger = logging.getLogger(__name__)
_COMPACT_FIGURES = ContextVar("event_comparison_compact_figures", default=None)
_COMPARISON_TITLE = ContextVar("event_comparison_title", default=None)


def _output_directory_for_array_layout_selection(output_directory, array_layout_name):
    """Return a plot output directory for an optional array-layout selection."""
    if not array_layout_name:
        return output_directory

    if not isinstance(array_layout_name, str):
        if len(array_layout_name) != 1:
            raise ValueError("Plot one array layout at a time.")
        array_layout_name = array_layout_name[0]

    output_path = output_directory / array_layout_name
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def plot(
    metrics_per_production,
    output_path=None,
    array_layout_name=None,
    bins=40,
    figure_format=None,
    report_mode=DIAGNOSTIC_REPORT,
    compact_figures=None,
    comparison_title=None,
    comparison_context=None,
):
    """Create all event-level production comparison plots.

    Parameters
    ----------
    metrics_per_production : list[ProductionEventMetrics]
        Aggregated metrics per production.
    output_path : pathlib.Path, optional
        Output directory for the generated plots. Falls back to the current
        application output directory when omitted.
    array_layout_name : str, optional
        Array-layout selection used to collect the metrics. When set, plots are
        written into a subdirectory with that name.
    bins : int, optional
        Number of bins for 1D histograms.
    figure_format : list[str] or str, optional
        Figure formats to write. Formats supported by Matplotlib can be used.
        Defaults to the configured formats or ``["png"]``.
    comparison_title : str, optional
        Context heading shown above each plot, such as the configuration of a
        matched production pair.
    comparison_context : dict, optional
        Baseline and candidate configuration metadata. Compact reports omit fixed-energy and
        on-axis angular distributions when both configurations establish those conditions.

    Returns
    -------
    pathlib.Path
        Path to the JSON file containing the comparison statistics.
    """
    if report_mode not in {COMPACT_REPORT, DIAGNOSTIC_REPORT, "none"}:
        raise ValueError(f"Unsupported report mode: {report_mode}")
    comparison_statistics = _initialize_comparison_statistics(metrics_per_production)
    output_path = output_path or io_handler.IOHandler().get_output_directory()
    output_path = _output_directory_for_array_layout_selection(output_path, array_layout_name)
    output_path.mkdir(parents=True, exist_ok=True)
    suppressed_distributions = _metadata_suppressed_distributions(comparison_context)

    with _comparison_report(output_path, report_mode, compact_figures, comparison_title):
        for plot_name, plot_function in _TOP_LEVEL_STAT_PLOTS:
            _record_plot_statistics(
                comparison_statistics,
                plot_name,
                plot_function(metrics_per_production, output_path, figure_format=figure_format),
            )

        _record_telescope_type_participation(comparison_statistics, metrics_per_production)

        _plot_quantity_comparisons(
            metrics_per_production,
            output_path,
            bins,
            comparison_statistics,
            figure_format,
            report_mode,
            suppressed_distributions,
        )
        _plot_per_type_comparisons(
            metrics_per_production,
            output_path,
            bins,
            comparison_statistics,
            figure_format,
            report_mode,
            suppressed_distributions,
        )

    participation = comparison_statistics[PLOT_STATISTICS_KEY].get(_PARTICIPATION_PLOT)
    if participation is not None:
        participation[_TELESCOPE_TYPES_KEY] = sorted(
            {
                telescope
                for metrics in metrics_per_production
                for telescope in metrics.per_type
                if telescope not in _SPECIAL_TRIGGER_SUBSETS
            }
        )

    # Keep JSON scalar-only in every mode; detailed histograms are already in the plots.
    comparison_statistics[PLOT_STATISTICS_KEY] = _compact_plot_statistics(
        comparison_statistics[PLOT_STATISTICS_KEY]
    )
    return _write_comparison_statistics_json(comparison_statistics, output_path)


def _compact_plot_statistics(plot_statistics):
    """Keep JSON scalar-only so diagnostic histogram arrays do not create huge files."""
    compact = {}
    for plot_name, statistics in plot_statistics.items():
        item = {
            key: statistics[key]
            for key in (
                _PLOT_NAME_KEY,
                "baseline_label",
                "metric_type",
                _TELESCOPE_TYPES_KEY,
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
    for population in (_SIMULATED_POPULATION, _TRIGGERED_POPULATION):
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


@contextmanager
def _comparison_report(output_path, report_mode, compact_figures=None, comparison_title=None):
    """Route pages to one compact PDF or suppress rendering without changing metrics."""
    if report_mode == COMPACT_REPORT and compact_figures is None:
        with pdf_report(output_path / "comparison-report.pdf") as document:
            with _comparison_context(document, comparison_title):
                yield
        return
    document = compact_figures if report_mode == COMPACT_REPORT else None
    if report_mode == "none":
        document = False
    with _comparison_context(document, comparison_title):
        yield


@contextmanager
def _comparison_context(document, comparison_title):
    """Set and restore the active report and comparison title."""
    title_token = _COMPARISON_TITLE.set(comparison_title)
    token = _COMPACT_FIGURES.set(document)
    try:
        yield
    finally:
        _COMPACT_FIGURES.reset(token)
        _COMPARISON_TITLE.reset(title_token)


def _plot_quantity_comparisons(
    metrics_per_production,
    output_path,
    bins,
    comparison_statistics,
    figure_format,
    report_mode,
    suppressed_distributions,
):
    """Plot quantity-based comparisons and collect their statistics."""
    for quantity_name, x_label, x_scale in _QUANTITY_CONFIGS:
        if quantity_name in _TRIGGERED_FRACTION_QUANTITIES:
            _plot_triggered_vs_quantity(
                metrics_per_production,
                output_path,
                quantity_name=quantity_name,
                x_label=x_label,
                x_scale=x_scale,
                bins=bins,
                figure_format=figure_format,
            )
        for cumulative in _distribution_cumulative_variants(quantity_name):
            if cumulative is None:
                continue
            cumulative_suffix = _CUMULATIVE_SUFFIX if cumulative else ""
            _record_plot_statistics(
                comparison_statistics,
                f"distribution_{quantity_name}{cumulative_suffix}",
                _plot_quantity_distribution(
                    metrics_per_production,
                    output_path,
                    quantity_name=quantity_name,
                    x_label=x_label,
                    x_scale=x_scale,
                    bins=bins,
                    cumulative=cumulative,
                    figure_format=figure_format,
                    render=_render_quantity(
                        quantity_name, cumulative, report_mode, suppressed_distributions
                    ),
                ),
            )


def _render_quantity(quantity_name, cumulative, report_mode, suppressed_distributions):
    """Suppress redundant compact pages while retaining their numerical statistics."""
    if report_mode != COMPACT_REPORT:
        return True
    if quantity_name == CORE_DISTANCE and cumulative:
        return False
    return quantity_name not in suppressed_distributions


def _metadata_suppressed_distributions(comparison_context):
    """Identify constant distributions only when both production metadata entries agree."""
    configurations = (comparison_context or {}).get("configurations", {})
    if len(configurations) < 2:
        return set()
    suppressed = set()
    if _constant_metadata_bounds(configurations.values(), "energy_min", "energy_max", u.GeV):
        suppressed.add(ENERGY)
    if _constant_metadata_bounds(
        configurations.values(), "view_cone_min", "view_cone_max", u.deg, target=0.0
    ):
        suppressed.add(ANGULAR_DISTANCE)
    return suppressed


def _constant_metadata_bounds(configurations, lower_key, upper_key, unit, target=None):
    """Compare finite metadata bounds in common units with floating-point tolerance."""
    bounds = []
    for configuration in configurations:
        for key in (lower_key, upper_key):
            try:
                value = configuration[key]
                quantity = (
                    u.Quantity(value[VALUE_KEY], value["unit"])
                    if isinstance(value, dict)
                    else u.Quantity(value, unit)
                )
                bounds.append(float(quantity.to_value(unit)))
            except KeyError, TypeError, ValueError:
                return False
    if not bounds or not np.all(np.isfinite(bounds)):
        return False
    reference = bounds[0] if target is None else target
    return bool(np.all(np.isclose(bounds, reference, rtol=1e-8, atol=1e-12)))


def _plot_per_type_comparisons(
    metrics_per_production,
    output_path,
    bins,
    comparison_statistics,
    figure_format,
    report_mode,
    suppressed_distributions,
):
    """Plot per-telescope-type comparisons and collect their statistics."""
    all_types = sorted(
        {
            tel_type
            for metrics in metrics_per_production
            for tel_type in metrics.per_type
            if tel_type not in _SPECIAL_TRIGGER_SUBSETS
        }
    )
    for tel_type in all_types:
        type_metrics = [
            metrics.per_type[tel_type]
            for metrics in metrics_per_production
            if tel_type in metrics.per_type
        ]
        for plot_fn in _PER_TYPE_PLOT_FNS:
            plot_options = {}
            if (
                isinstance(plot_fn, functools.partial)
                and plot_fn.func is _plot_quantity_distribution
            ):
                plot_options["render"] = (
                    report_mode != COMPACT_REPORT
                    or plot_fn.keywords[QUANTITY_NAME_KEY] not in suppressed_distributions
                )
            statistics = plot_fn(
                type_metrics,
                output_path,
                suffix=f"_{tel_type}",
                bins=bins,
                figure_format=figure_format,
                **plot_options,
            )
            if statistics is None:
                continue
            plot_name = _plot_name_from_statistics(statistics)
            _record_plot_statistics(comparison_statistics, plot_name, statistics)


def _save_figure(fig, output_path, filename, figure_format=None):
    """Save figure and close it."""
    comparison_title = _COMPARISON_TITLE.get()
    if comparison_title:
        fig.suptitle(comparison_title, fontsize=10, fontweight="bold")
    document = _COMPACT_FIGURES.get()
    if document is False:
        plt.close(fig)
        return
    if document is not None:
        label = Path(filename).stem
        fig.set_label(label)
        if hasattr(document, "add"):
            document.add(fig)
        else:
            document.append(fig)
        return
    output_file = output_path / Path(filename).with_suffix("")
    save_figure(
        fig,
        output_file,
        figure_format=figure_format,
        dpi=300,
        close=True,
        pil_kwargs={"compress_level": 1},
    )


def _plot_trigger_multiplicity(
    metrics_per_production, output_path, suffix="", bins=None, figure_format=None
):
    """Plot triggered telescope multiplicity distributions."""
    del bins
    global_max = 0
    for metrics in metrics_per_production:
        if metrics.trigger_multiplicity_histogram:
            _, bin_edges = metrics.trigger_multiplicity_histogram
            global_max = max(global_max, int(bin_edges[-1] - 1))
        elif metrics.trigger_multiplicity.size > 0:
            global_max = max(global_max, int(np.max(metrics.trigger_multiplicity)))

    if global_max == 0:
        _logger.warning("Skipping trigger multiplicity plot, no triggered events available.")
        return None

    bin_edges = np.arange(1, global_max + 2)
    counts_per_label = {}
    for metrics in metrics_per_production:
        counts = _get_trigger_multiplicity_counts(metrics, bin_edges)
        if counts is None:
            continue
        counts_per_label[metrics.label] = counts
    statistics = _build_aligned_count_statistics(
        metrics_per_production,
        counts_per_label,
        metadata={
            _BIN_EDGES_KEY: bin_edges.astype(float).tolist(),
            "bin_count": len(bin_edges) - 1,
        },
        metric=_WASSERSTEIN_METRIC,
        bin_edges=bin_edges,
    )
    statistics[_PLOT_NAME_KEY] = f"trigger_multiplicity{suffix}"
    if _COMPACT_FIGURES.get() is False:
        return statistics
    fig, ax = plt.subplots(figsize=(9, 6))
    for label, counts in counts_per_label.items():
        fractions, errors = _fraction_with_poisson_errors(counts)
        stairs_artist = ax.stairs(fractions, bin_edges, linewidth=1.5, label=label)
        _plot_histogram_error_bars(
            ax,
            bin_edges,
            fractions,
            errors,
            color=_artist_color(stairs_artist),
        )

    ax.set_xlabel("Triggered Telescopes per Event")
    ax.set_ylabel("Fraction of Triggered Events")
    type_label = f" ({suffix.lstrip('_').replace('_', ' ')})" if suffix else ""
    ax.set_title(f"Trigger Multiplicity {type_label}")
    ax.set_xticks(bin_edges[:-1])
    ax.grid(alpha=0.25)
    ax.legend()

    annotate_comparison_statistics(ax, statistics)

    _save_figure(fig, output_path, f"trigger_multiplicity{suffix}.png", figure_format)
    return statistics


def _get_trigger_multiplicity_counts(metrics, bin_edges):
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


def _plot_trigger_combinations(metrics_per_production, output_path, top_n=12, figure_format=None):
    """Plot trigger combination distributions for each production."""
    combinations = set()
    for metrics in metrics_per_production:
        combinations.update(metrics.trigger_combinations.keys())

    if len(combinations) == 0:
        _logger.warning("Skipping trigger combination plot, no combinations available.")
        return None

    totals = {}
    for combination in combinations:
        totals[combination] = sum(
            metrics.trigger_combinations.get(combination, 0) for metrics in metrics_per_production
        )

    all_categories = sorted(combinations)
    selected = [name for name, _ in sorted(totals.items(), key=lambda item: item[1], reverse=True)]
    selected = selected[:top_n]

    statistics = _plot_grouped_fraction_bars(
        metrics_per_production,
        categories=selected,
        counts_getter=lambda metrics, category_labels: [
            metrics.trigger_combinations.get(name, 0) for name in category_labels
        ],
        normalization_fn=_fractions_per_triggered_events,
        output_path=output_path,
        filename="trigger_combination.png",
        y_label="Fraction of Triggered Events",
        title="Top Trigger Combinations",
        figure_width=max(10, len(selected) * 1.1),
        x_rotation=45,
        x_ha="right",
        statistics_categories=all_categories,
        figure_format=figure_format,
    )
    if statistics is not None:
        statistics[_PLOT_NAME_KEY] = _TRIGGER_COMBINATION
    return statistics


def _plot_single_telescope_trigger_frequencies(
    metrics_per_production, output_path, figure_format=None
):
    """Plot single-telescope trigger frequency distributions by telescope name."""
    telescope_names = sorted(
        {
            combination
            for metrics in metrics_per_production
            for combination in metrics.trigger_combinations
            if "," not in combination
        }
    )

    if len(telescope_names) == 0:
        _logger.warning("Skipping single-telescope trigger frequency plot, no data available.")
        return None

    statistics = _plot_grouped_fraction_bars(
        metrics_per_production,
        categories=telescope_names,
        counts_getter=lambda metrics, category_labels: [
            metrics.trigger_combinations.get(telescope_name, 0)
            for telescope_name in category_labels
        ],
        normalization_fn=_fraction_with_poisson_errors,
        output_path=output_path,
        filename="single_telescope_trigger_distribution.png",
        y_label="Fraction of Single-Telescope Triggers",
        title="Single-Telescope Trigger Distribution",
        figure_width=max(10, len(telescope_names) * 0.45),
        x_rotation=90,
        x_ha="center",
        figure_format=figure_format,
    )
    if statistics is not None:
        statistics[_PLOT_NAME_KEY] = _SINGLE_TELESCOPE_TRIGGER_DISTRIBUTION
    return statistics


def _plot_mixed_trigger_combinations(
    metrics_per_production, output_path, top_n=12, figure_format=None
):
    """Plot mixed-type trigger combinations with multiplicity signatures and telescope names."""
    counts_by_production = {
        metrics.label: _aggregate_mixed_trigger_combinations(metrics.trigger_combinations)
        for metrics in metrics_per_production
    }
    total_counts = Counter()
    for counts in counts_by_production.values():
        total_counts.update(counts)
    mixed_labels = [label for label, _ in total_counts.most_common(top_n)]
    if not mixed_labels:
        _logger.warning("Skipping mixed trigger combination plot, no mixed-type combinations.")
        return None

    statistics = _plot_grouped_fraction_bars(
        metrics_per_production,
        categories=mixed_labels,
        counts_getter=lambda metrics, category_labels: [
            counts_by_production[metrics.label].get(label, 0) for label in category_labels
        ],
        normalization_fn=_fraction_with_poisson_errors,
        output_path=output_path,
        filename="mixed_trigger_combinations.png",
        y_label="Fraction of Mixed-Type Triggers",
        title="Mixed-Type Trigger Combinations",
        figure_width=max(12, len(mixed_labels) * 0.8),
        x_rotation=45,
        x_ha="right",
        statistics_categories=sorted(total_counts),
        figure_format=figure_format,
    )
    if statistics is not None:
        statistics[_PLOT_NAME_KEY] = _MIXED_TRIGGER_COMBINATIONS
    return statistics


def _aggregate_mixed_trigger_combinations(trigger_combinations):
    """Aggregate mixed trigger combinations by display label in one pass."""
    counts = Counter()
    for combination, count in trigger_combinations.items():
        label = _format_mixed_combination_label(combination)
        if label is not None:
            counts[label] += count
    return counts


def _fractions_per_triggered_events(counts, metrics):
    """Normalize counts by triggered-event count and return Poisson errors."""
    event_norm = metrics.triggered_event_count if metrics.triggered_event_count > 0 else 1
    return counts / event_norm, np.sqrt(counts) / event_norm


def _plot_grouped_fraction_bars(
    metrics_per_production,
    categories,
    counts_getter,
    normalization_fn,
    output_path,
    filename,
    y_label,
    title,
    figure_width,
    x_rotation,
    x_ha,
    statistics_categories=None,
    figure_format=None,
):
    """Plot grouped bars with fractions and Poisson error bars for each production."""
    statistics_categories = statistics_categories or categories
    statistics = _build_aligned_count_statistics(
        metrics_per_production,
        {
            metrics.label: np.asarray(counts_getter(metrics, statistics_categories), dtype=float)
            for metrics in metrics_per_production
        },
        metadata={
            "categories": list(statistics_categories),
            _CATEGORY_COUNT_KEY: len(statistics_categories),
            "display_categories": list(categories),
        },
        metric=_JENSEN_SHANNON_METRIC,
    )
    if _COMPACT_FIGURES.get() is False:
        return statistics
    x_values = np.arange(len(categories))
    width = 0.8 / max(1, len(metrics_per_production))
    fig, ax = plt.subplots(figsize=(figure_width, 6))

    for index, metrics in enumerate(metrics_per_production):
        counts = np.asarray(counts_getter(metrics, categories), dtype=float)
        fractions, errors = normalization_fn(counts, metrics)
        offset = (index - (len(metrics_per_production) - 1) / 2.0) * width
        ax.bar(
            x_values + offset,
            fractions,
            width=width,
            label=metrics.label,
            yerr=errors,
            error_kw={"elinewidth": 0.7, "capsize": 1, "capthick": 0.7},
        )

    ax.set_xticks(x_values)
    ax.set_xticklabels(categories, rotation=x_rotation, ha=x_ha)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    annotate_comparison_statistics(ax, statistics)

    _save_figure(fig, output_path, filename, figure_format)
    return statistics


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


def _plot_triggered_vs_quantity(
    metrics_per_production,
    output_path,
    quantity_name,
    x_label,
    x_scale,
    bins,
    suffix="",
    figure_format=None,
):
    """Plot simulated vs triggered distributions for one quantity."""
    if _COMPACT_FIGURES.get() is False:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    bin_edges = _get_global_quantity_bin_edges(
        metrics_per_production,
        quantity_name=quantity_name,
        x_scale=x_scale,
        bins=bins,
    )

    plotted = False
    for metrics in metrics_per_production:
        simulated, triggered = _get_quantity_arrays(metrics, quantity_name)
        sim_counts, _ = _get_quantity_counts(
            metrics, quantity_name, bin_edges, _SIMULATED_POPULATION
        )
        trig_counts, _ = _get_quantity_counts(
            metrics, quantity_name, bin_edges, _TRIGGERED_POPULATION
        )
        if sim_counts is None and simulated.size == 0:
            continue
        plotted = True
        if sim_counts is None:
            sim_counts, _ = np.histogram(simulated, bins=bin_edges)
        if trig_counts is None:
            trig_counts, _ = np.histogram(triggered, bins=bin_edges)
        with np.errstate(divide="ignore", invalid="ignore"):
            efficiency = np.divide(
                trig_counts,
                sim_counts,
                out=np.zeros_like(trig_counts, dtype=float),
                where=sim_counts > 0,
            )
        _plot_series(ax, bin_edges, efficiency, metrics.label, quantity_name, linewidth=2)

    if not plotted:
        _logger.warning(f"Skipping triggered fraction plot for {quantity_name}, no data available.")
        plt.close(fig)
        return
    ax.set_xlabel(x_label)
    ax.set_ylabel("Triggered / Simulated")
    type_label = f" ({suffix.lstrip('_').replace('_', ' ')})" if suffix else ""
    ax.set_title(f"Triggered Event Fraction vs {x_label}{type_label}")
    ax.set_xscale(x_scale)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend()

    _save_figure(
        fig,
        output_path,
        f"triggered_fraction_vs_{quantity_name}{suffix}.png",
        figure_format,
    )


def _get_bin_edges(values, x_scale, bins):
    """Return robust bin edges for the given values."""
    if len(values) == 0:
        return np.linspace(0.0, 1.0, bins + 1)

    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if np.isclose(min_value, max_value):
        max_value = min_value + 1.0

    if x_scale == "log":
        min_value = max(min_value, np.finfo(float).tiny)
        return np.logspace(np.log10(min_value), np.log10(max_value), bins + 1)
    return np.linspace(min_value, max_value, bins + 1)


def _get_quantity_arrays(metrics, quantity_name):
    """Return (simulated, triggered) arrays for a named quantity.

    Parameters
    ----------
    metrics : ProductionEventMetrics
        Metrics for one production.
    quantity_name : str
        One of ``"energy"``, ``"core_distance"``, or ``"angular_distance"``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Simulated and triggered arrays for the requested quantity.
    """
    if quantity_name == ENERGY:
        return metrics.simulated_energies, metrics.triggered_energies
    if quantity_name == CORE_DISTANCE:
        return metrics.simulated_core_distances, metrics.triggered_core_distances
    return metrics.simulated_angular_distances, metrics.triggered_angular_distances


def _plot_quantity_distribution(
    metrics_per_production,
    output_path,
    quantity_name,
    x_label,
    x_scale,
    bins=40,
    suffix="",
    cumulative=False,
    figure_format=None,
    render=True,
):
    """Plot simulated and triggered distributions for one quantity.

    Parameters
    ----------
    metrics_per_production : list[ProductionEventMetrics]
        Aggregated metrics per production.
    output_path : pathlib.Path
        Output directory for generated figures.
    quantity_name : str
        One of ``"energy"``, ``"core_distance"``, or ``"angular_distance"``.
    x_label : str
        Axis label for the quantity.
    x_scale : str
        Axis scale (``"log"`` or ``"linear"``).
    bins : int, optional
        Number of histogram bins.
    suffix : str, optional
        Filename and title suffix for per-type variants.
    cumulative : bool, optional
        Whether to plot cumulative distributions.
    """
    bin_edges = _get_global_quantity_bin_edges(
        metrics_per_production,
        quantity_name=quantity_name,
        x_scale=x_scale,
        bins=bins,
    )
    counts_by_label = {}
    for metrics in metrics_per_production:
        simulated, triggered = _get_quantity_arrays(metrics, quantity_name)
        sim_counts, simulated_samples = _get_quantity_counts(
            metrics, quantity_name, bin_edges, _SIMULATED_POPULATION
        )
        trig_counts, triggered_samples = _get_quantity_counts(
            metrics, quantity_name, bin_edges, _TRIGGERED_POPULATION
        )
        if sim_counts is None and simulated.size == 0:
            continue

        if sim_counts is None:
            sim_counts, _ = np.histogram(simulated, bins=bin_edges)
            simulated_samples = simulated
        if trig_counts is None:
            trig_counts, _ = np.histogram(triggered, bins=bin_edges)
            triggered_samples = triggered
        counts_by_label[metrics.label] = {
            _SIMULATED_POPULATION: sim_counts,
            _TRIGGERED_POPULATION: trig_counts,
            "simulated_samples": simulated_samples,
            "triggered_samples": triggered_samples,
            _BIN_EDGES_KEY: bin_edges,
        }
    if not counts_by_label:
        _logger.warning(f"Skipping distribution plot for {quantity_name}, no data available.")
        return None
    statistics = _build_quantity_distribution_statistics(
        metrics_per_production,
        quantity_name,
        counts_by_label,
        cumulative=cumulative,
    )
    cum_str = _CUMULATIVE_SUFFIX if cumulative else ""
    statistics[_PLOT_NAME_KEY] = f"distribution_{quantity_name}{cum_str}{suffix}"
    if _COMPACT_FIGURES.get() is False:
        return statistics
    if not render:
        statistics["rendering"] = {"rendered": False, _REASON_KEY: "redundant_distribution"}
        return statistics
    fig, ax = plt.subplots(figsize=(9, 6))
    for label, counts in counts_by_label.items():
        _plot_distribution_series(
            ax,
            bin_edges,
            counts[_SIMULATED_POPULATION],
            label=f"{label} (simulated)",
            quantity_name=quantity_name,
            cumulative=cumulative,
            linewidth=1.5,
            linestyle="--",
        )
        _plot_distribution_series(
            ax,
            bin_edges,
            counts[_TRIGGERED_POPULATION],
            label=f"{label} (triggered)",
            quantity_name=quantity_name,
            cumulative=cumulative,
            linewidth=2,
        )

    cum_label = "Cumulative " if cumulative else ""
    type_label = f" ({suffix.lstrip('_').replace('_', ' ')})" if suffix else ""
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"{cum_label}Fraction of Events")
    ax.set_title(f"{cum_label}Distribution: {x_label}{type_label}")
    ax.set_xscale(x_scale)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    annotate_comparison_statistics(ax, statistics)

    _save_figure(
        fig, output_path, f"distribution_{quantity_name}{cum_str}{suffix}.png", figure_format
    )
    return statistics


def _get_global_quantity_bin_edges(metrics_per_production, quantity_name, x_scale, bins):
    """Return one shared set of bin edges for a quantity across productions."""
    histogram_edges = [
        metrics.quantity_histograms[quantity_name][event_kind][1]
        for metrics in metrics_per_production
        if quantity_name in metrics.quantity_histograms
        for event_kind in (_SIMULATED_POPULATION, _TRIGGERED_POPULATION)
        if event_kind in metrics.quantity_histograms[quantity_name]
    ]
    if histogram_edges:
        return np.unique(np.concatenate(histogram_edges))

    all_simulated_values = []
    for metrics in metrics_per_production:
        simulated, _ = _get_quantity_arrays(metrics, quantity_name)
        if simulated.size == 0:
            continue
        all_simulated_values.append(np.asarray(simulated))

    if len(all_simulated_values) == 0:
        return _get_bin_edges(np.array([]), x_scale=x_scale, bins=bins)
    return _get_bin_edges(np.concatenate(all_simulated_values), x_scale=x_scale, bins=bins)


def _get_quantity_counts(metrics, quantity_name, bin_edges, event_kind):
    """Return counts for a quantity and event kind from histogram-backed or raw metrics."""
    if quantity_name not in metrics.quantity_histograms:
        return None, None
    histogram_by_kind = metrics.quantity_histograms[quantity_name]
    if event_kind not in histogram_by_kind:
        return None, None
    counts, source_edges = histogram_by_kind[event_kind]
    counts = np.asarray(counts, dtype=float)
    source_edges = np.asarray(source_edges, dtype=float)
    if np.array_equal(source_edges, bin_edges):
        return counts, None
    return _rebin_histogram_counts(counts, source_edges, bin_edges), None


def _rebin_histogram_counts(counts, source_edges, target_edges):
    """Rebin counts by distributing each source bin over its overlapping target bins.

    Counts are assumed to be uniformly distributed within each source bin. This
    preserves the total count and avoids assigning a complete source bin to one
    target bin when the target edges split it.
    """
    rebinned = np.zeros(len(target_edges) - 1, dtype=float)
    for count, source_low, source_high in zip(counts, source_edges[:-1], source_edges[1:]):
        source_width = source_high - source_low
        target_start = np.searchsorted(target_edges, source_low, side="right") - 1
        target_stop = np.searchsorted(target_edges, source_high, side="left")
        for target_index in range(max(0, target_start), min(len(rebinned), target_stop + 1)):
            overlap = min(source_high, target_edges[target_index + 1]) - max(
                source_low, target_edges[target_index]
            )
            if overlap > 0:
                rebinned[target_index] += count * overlap / source_width
    return rebinned


def _plot_telescope_participation(metrics_per_production, output_path, figure_format=None):
    """Plot telescope participation fractions by production."""
    telescopes = sorted(
        {
            telescope
            for metrics in metrics_per_production
            for telescope in metrics.telescope_participation.keys()
        }
    )
    if len(telescopes) == 0:
        _logger.warning("Skipping telescope participation plot, no triggered events available.")
        return None

    statistics = _plot_grouped_fraction_bars(
        metrics_per_production,
        categories=telescopes,
        counts_getter=lambda metrics, category_labels: [
            metrics.telescope_participation.get(telescope, 0) for telescope in category_labels
        ],
        normalization_fn=_fractions_per_triggered_events,
        output_path=output_path,
        filename="telescope_participation_fraction.png",
        y_label="Participation Fraction",
        title="Telescope Participation in Triggered Events",
        figure_width=max(10, len(telescopes) * 0.4),
        x_rotation=90,
        x_ha="center",
        figure_format=figure_format,
    )
    if statistics is not None:
        statistics[_PLOT_NAME_KEY] = _PARTICIPATION_PLOT
    return statistics


def _record_telescope_type_participation(comparison_statistics, metrics_per_production):
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
        statistics = _build_aligned_count_statistics(
            metrics_per_production,
            counts_per_label,
            metadata={_CATEGORY_COUNT_KEY: len(categories)},
            metric=_JENSEN_SHANNON_METRIC,
        )
        statistics[_PLOT_NAME_KEY] = f"telescope_participation_fraction_{telescope_type}"
        _record_plot_statistics(comparison_statistics, statistics[_PLOT_NAME_KEY], statistics)


def _build_aligned_count_statistics(
    metrics_per_production,
    counts_per_label,
    metadata,
    metric=_JENSEN_SHANNON_METRIC,
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
        if metric == _WASSERSTEIN_METRIC and metadata.get(_BIN_EDGES_KEY) and result.get(_VALID):
            centers = 0.5 * (
                np.asarray(metadata[_BIN_EDGES_KEY], dtype=float)[:-1]
                + np.asarray(metadata[_BIN_EDGES_KEY], dtype=float)[1:]
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


def _build_quantity_distribution_statistics(
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
            _SIMULATED_POPULATION,
        )
        trig_stats = _compare_distribution_data(
            baseline_data,
            candidate_data,
            _TRIGGERED_POPULATION,
        )

        comparison_record = {
            "candidate_label": metrics.label,
            _SIMULATED_POPULATION: sim_stats,
            _TRIGGERED_POPULATION: trig_stats,
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
            baseline_data[_BIN_EDGES_KEY],
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
    result[_BIN_EDGES_KEY] = np.asarray(baseline_data[_BIN_EDGES_KEY], dtype=float).tolist()
    result["baseline_total"] = float(np.sum(baseline_data[event_kind]))
    result["candidate_total"] = float(np.sum(candidate_data[event_kind]))
    return result


def annotate_comparison_statistics(ax, statistics):
    """Overlay comparison metric values for baseline-vs-candidate comparisons on a plot."""
    if statistics is None or not statistics.get("comparisons"):
        return

    metric_type = statistics.get("metric_type")
    lines = [f"Baseline: {statistics.get('baseline_label', 'n/a')}"]
    for comparison in statistics["comparisons"]:
        label = comparison.get("candidate_label", "candidate")
        if metric_type == "quantity_distribution":
            simulated_label, simulated_value = _format_metric_value(
                comparison.get(_SIMULATED_POPULATION, {})
            )
            triggered_label, triggered_value = _format_metric_value(
                comparison.get(_TRIGGERED_POPULATION, {})
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

    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75, "edgecolor": "0.8"},
    )


def _format_statistic_value(value):
    """Format statistic values for plot text annotations."""
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _format_metric_value(statistics):
    """Return a display label and formatted value for one metric result."""
    metric_labels = {
        "ks": "KS",
        _JENSEN_SHANNON_METRIC: "JS",
        _WASSERSTEIN_METRIC: "W1",
    }
    metric = statistics.get("metric", "ks")
    value_key = {
        "ks": "ks_statistic",
        _JENSEN_SHANNON_METRIC: _JENSEN_SHANNON_DISTANCE,
        _WASSERSTEIN_METRIC: _WASSERSTEIN_DISTANCE,
    }.get(metric, VALUE_KEY)
    value = statistics.get(value_key, statistics.get(VALUE_KEY))
    return metric_labels.get(metric, metric), _format_statistic_value(value)


def _initialize_comparison_statistics(metrics_per_production):
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


def _record_plot_statistics(comparison_statistics, plot_name, statistics):
    """Store plot-level statistics if available."""
    if statistics is None:
        return
    comparison_statistics[PLOT_STATISTICS_KEY][plot_name] = statistics


def _plot_name_from_statistics(statistics):
    """Return plot name from statistics object or fallback placeholder."""
    return statistics.get(_PLOT_NAME_KEY, "unnamed_plot")


def _write_comparison_statistics_json(comparison_statistics, output_path):
    """Write comparison statistics JSON file in the output directory."""
    output_file = output_path / product_filename("comparison_statistics")
    _logger.info(f"Writing comparison statistics: {output_file}")
    ascii_handler.write_data_to_file(comparison_statistics, output_file, sort_keys=True)
    return output_file


def _distribution_cumulative_variants(quantity_name):
    """Return two cumulative slots, using None to disable one for specific quantities."""
    if quantity_name == ENERGY:
        return (False, None)
    return (False, True)


def _plot_series(
    ax,
    bin_edges,
    values,
    label,
    quantity_name,
    linewidth=2,
    linestyle="-",
    force_histogram=False,
):
    """Plot one series either as histogram stairs or as x/y line values."""
    if force_histogram or quantity_name in _HISTOGRAM_STYLE_QUANTITIES:
        return ax.stairs(values, bin_edges, linewidth=linewidth, linestyle=linestyle, label=label)
    x_values = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    (line,) = ax.plot(x_values, values, linewidth=linewidth, linestyle=linestyle, label=label)
    return line


def _normalized_histogram_values(counts, cumulative=False):
    """Return normalized bin values and Poisson errors for histogram counts."""
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        zeros = np.zeros_like(counts, dtype=float)
        return zeros, zeros
    if cumulative:
        values = np.cumsum(counts) / total
        errors = np.zeros_like(values, dtype=float)
        return values, errors
    values = counts / total
    errors = np.sqrt(counts) / total
    return values, errors


def _plot_distribution_series(
    ax,
    bin_edges,
    counts,
    label,
    quantity_name,
    cumulative,
    linewidth,
    linestyle="-",
):
    """Plot one normalized distribution series and optional Poisson error bars."""
    values, errors = _normalized_histogram_values(counts, cumulative=cumulative)
    artist = _plot_series(
        ax,
        bin_edges,
        values,
        label,
        quantity_name,
        linewidth=linewidth,
        linestyle=linestyle,
        force_histogram=True,
    )
    if not cumulative:
        _plot_histogram_error_bars(ax, bin_edges, values, errors, color=_artist_color(artist))


def _fraction_with_poisson_errors(counts, _metrics=None):
    """Return normalized bin fractions and Poisson errors for counts."""
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        return np.zeros_like(counts, dtype=float), np.zeros_like(counts, dtype=float)
    fractions = counts / total
    errors = np.sqrt(counts) / total
    return fractions, errors


def _artist_color(artist):
    """Return a representative color from a matplotlib artist."""
    if artist is None:
        return "black"
    if hasattr(artist, "get_edgecolor"):
        color = artist.get_edgecolor()
        if isinstance(color, np.ndarray):
            return color[0] if color.ndim > 1 else color
        return color
    if hasattr(artist, "get_color"):
        return artist.get_color()
    return "black"


def _plot_histogram_error_bars(ax, bin_edges, values, errors, color):
    """Overlay thin error bars on histogram-like series."""
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    ax.errorbar(
        centers,
        values,
        yerr=errors,
        fmt="none",
        ecolor=color,
        elinewidth=0.7,
        capsize=0,
        alpha=0.9,
    )


_QUANTITY_CONFIGS = [
    (ENERGY, "Primary Energy (TeV)", "log"),
    (CORE_DISTANCE, "Core Distance (m)", "linear"),
    (ANGULAR_DISTANCE, "Angular Distance (deg)", "linear"),
]

_HISTOGRAM_STYLE_QUANTITIES = {ENERGY}
_TRIGGERED_FRACTION_QUANTITIES = set()
_SPECIAL_TRIGGER_SUBSETS = {"single_telescope", "mixed_type"}

_TOP_LEVEL_STAT_PLOTS = [
    ("trigger_multiplicity", _plot_trigger_multiplicity),
    (_TRIGGER_COMBINATION, _plot_trigger_combinations),
    (_SINGLE_TELESCOPE_TRIGGER_DISTRIBUTION, _plot_single_telescope_trigger_frequencies),
    (_MIXED_TRIGGER_COMBINATIONS, _plot_mixed_trigger_combinations),
    (_PARTICIPATION_PLOT, _plot_telescope_participation),
]

_PER_TYPE_PLOT_FNS = [
    _plot_trigger_multiplicity,
    *[
        functools.partial(_plot_triggered_vs_quantity, quantity_name=q, x_label=lbl, x_scale=sc)
        for q, lbl, sc in _QUANTITY_CONFIGS
        if q in _TRIGGERED_FRACTION_QUANTITIES
    ],
    *[
        functools.partial(
            _plot_quantity_distribution, quantity_name=q, x_label=lbl, x_scale=sc, cumulative=cum
        )
        for q, lbl, sc in _QUANTITY_CONFIGS
        for cum in _distribution_cumulative_variants(q)
        if cum is not None
    ],
]
