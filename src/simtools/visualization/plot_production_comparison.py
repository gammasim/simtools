"""Plot event-level and signal-level comparisons across simulation productions."""

import logging
from collections import Counter
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from pathlib import Path

import numpy as np
from astropy import units as u

from simtools.context import RunContext
from simtools.histogram.histogram import compute_bin_edges, normalized_counts, rebin_counts
from simtools.io import ascii_handler
from simtools.io.file_type import product_filename
from simtools.sim_events.production_comparison_statistics import (
    ANGULAR_DISTANCE,
    BIN_EDGES_KEY,
    CATEGORY_COUNT_KEY,
    COMPACT_REPORT,
    CORE_DISTANCE,
    DIAGNOSTIC_REPORT,
    ENERGY,
    JENSEN_SHANNON_METRIC,
    PLOT_NAME_KEY,
    PLOT_STATISTICS_KEY,
    REASON_KEY,
    SIMULATED_POPULATION,
    TELESCOPE_TYPES_KEY,
    TRIGGERED_POPULATION,
    VALUE_KEY,
    WASSERSTEIN_METRIC,
    aggregate_mixed_trigger_combinations,
    build_aligned_count_statistics,
    build_quantity_distribution_statistics,
    compact_plot_statistics,
    comparison_statistics_annotation,
    fraction_with_poisson_errors,
    fractions_per_triggered_events,
    get_trigger_multiplicity_counts,
    initialize_comparison_statistics,
    record_plot_statistics,
    record_telescope_type_participation,
    write_comparison_statistics_json,
)
from simtools.statistics import compare_samples_with_statistics
from simtools.visualization.matplotlib_backend import pyplot as plt
from simtools.visualization.visualize import pdf_report, save_figure

_PARTICIPATION_PLOT = "telescope_participation_fraction"
_SPECIAL_TRIGGER_SUBSETS = {"single_telescope", "mixed_type"}
_QUANTITY_CONFIGS = [
    (ENERGY, "Primary Energy (TeV)", "log"),
    (CORE_DISTANCE, "Core Distance (m)", "linear"),
    (ANGULAR_DISTANCE, "Angular Distance (deg)", "linear"),
]
_QUANTITY_ARRAYS = {
    ENERGY: ("simulated_energies", "triggered_energies"),
    CORE_DISTANCE: ("simulated_core_distances", "triggered_core_distances"),
    ANGULAR_DISTANCE: ("simulated_angular_distances", "triggered_angular_distances"),
}
# Observable, x label, file name, comparison metric, logarithmic y axis.
_OBSERVABLES = (
    ("pedestals", "Pedestal", "pedestals.png", "ks", True),
    ("signals", "Integrated signal", "signals.png", "ks", True),
    ("peak_timing", "Peak sample", "peak_timing.png", "ks", False),
    (
        "triggered_pixels",
        "Triggered pixels per event",
        "triggered_pixels.png",
        "wasserstein",
        False,
    ),
)

_COMPACT_FIGURES = ContextVar("event_comparison_compact_figures", default=None)
_COMPARISON_TITLE = ContextVar("event_comparison_title", default=None)

_logger = logging.getLogger(__name__)


def plot_event_level_comparison(
    metrics_per_production,
    output_path=None,
    array_layout_name=None,
    bins=40,
    figure_format=None,
    report_mode=DIAGNOSTIC_REPORT,
    compact_figures=None,
    comparison_title=None,
    comparison_context=None,
    *,
    context=None,
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
    comparison_statistics = initialize_comparison_statistics(metrics_per_production)
    context = context or RunContext.from_environment()
    output_path = output_path or context.paths.get_output_directory()
    if array_layout_name:
        if not isinstance(array_layout_name, str):
            if len(array_layout_name) != 1:
                raise ValueError("Plot one array layout at a time.")
            array_layout_name = array_layout_name[0]
        output_path = output_path / array_layout_name
    output_path.mkdir(parents=True, exist_ok=True)
    suppressed = _metadata_suppressed_distributions(comparison_context)

    def render_top_level(quantity_name, cumulative):
        return _render_quantity(quantity_name, cumulative, report_mode, suppressed)

    with _comparison_report(output_path, report_mode, compact_figures, comparison_title):
        _record_plots(
            comparison_statistics,
            [
                plot(metrics_per_production, output_path, figure_format=figure_format)
                for plot in (
                    _plot_trigger_multiplicity,
                    _plot_trigger_combinations,
                    _plot_single_telescope_trigger_frequencies,
                    _plot_mixed_trigger_combinations,
                    _plot_telescope_participation,
                )
            ],
        )
        record_telescope_type_participation(comparison_statistics, metrics_per_production)
        _record_plots(
            comparison_statistics,
            _plot_distributions(
                metrics_per_production, output_path, bins, figure_format, render_top_level
            ),
        )
        _plot_per_type_comparisons(
            metrics_per_production,
            output_path,
            bins,
            comparison_statistics,
            figure_format,
            report_mode,
            suppressed,
        )

    participation = comparison_statistics[PLOT_STATISTICS_KEY].get(_PARTICIPATION_PLOT)
    if participation is not None:
        participation[TELESCOPE_TYPES_KEY] = sorted(
            {
                telescope
                for metrics in metrics_per_production
                for telescope in metrics.per_type
                if telescope not in _SPECIAL_TRIGGER_SUBSETS
            }
        )

    # Keep JSON scalar-only in every mode; detailed histograms are already in the plots.
    comparison_statistics[PLOT_STATISTICS_KEY] = compact_plot_statistics(
        comparison_statistics[PLOT_STATISTICS_KEY]
    )
    return write_comparison_statistics_json(comparison_statistics, output_path)


@contextmanager
def _comparison_report(output_path, report_mode, compact_figures, comparison_title):
    """Route figures to a compact PDF, the given report, or nowhere; set the figure title."""
    if report_mode == COMPACT_REPORT and compact_figures is None:
        report = pdf_report(output_path / "comparison-report.pdf")
    else:
        document = compact_figures if report_mode == COMPACT_REPORT else None
        report = nullcontext(False if report_mode == "none" else document)
    with report as document:
        title_token = _COMPARISON_TITLE.set(comparison_title)
        token = _COMPACT_FIGURES.set(document)
        try:
            yield
        finally:
            _COMPACT_FIGURES.reset(token)
            _COMPARISON_TITLE.reset(title_token)


def _record_plots(comparison_statistics, plot_statistics):
    """Record the statistics of each produced plot under its plot name."""
    for statistics in plot_statistics:
        if statistics is not None:
            record_plot_statistics(comparison_statistics, statistics[PLOT_NAME_KEY], statistics)


def _plot_distributions(
    metrics_per_production, output_path, bins, figure_format, render, suffix=""
):
    """Plot the distribution, and where defined the cumulative one, of each quantity.

    ``render(quantity_name, cumulative)`` decides whether a figure is drawn.
    """
    return [
        _plot_quantity_distribution(
            metrics_per_production,
            output_path,
            quantity_name,
            x_label,
            x_scale,
            bins=bins,
            suffix=suffix,
            cumulative=cumulative,
            figure_format=figure_format,
            render=render(quantity_name, cumulative),
        )
        for quantity_name, x_label, x_scale in _QUANTITY_CONFIGS
        for cumulative in ((False,) if quantity_name == ENERGY else (False, True))
    ]


def _render_quantity(quantity_name, cumulative, report_mode, suppressed):
    """Suppress redundant compact pages while retaining their numerical statistics."""
    if report_mode != COMPACT_REPORT:
        return True
    if quantity_name == CORE_DISTANCE and cumulative:
        return False
    return quantity_name not in suppressed


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
    """Return True if all finite metadata bounds agree within floating-point tolerance."""
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
    suppressed,
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
        suffix = f"_{tel_type}"
        plots = [
            _plot_trigger_multiplicity(
                type_metrics, output_path, suffix=suffix, figure_format=figure_format
            ),
            *_plot_distributions(
                type_metrics,
                output_path,
                bins,
                figure_format,
                lambda quantity_name, _cumulative: (
                    report_mode != COMPACT_REPORT or quantity_name not in suppressed
                ),
                suffix=suffix,
            ),
        ]
        _record_plots(comparison_statistics, plots)


def _save_figure(fig, output_path, filename, figure_format=None):
    """Save the figure, add it to the active compact report, or suppress it, then close it."""
    comparison_title = _COMPARISON_TITLE.get()
    if comparison_title:
        fig.suptitle(comparison_title, fontsize=10, fontweight="bold")
    document = _COMPACT_FIGURES.get()
    if document is False:
        plt.close(fig)
        return
    if document is not None:
        fig.set_label(Path(filename).stem)
        if hasattr(document, "add"):
            document.add(fig)
        else:
            document.append(fig)
        return
    save_figure(
        fig,
        output_path / Path(filename).with_suffix(""),
        figure_format=figure_format,
        dpi=300,
        close=True,
        pil_kwargs={"compress_level": 1},
    )


def _type_label(suffix):
    """Return a title suffix such as ' (LST)' for a per-telescope-type suffix."""
    return f" ({suffix.lstrip('_').replace('_', ' ')})" if suffix else ""


def _plot_stairs(ax, bin_edges, values, label, errors=None, **style):
    """Draw histogram stairs, with thin error bars in the stairs color when errors are given."""
    artist = ax.stairs(values, bin_edges, label=label, **style)
    if errors is not None:
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ax.errorbar(
            centers,
            values,
            yerr=errors,
            fmt="none",
            ecolor=_artist_color(artist),
            elinewidth=0.7,
            capsize=0,
            alpha=0.9,
        )


def _artist_color(artist):
    """Return a representative color from a matplotlib artist."""
    if artist is None:
        return "black"
    if hasattr(artist, "get_edgecolor"):
        color = artist.get_edgecolor()
        return color[0] if isinstance(color, np.ndarray) and color.ndim > 1 else color
    if hasattr(artist, "get_color"):
        return artist.get_color()
    return "black"


def _plot_trigger_multiplicity(metrics_per_production, output_path, suffix="", figure_format=None):
    """Plot triggered telescope multiplicity distributions."""
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
        counts = get_trigger_multiplicity_counts(metrics, bin_edges)
        if counts is not None:
            counts_per_label[metrics.label] = counts
    statistics = build_aligned_count_statistics(
        metrics_per_production,
        counts_per_label,
        metadata={
            BIN_EDGES_KEY: bin_edges.astype(float).tolist(),
            "bin_count": len(bin_edges) - 1,
        },
        metric=WASSERSTEIN_METRIC,
        bin_edges=bin_edges,
    )
    statistics[PLOT_NAME_KEY] = f"trigger_multiplicity{suffix}"
    if _COMPACT_FIGURES.get() is False:
        return statistics

    fig, ax = plt.subplots(figsize=(9, 6))
    for label, counts in counts_per_label.items():
        fractions, errors = fraction_with_poisson_errors(counts)
        _plot_stairs(ax, bin_edges, fractions, label, errors, linewidth=1.5)
    ax.set_xlabel("Triggered Telescopes per Event")
    ax.set_ylabel("Fraction of Triggered Events")
    ax.set_title(f"Trigger Multiplicity {_type_label(suffix)}")
    ax.set_xticks(bin_edges[:-1])
    ax.grid(alpha=0.25)
    ax.legend()
    annotate_comparison_statistics(ax, statistics)
    _save_figure(fig, output_path, f"trigger_multiplicity{suffix}.png", figure_format)
    return statistics


def _plot_trigger_combinations(metrics_per_production, output_path, top_n=12, figure_format=None):
    """Plot the most frequent trigger combinations for each production."""
    combinations = set()
    for metrics in metrics_per_production:
        combinations.update(metrics.trigger_combinations.keys())
    if not combinations:
        _logger.warning("Skipping trigger combination plot, no combinations available.")
        return None

    totals = {
        combination: sum(
            metrics.trigger_combinations.get(combination, 0) for metrics in metrics_per_production
        )
        for combination in combinations
    }
    ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    selected = [name for name, _ in ranked][:top_n]
    return _plot_grouped_fraction_bars(
        metrics_per_production,
        output_path,
        plot_name="trigger_combination",
        categories=selected,
        counts_of=lambda metrics: metrics.trigger_combinations,
        normalization_fn=fractions_per_triggered_events,
        y_label="Fraction of Triggered Events",
        title="Top Trigger Combinations",
        figure_width=max(10, len(selected) * 1.1),
        x_rotation=45,
        x_ha="right",
        statistics_categories=sorted(combinations),
        figure_format=figure_format,
    )


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
    if not telescope_names:
        _logger.warning("Skipping single-telescope trigger frequency plot, no data available.")
        return None
    return _plot_grouped_fraction_bars(
        metrics_per_production,
        output_path,
        plot_name="single_telescope_trigger_distribution",
        categories=telescope_names,
        counts_of=lambda metrics: metrics.trigger_combinations,
        normalization_fn=fraction_with_poisson_errors,
        y_label="Fraction of Single-Telescope Triggers",
        title="Single-Telescope Trigger Distribution",
        figure_width=max(10, len(telescope_names) * 0.45),
        x_rotation=90,
        x_ha="center",
        figure_format=figure_format,
    )


def _plot_mixed_trigger_combinations(
    metrics_per_production, output_path, top_n=12, figure_format=None
):
    """Plot mixed-type trigger combinations with multiplicity signatures and telescope names."""
    counts_by_production = {
        metrics.label: aggregate_mixed_trigger_combinations(metrics.trigger_combinations)
        for metrics in metrics_per_production
    }
    total_counts = Counter()
    for counts in counts_by_production.values():
        total_counts.update(counts)
    mixed_labels = [label for label, _ in total_counts.most_common(top_n)]
    if not mixed_labels:
        _logger.warning("Skipping mixed trigger combination plot, no mixed-type combinations.")
        return None
    return _plot_grouped_fraction_bars(
        metrics_per_production,
        output_path,
        plot_name="mixed_trigger_combinations",
        categories=mixed_labels,
        counts_of=lambda metrics: counts_by_production[metrics.label],
        normalization_fn=fraction_with_poisson_errors,
        y_label="Fraction of Mixed-Type Triggers",
        title="Mixed-Type Trigger Combinations",
        figure_width=max(12, len(mixed_labels) * 0.8),
        x_rotation=45,
        x_ha="right",
        statistics_categories=sorted(total_counts),
        figure_format=figure_format,
    )


def _plot_telescope_participation(metrics_per_production, output_path, figure_format=None):
    """Plot telescope participation fractions by production."""
    telescopes = sorted(
        {
            telescope
            for metrics in metrics_per_production
            for telescope in metrics.telescope_participation.keys()
        }
    )
    if not telescopes:
        _logger.warning("Skipping telescope participation plot, no triggered events available.")
        return None
    return _plot_grouped_fraction_bars(
        metrics_per_production,
        output_path,
        plot_name=_PARTICIPATION_PLOT,
        categories=telescopes,
        counts_of=lambda metrics: metrics.telescope_participation,
        normalization_fn=fractions_per_triggered_events,
        y_label="Participation Fraction",
        title="Telescope Participation in Triggered Events",
        figure_width=max(10, len(telescopes) * 0.4),
        x_rotation=90,
        x_ha="center",
        figure_format=figure_format,
    )


def _plot_grouped_fraction_bars(
    metrics_per_production,
    output_path,
    *,
    plot_name,
    categories,
    counts_of,
    normalization_fn,
    y_label,
    title,
    figure_width,
    x_rotation,
    x_ha,
    statistics_categories=None,
    figure_format=None,
):
    """Plot grouped bars with fractions and Poisson error bars for each production.

    ``counts_of(metrics)`` returns the category-to-count mapping of one production.
    """
    statistics_categories = statistics_categories or categories

    def counts(metrics, labels):
        return np.array([counts_of(metrics).get(label, 0) for label in labels], dtype=float)

    statistics = build_aligned_count_statistics(
        metrics_per_production,
        {
            metrics.label: counts(metrics, statistics_categories)
            for metrics in metrics_per_production
        },
        metadata={
            "categories": list(statistics_categories),
            CATEGORY_COUNT_KEY: len(statistics_categories),
            "display_categories": list(categories),
        },
        metric=JENSEN_SHANNON_METRIC,
    )
    statistics[PLOT_NAME_KEY] = plot_name
    if _COMPACT_FIGURES.get() is False:
        return statistics

    x_values = np.arange(len(categories))
    width = 0.8 / max(1, len(metrics_per_production))
    fig, ax = plt.subplots(figsize=(figure_width, 6))
    for index, metrics in enumerate(metrics_per_production):
        fractions, errors = normalization_fn(counts(metrics, categories), metrics)
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
    _save_figure(fig, output_path, f"{plot_name}.png", figure_format)
    return statistics


def _plot_quantity_stairs(ax, counts_by_label, bin_edges, cumulative):
    """Draw simulated and triggered step histograms for each production label."""
    for label, counts in counts_by_label.items():
        for population, text, linewidth, linestyle in (
            (SIMULATED_POPULATION, " (simulated)", 1.5, "--"),
            (TRIGGERED_POPULATION, " (triggered)", 2, "-"),
        ):
            values, errors = normalized_counts(counts[population], cumulative=cumulative)
            _plot_stairs(
                ax,
                bin_edges,
                values,
                f"{label}{text}",
                None if cumulative else errors,
                linewidth=linewidth,
                linestyle=linestyle,
            )


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
    """Plot simulated and triggered distributions of one quantity, optionally cumulative."""
    bin_edges = _get_global_quantity_bin_edges(metrics_per_production, quantity_name, x_scale, bins)
    counts_by_label = {}
    for metrics in metrics_per_production:
        simulated, triggered = _get_quantity_arrays(metrics, quantity_name)
        sim_histogram = _get_quantity_counts(
            metrics, quantity_name, bin_edges, SIMULATED_POPULATION
        )
        trig_histogram = _get_quantity_counts(
            metrics, quantity_name, bin_edges, TRIGGERED_POPULATION
        )
        if sim_histogram is None and simulated.size == 0:
            continue
        sim_counts, sim_samples = _counts_or_samples(sim_histogram, simulated, bin_edges)
        trig_counts, trig_samples = _counts_or_samples(trig_histogram, triggered, bin_edges)
        counts_by_label[metrics.label] = {
            SIMULATED_POPULATION: sim_counts,
            TRIGGERED_POPULATION: trig_counts,
            "simulated_samples": sim_samples,
            "triggered_samples": trig_samples,
            BIN_EDGES_KEY: bin_edges,
        }
    if not counts_by_label:
        _logger.warning(f"Skipping distribution plot for {quantity_name}, no data available.")
        return None

    statistics = build_quantity_distribution_statistics(
        metrics_per_production, quantity_name, counts_by_label, cumulative=cumulative
    )
    cum_str = "_cumulative" if cumulative else ""
    plot_name = f"distribution_{quantity_name}{cum_str}{suffix}"
    statistics[PLOT_NAME_KEY] = plot_name
    if _COMPACT_FIGURES.get() is False:
        return statistics
    if not render:
        statistics["rendering"] = {"rendered": False, REASON_KEY: "redundant_distribution"}
        return statistics

    fig, ax = plt.subplots(figsize=(9, 6))
    _plot_quantity_stairs(ax, counts_by_label, bin_edges, cumulative)
    cum_label = "Cumulative " if cumulative else ""
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"{cum_label}Fraction of Events")
    ax.set_title(f"{cum_label}Distribution: {x_label}{_type_label(suffix)}")
    ax.set_xscale(x_scale)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    annotate_comparison_statistics(ax, statistics)
    _save_figure(fig, output_path, f"{plot_name}.png", figure_format)
    return statistics


def _counts_or_samples(histogram_counts, samples, bin_edges):
    """Return histogram counts, or binned raw samples together with the samples."""
    if histogram_counts is not None:
        return histogram_counts, None
    return np.histogram(samples, bins=bin_edges)[0], samples


def _get_quantity_arrays(metrics, quantity_name):
    """Return the raw (simulated, triggered) arrays of a quantity."""
    simulated, triggered = _QUANTITY_ARRAYS[quantity_name]
    return getattr(metrics, simulated), getattr(metrics, triggered)


def _get_global_quantity_bin_edges(metrics_per_production, quantity_name, x_scale, bins):
    """Return one shared set of bin edges for a quantity across productions."""
    histogram_edges = [
        metrics.quantity_histograms[quantity_name][event_kind][1]
        for metrics in metrics_per_production
        if quantity_name in metrics.quantity_histograms
        for event_kind in (SIMULATED_POPULATION, TRIGGERED_POPULATION)
        if event_kind in metrics.quantity_histograms[quantity_name]
    ]
    if histogram_edges:
        return np.unique(np.concatenate(histogram_edges))

    simulated_values = []
    for metrics in metrics_per_production:
        simulated, _ = _get_quantity_arrays(metrics, quantity_name)
        if simulated.size > 0:
            simulated_values.append(np.asarray(simulated))
    if not simulated_values:
        return compute_bin_edges(np.array([]), scale=x_scale, bins=bins)
    return compute_bin_edges(np.concatenate(simulated_values), scale=x_scale, bins=bins)


def _get_quantity_counts(metrics, quantity_name, bin_edges, event_kind):
    """Return histogram counts rebinned to bin_edges, or None when no histogram exists."""
    histogram = metrics.quantity_histograms.get(quantity_name, {}).get(event_kind)
    if histogram is None:
        return None
    counts, source_edges = histogram
    counts = np.asarray(counts, dtype=float)
    source_edges = np.asarray(source_edges, dtype=float)
    if np.array_equal(source_edges, bin_edges):
        return counts
    return rebin_counts(counts, source_edges, bin_edges)


def annotate_comparison_statistics(ax, statistics):
    """Overlay comparison metric values for baseline-vs-candidate comparisons on a plot."""
    text = comparison_statistics_annotation(statistics)
    if text is None:
        return
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75, "edgecolor": "0.8"},
    )


def plot_signal_level_comparison(metrics_by_telescope, output_path, bins=40, figure_format=None):
    """Create signal comparison plots for every telescope in a layout.

    Parameters
    ----------
    metrics_by_telescope : dict[str, list[ProductionSignalMetrics]]
        Metrics grouped by telescope name and ordered by production.
    output_path : pathlib.Path
        Base output directory.
    bins : int, optional
        Number of histogram bins.
    figure_format : list[str] or str, optional
        Figure formats to write. Defaults to the configured formats or ``["png"]``.

    Returns
    -------
    list[pathlib.Path]
        Paths to the comparison statistics JSON files.
    """
    statistics_files = []
    for telescope_name, metrics in metrics_by_telescope.items():
        telescope_path = Path(output_path) / telescope_name
        telescope_path.mkdir(parents=True, exist_ok=True)
        comparison_statistics = _plot_telescope_comparisons(
            metrics, telescope_path, bins, figure_format
        )
        statistics_file = telescope_path / product_filename("comparison_statistics")
        ascii_handler.write_data_to_file(comparison_statistics, statistics_file, sort_keys=True)
        statistics_files.append(statistics_file)
    return statistics_files


def _plot_telescope_comparisons(metrics, output_path, bins, figure_format=None):
    """Plot all observables for one telescope and return their statistics."""
    if not metrics:
        raise ValueError("At least one production is required for signal comparison.")
    statistics = {
        "format_version": 1,
        "baseline": _production_summary(metrics[0]),
        "comparison_sets": [_production_summary(item) for item in metrics[1:]],
        "plot_statistics": {},
    }
    for spec in _OBSERVABLES:
        plot_statistics = _plot_observable(metrics, output_path, spec, bins, figure_format)
        if plot_statistics is not None:
            statistics["plot_statistics"][spec[0]] = plot_statistics
    return statistics


def _production_summary(metrics):
    """Return the common comparison summary for one telescope production."""
    return {
        "label": metrics.label,
        "simulated_event_count": int(metrics.triggered_pixels.size),
        "triggered_event_count": int(np.count_nonzero(metrics.triggered_pixels > 0)),
    }


def _plot_observable(metrics, output_path, spec, bins, figure_format=None):
    """Plot one signal observable as normalized histograms and compare the productions."""
    observable, x_label, filename, metric, log_y = spec
    samples = [np.asarray(getattr(item, observable)) for item in metrics]
    non_empty = [values for values in samples if values.size]
    if not non_empty:
        _logger.warning("Skipping %s plot because no values are available.", observable)
        return None

    bin_edges = np.histogram_bin_edges(np.concatenate(non_empty), bins=bins)
    fig, ax = plt.subplots(figsize=(8, 5))
    for item, values in zip(metrics, samples):
        if values.size:
            counts, _ = np.histogram(values, bins=bin_edges)
            ax.stairs(counts / counts.sum(), bin_edges, label=item.label)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Fraction")
    ax.set_title(f"{x_label} comparison")
    if log_y:
        ax.set_yscale("log")
    ax.grid(alpha=0.25)

    comparisons = []
    if samples[0].size:
        for item, values in zip(metrics[1:], samples[1:]):
            if values.size:
                comparison = compare_samples_with_statistics(
                    samples[0], values, bin_edges, metric=metric
                )
                comparison["candidate_label"] = item.label
                comparisons.append(comparison)
    statistics = {
        "baseline_label": metrics[0].label,
        "metric_type": "sample_distribution",
        "metric": metric,
        "metadata": {"bin_edges": bin_edges.tolist()},
        "comparisons": comparisons,
    }
    annotate_comparison_statistics(ax, statistics)
    ax.legend()
    _save_figure(fig, Path(output_path), filename, figure_format)
    return statistics
