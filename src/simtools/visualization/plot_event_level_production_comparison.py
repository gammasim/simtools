"""Plot event-level comparisons across multiple simulation productions."""

import functools
import logging

import matplotlib.pyplot as plt
import numpy as np

from simtools.utils import names

_logger = logging.getLogger(__name__)


def plot(metrics_per_production, output_path, bins=40):
    """Create all event-level production comparison plots.

    Parameters
    ----------
    metrics_per_production : list[ProductionEventMetrics]
        Aggregated metrics per production.
    output_path : pathlib.Path
        Output directory for generated figures.
    bins : int, optional
        Number of bins for 1D histograms.
    """
    _plot_trigger_multiplicity(metrics_per_production, output_path)
    _plot_trigger_combinations(metrics_per_production, output_path)
    _plot_single_telescope_trigger_frequencies(metrics_per_production, output_path)
    _plot_mixed_trigger_combinations(metrics_per_production, output_path)
    _plot_telescope_participation(metrics_per_production, output_path)

    for quantity_name, x_label, x_scale in _QUANTITY_CONFIGS:
        if quantity_name in _TRIGGERED_FRACTION_QUANTITIES:
            _plot_triggered_vs_quantity(
                metrics_per_production,
                output_path,
                quantity_name=quantity_name,
                x_label=x_label,
                x_scale=x_scale,
                bins=bins,
            )
        for cumulative in _distribution_cumulative_variants(quantity_name):
            if cumulative is None:
                continue
            _plot_quantity_distribution(
                metrics_per_production,
                output_path,
                quantity_name=quantity_name,
                x_label=x_label,
                x_scale=x_scale,
                bins=bins,
                cumulative=cumulative,
            )

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
            plot_fn(type_metrics, output_path, suffix=f"_{tel_type}", bins=bins)


def _save_figure(fig, output_path, filename):
    """Save figure and close it."""
    output_file = output_path / filename
    _logger.info(f"Saving comparison plot: {output_file}")
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_trigger_multiplicity(metrics_per_production, output_path, suffix="", bins=None):
    """Plot triggered telescope multiplicity distributions."""
    del bins
    fig, ax = plt.subplots(figsize=(9, 6))

    global_max = 0
    for metrics in metrics_per_production:
        if metrics.trigger_multiplicity.size == 0:
            continue
        global_max = max(global_max, int(np.max(metrics.trigger_multiplicity)))

    if global_max == 0:
        _logger.warning("Skipping trigger multiplicity plot, no triggered events available.")
        plt.close(fig)
        return

    bin_edges = np.arange(1, global_max + 2)
    for metrics in metrics_per_production:
        if metrics.trigger_multiplicity.size == 0:
            continue
        counts, _ = np.histogram(metrics.trigger_multiplicity, bins=bin_edges)
        fractions, errors = _fraction_with_poisson_errors(counts)
        stairs_artist = ax.stairs(fractions, bin_edges, linewidth=1.5, label=metrics.label)
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

    _save_figure(fig, output_path, f"trigger_multiplicity{suffix}.png")


def _plot_trigger_combinations(metrics_per_production, output_path, top_n=12):
    """Plot trigger combination distributions for each production."""
    combinations = set()
    for metrics in metrics_per_production:
        combinations.update(metrics.trigger_combinations.keys())

    if len(combinations) == 0:
        _logger.warning("Skipping trigger combination plot, no combinations available.")
        return

    totals = {}
    for combination in combinations:
        totals[combination] = sum(
            metrics.trigger_combinations.get(combination, 0) for metrics in metrics_per_production
        )

    selected = [name for name, _ in sorted(totals.items(), key=lambda item: item[1], reverse=True)]
    selected = selected[:top_n]

    _plot_grouped_fraction_bars(
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
    )


def _plot_single_telescope_trigger_frequencies(metrics_per_production, output_path):
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
        return

    _plot_grouped_fraction_bars(
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
    )


def _plot_mixed_trigger_combinations(metrics_per_production, output_path):
    """Plot mixed-type trigger combinations with multiplicity signatures and telescope names."""
    mixed_labels = sorted(
        {
            _format_mixed_combination_label(combination)
            for metrics in metrics_per_production
            for combination in metrics.trigger_combinations
            if _is_mixed_type_combination(combination)
        }
    )
    if len(mixed_labels) == 0:
        _logger.warning("Skipping mixed trigger combination plot, no mixed-type combinations.")
        return

    _plot_grouped_fraction_bars(
        metrics_per_production,
        categories=mixed_labels,
        counts_getter=lambda metrics, category_labels: [
            sum(
                count
                for combination, count in metrics.trigger_combinations.items()
                if _is_mixed_type_combination(combination)
                and _format_mixed_combination_label(combination) == label
            )
            for label in category_labels
        ],
        normalization_fn=_fraction_with_poisson_errors,
        output_path=output_path,
        filename="mixed_trigger_combinations.png",
        y_label="Fraction of Mixed-Type Triggers",
        title="Mixed-Type Trigger Combinations",
        figure_width=max(12, len(mixed_labels) * 0.8),
        x_rotation=45,
        x_ha="right",
    )


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
):
    """Plot grouped bars with fractions and Poisson error bars for each production."""
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

    _save_figure(fig, output_path, filename)


def _is_mixed_type_combination(combination):
    """Return True only for mixed signatures 1+1 and 1+2 (max multiplicity 3)."""
    type_counts, _ = _type_counts_from_combination(combination)
    if len(type_counts) < 2:
        return False
    signature = tuple(sorted(type_counts.values()))
    return signature in {(1, 1), (1, 2)}


def _format_mixed_combination_label(combination):
    """Format mixed trigger combination as '<signature> | <tel1> + <tel2> + ...'."""
    type_counts, telescope_names = _type_counts_from_combination(combination)
    signature = "+".join(str(count) for _, count in sorted(type_counts.items()))
    telescopes_label = " + ".join(telescope_names)
    return f"{signature} | {telescopes_label}"


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
):
    """Plot simulated vs triggered distributions for one quantity."""
    fig, ax = plt.subplots(figsize=(9, 6))

    plotted = False
    for metrics in metrics_per_production:
        simulated, triggered = _get_quantity_arrays(metrics, quantity_name)
        if simulated.size == 0:
            continue
        plotted = True

        bin_edges = _get_bin_edges(simulated, x_scale=x_scale, bins=bins)
        sim_counts, _ = np.histogram(simulated, bins=bin_edges)
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

    _save_figure(fig, output_path, f"triggered_fraction_vs_{quantity_name}{suffix}.png")


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
    if quantity_name == "energy":
        return metrics.simulated_energies, metrics.triggered_energies
    if quantity_name == "core_distance":
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
    fig, ax = plt.subplots(figsize=(9, 6))
    plotted = False
    for metrics in metrics_per_production:
        simulated, triggered = _get_quantity_arrays(metrics, quantity_name)
        if simulated.size == 0:
            continue
        plotted = True

        bin_edges = _get_bin_edges(simulated, x_scale=x_scale, bins=bins)
        sim_counts, _ = np.histogram(simulated, bins=bin_edges)
        trig_counts, _ = np.histogram(triggered, bins=bin_edges)
        _plot_distribution_series(
            ax,
            bin_edges,
            sim_counts,
            label=f"{metrics.label} (simulated)",
            quantity_name=quantity_name,
            cumulative=cumulative,
            linewidth=1.5,
            linestyle="--",
        )
        _plot_distribution_series(
            ax,
            bin_edges,
            trig_counts,
            label=f"{metrics.label} (triggered)",
            quantity_name=quantity_name,
            cumulative=cumulative,
            linewidth=2,
        )

    if not plotted:
        _logger.warning(f"Skipping distribution plot for {quantity_name}, no data available.")
        plt.close(fig)
        return
    cum_label = "Cumulative " if cumulative else ""
    type_label = f" ({suffix.lstrip('_').replace('_', ' ')})" if suffix else ""
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"{cum_label}Fraction of Events")
    ax.set_title(f"{cum_label}Distribution: {x_label}{type_label}")
    ax.set_xscale(x_scale)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    cum_str = "_cumulative" if cumulative else ""
    _save_figure(fig, output_path, f"distribution_{quantity_name}{cum_str}{suffix}.png")


def _plot_telescope_participation(metrics_per_production, output_path):
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
        return

    _plot_grouped_fraction_bars(
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
    )


def _distribution_cumulative_variants(quantity_name):
    """Return two cumulative slots, using None to disable one for specific quantities."""
    if quantity_name == "energy":
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
    ("energy", "Primary Energy (TeV)", "log"),
    ("core_distance", "Core Distance (m)", "linear"),
    ("angular_distance", "Angular Distance (deg)", "linear"),
]

_HISTOGRAM_STYLE_QUANTITIES = {"energy"}
_TRIGGERED_FRACTION_QUANTITIES = set()
_SPECIAL_TRIGGER_SUBSETS = {"single_telescope", "mixed_type"}

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
