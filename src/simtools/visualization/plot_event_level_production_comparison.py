"""Plot event-level comparisons across multiple simulation productions."""

import logging

import matplotlib.pyplot as plt
import numpy as np

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
    _plot_trigger_fraction(metrics_per_production, output_path)
    _plot_trigger_multiplicity(metrics_per_production, output_path)
    _plot_trigger_combinations(metrics_per_production, output_path)
    _plot_triggered_vs_quantity(
        metrics_per_production,
        output_path,
        quantity_name="energy",
        x_label="Primary Energy (TeV)",
        x_scale="log",
        bins=bins,
    )
    _plot_triggered_vs_quantity(
        metrics_per_production,
        output_path,
        quantity_name="core_distance",
        x_label="Core Distance (m)",
        x_scale="linear",
        bins=bins,
    )
    _plot_telescope_participation(metrics_per_production, output_path)


def _save_figure(fig, output_path, filename):
    """Save figure and close it."""
    output_file = output_path / filename
    _logger.info(f"Saving comparison plot: {output_file}")
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_trigger_fraction(metrics_per_production, output_path):
    """Plot triggered/simulated event fractions by production."""
    labels = [metrics.label for metrics in metrics_per_production]
    values = [metrics.trigger_fraction for metrics in metrics_per_production]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(labels, values, color="tab:blue", alpha=0.85)
    ax.set_ylabel("Trigger Fraction")
    ax.set_title("Trigger Fraction by Production")
    ax.set_ylim(0.0, max(1.0, max(values, default=0.0) * 1.15))
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    _save_figure(fig, output_path, "trigger_fraction_comparison.png")


def _plot_trigger_multiplicity(metrics_per_production, output_path):
    """Plot triggered telescope multiplicity distributions."""
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

    bins = np.arange(1, global_max + 2)
    for metrics in metrics_per_production:
        if metrics.trigger_multiplicity.size == 0:
            continue
        counts, _ = np.histogram(metrics.trigger_multiplicity, bins=bins)
        fractions = counts / counts.sum() if counts.sum() > 0 else counts
        ax.step(bins[:-1], fractions, where="post", linewidth=2, label=metrics.label)

    ax.set_xlabel("Triggered Telescopes per Event")
    ax.set_ylabel("Fraction of Triggered Events")
    ax.set_title("Trigger Multiplicity Distribution")
    ax.set_xticks(bins[:-1])
    ax.grid(alpha=0.25)
    ax.legend()

    _save_figure(fig, output_path, "trigger_multiplicity_comparison.png")


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

    x_values = np.arange(len(selected))
    width = 0.8 / max(1, len(metrics_per_production))
    fig, ax = plt.subplots(figsize=(max(10, len(selected) * 1.1), 6))

    for index, metrics in enumerate(metrics_per_production):
        event_norm = metrics.triggered_event_count if metrics.triggered_event_count > 0 else 1
        y_values = [metrics.trigger_combinations.get(name, 0) / event_norm for name in selected]
        offset = (index - (len(metrics_per_production) - 1) / 2.0) * width
        ax.bar(x_values + offset, y_values, width=width, label=metrics.label)

    ax.set_xticks(x_values)
    ax.set_xticklabels(selected, rotation=45, ha="right")
    ax.set_ylabel("Fraction of Triggered Events")
    ax.set_title("Top Trigger Combinations")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    _save_figure(fig, output_path, "trigger_combination_comparison.png")


def _plot_triggered_vs_quantity(
    metrics_per_production,
    output_path,
    quantity_name,
    x_label,
    x_scale,
    bins,
):
    """Plot simulated vs triggered distributions for one quantity."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for metrics in metrics_per_production:
        if quantity_name == "energy":
            simulated = metrics.simulated_energies
            triggered = metrics.triggered_energies
        else:
            simulated = metrics.simulated_core_distances
            triggered = metrics.triggered_core_distances

        if simulated.size == 0:
            continue

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

        x_values = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ax.plot(x_values, efficiency, linewidth=2, label=metrics.label)

    ax.set_xlabel(x_label)
    ax.set_ylabel("Triggered / Simulated")
    ax.set_title(f"Triggered Event Fraction vs {x_label}")
    ax.set_xscale(x_scale)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend()

    filename = f"triggered_fraction_vs_{quantity_name}.png"
    _save_figure(fig, output_path, filename)


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

    x_values = np.arange(len(telescopes))
    width = 0.8 / max(1, len(metrics_per_production))
    fig, ax = plt.subplots(figsize=(max(10, len(telescopes) * 0.4), 6))

    for index, metrics in enumerate(metrics_per_production):
        event_norm = metrics.triggered_event_count if metrics.triggered_event_count > 0 else 1
        fractions = [
            metrics.telescope_participation.get(telescope, 0) / event_norm
            for telescope in telescopes
        ]
        offset = (index - (len(metrics_per_production) - 1) / 2.0) * width
        ax.bar(x_values + offset, fractions, width=width, label=metrics.label)

    ax.set_xticks(x_values)
    ax.set_xticklabels(telescopes, rotation=90)
    ax.set_ylabel("Participation Fraction")
    ax.set_title("Telescope Participation in Triggered Events")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    _save_figure(fig, output_path, "telescope_participation_fraction.png")
