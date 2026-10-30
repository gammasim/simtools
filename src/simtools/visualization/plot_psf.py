"""
Optical PSF plotting functions for parameter optimization visualization.

This module provides plotting functionality for PSF parameter optimization,
including parameter comparison plots, convergence plots, and PSF diameter vs off-axis plots.
"""

import logging
from pathlib import Path

import numpy as np

from simtools.visualization import visualize
from simtools.visualization.matplotlib_backend import pyplot as plt

# Constants
RADIUS = "Radius [cm]"
CUMULATIVE_PSF = "Cumulative PSF"
DEFAULT_FRACTION = 0.8  # Default PSF containment fraction

logger = logging.getLogger(__name__)


def get_psf_diameter_label(fraction, unit="cm"):
    """
    Generate PSF diameter label from the containment fraction, e.g. "D80 (cm)".

    Parameters
    ----------
    fraction : float
        PSF containment fraction (0.8 for D80).
    unit : str, optional
        Unit appended in parentheses (default: "cm").

    Returns
    -------
    str
        Label such as "D80 (cm)", or "D80" if unit is empty.
    """
    label = f"D{int(fraction * 100)}"
    return f"{label} ({unit})" if unit else label


def get_significance_label(p_value):
    """Get significance label for p-value: GOOD (>0.05), FAIR (>0.01) or POOR."""
    if p_value > 0.05:
        return "GOOD"
    if p_value > 0.01:
        return "FAIR"
    return "POOR"


def _format_metric_text(
    psf_diameter,
    metric,
    fraction=DEFAULT_FRACTION,
    p_value=None,
    use_ks_statistic=False,
    second_metric=None,
):
    text = f"{get_psf_diameter_label(fraction, unit='')} = {psf_diameter:.5f} cm"
    if second_metric is not None:
        text += f"\nRMSD = {metric:.4f}\nKS statistic = {second_metric:.4f}"
        if p_value is not None:
            text += f"\np-value = {p_value:.4f}"
    elif use_ks_statistic:
        text += f"\nKS stat = {metric:.6f}"
        if p_value is not None:
            text += f"\np-value = {p_value:.6f} ({get_significance_label(p_value)})"
    else:
        text += f"\nRMSD = {metric:.4f}"
    return text


def _create_base_plot_figure(data_to_plot, simulated_data=None):
    plot_data = dict(data_to_plot)
    if simulated_data is not None:
        plot_data["simulated"] = simulated_data
    fig = visualize.plot_1d(plot_data, plot_difference=True, no_markers=True)
    for difference_ax in fig.get_axes()[1:]:
        for line in difference_ax.lines:
            line.set_ydata(np.asarray(line.get_ydata()) * 100)
        difference_ax.set_ylabel("Difference\n[%]")
        difference_ax.relim()
        difference_ax.autoscale_view(scalex=False)
    ax = fig.get_axes()[0]
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(CUMULATIVE_PSF)
    return fig, ax


def _build_parameter_title(pars, is_best):
    lines = []
    if "mirror_reflection_random_angle" in pars:
        refl = pars["mirror_reflection_random_angle"]
        lines.append(f"reflection = {refl[0]:.5f}, {refl[1]:.5f}, {refl[2]:.5f}")
    for key, name in (
        ("mirror_align_random_vertical", "align_vertical"),
        ("mirror_align_random_horizontal", "align_horizontal"),
    ):
        if key in pars:
            lines.append(f"{name} = " + ", ".join(f"{x:.5f}" for x in pars[key]))
    return ("* " if is_best else "") + "\n".join(lines)


def _add_plot_annotations(
    ax,
    fig,
    pars,
    psf_diameter,
    metric,
    is_best,
    fraction=DEFAULT_FRACTION,
    p_value=None,
    use_ks_statistic=False,
    second_metric=None,
):
    ax.set_title(_build_parameter_title(pars, is_best))
    metrics_text = _format_metric_text(
        psf_diameter, metric, fraction, p_value, use_ks_statistic, second_metric
    )
    ax.text(
        0.5,
        0.3,
        metrics_text,
        verticalalignment="center",
        horizontalalignment="left",
        transform=ax.transAxes,
        color="red" if is_best else "black",
        weight="bold" if is_best else "normal",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "yellow", "alpha": 0.7}
        if is_best
        else None,
    )
    if is_best:
        fig.text(
            0.02,
            0.02,
            "* Best parameter set (lowest RMSD)",
            fontsize=8,
            style="italic",
            color="red",
        )


def create_psf_parameter_plot(
    data_to_plot,
    pars,
    psf_diameter,
    metric,
    is_best,
    pdf_pages,
    fraction=DEFAULT_FRACTION,
    p_value=None,
    use_ks_statistic=False,
    second_metric=None,
):
    """
    Add a cumulative PSF plot of one parameter set to a PDF report.

    Parameters
    ----------
    data_to_plot : dict
        Data dictionary for plotting.
    pars : dict
        Parameter set dictionary.
    psf_diameter, metric : float
        PSF diameter and metric (RMSD, or KS statistic if ``use_ks_statistic``).
    is_best : bool
        Whether this is the best parameter set.
    pdf_pages : PDF report object
        Open report for adding plots.
    fraction, p_value, use_ks_statistic, second_metric : optional
        PSF containment fraction, KS p-value, KS mode flag and second metric.
    """
    fig, ax = _create_base_plot_figure(data_to_plot)
    _add_plot_annotations(
        ax,
        fig,
        pars,
        psf_diameter,
        metric,
        is_best,
        fraction,
        p_value,
        use_ks_statistic,
        second_metric,
    )
    pdf_pages.add(fig)


def create_gradient_descent_convergence_plot(
    gd_results, threshold, output_file, fraction=DEFAULT_FRACTION, use_ks_statistic=False
):
    """
    Create convergence plot during gradient descent.

    Parameters
    ----------
    gd_results : list
        List of (params, metric, p_value, psf_diameter, simulated_data) tuples.
    threshold : float
        Optimization metric threshold used for convergence.
    output_file : Path
        Output file path for saving the plot.
    fraction : float, optional
        PSF containment fraction for labeling (default: 0.8).
    use_ks_statistic : bool
        Whether to use KS statistic or RMSD labels and titles.
    """
    logger.info("Creating gradient descent convergence plot...")
    has_p_values = len(gd_results[0]) >= 4 and gd_results[0][2] is not None
    show_p_panel = has_p_values and use_ks_statistic
    _, axes = plt.subplots(
        3 if show_p_panel else 2, 1, figsize=(10, 12 if show_p_panel else 8), tight_layout=True
    )

    iterations = list(range(len(gd_results)))
    metric_name = "KS Statistic" if use_ks_statistic else "RMSD"
    title = "Gradient Descent Convergence"
    axes[0].plot(iterations, [r[1] for r in gd_results], "b.-", linewidth=2, markersize=6)
    axes[0].axhline(y=threshold, color="r", linestyle="--", label=f"Threshold: {threshold}")
    axes[0].set(xlabel="Iteration", ylabel=metric_name, title=f"{title} - {metric_name}")
    axes[1].plot(iterations, [r[3] for r in gd_results], "g.-", linewidth=2, markersize=6)
    axes[1].set(
        xlabel="Iteration",
        ylabel=get_psf_diameter_label(fraction),
        title=f"{title} - {get_psf_diameter_label(fraction, unit='')}",
    )
    if show_p_panel:
        points = [(i, r[2]) for i, r in enumerate(gd_results) if r[2] is not None]
        axes[2].plot(
            [i for i, _ in points], [p for _, p in points], "m.-", linewidth=2, markersize=6
        )
        axes[2].axhline(
            y=0.05, color="orange", linestyle="--", alpha=0.7, label="p = 0.05 (significance)"
        )
        axes[2].axhline(
            y=0.01, color="r", linestyle="--", alpha=0.7, label="p = 0.01 (high significance)"
        )
        axes[2].set(
            xlabel="Iteration",
            ylabel="p-value",
            title=f"{title} - Statistical Significance",
            yscale="log",
        )
    for ax in axes:
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    if show_p_panel:
        axes[2].legend()

    visualize.save_figure(
        plt.gcf(),
        Path(output_file).with_suffix(""),
        figure_format=[Path(output_file).suffix[1:]],
        close=True,
    )
    logger.info(f"Convergence plot saved to {output_file}")


def _plot_moment_histogram(ax, values, mean, std, color, xlabel, title, fmt, unit=""):
    ax.hist(values, bins=20, alpha=0.7, color=color, edgecolor="black")
    ax.axvline(mean, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean:{fmt}}{unit}")
    ax.axvline(
        mean - std, color="orange", linestyle=":", alpha=0.7, label=f"$\\sigma$: {std:{fmt}}{unit}"
    )
    ax.axvline(mean + std, color="orange", linestyle=":", alpha=0.7)
    ax.set(xlabel=xlabel, ylabel="Counts", title=title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def create_monte_carlo_uncertainty_plot(
    mc_results, output_file, fraction=DEFAULT_FRACTION, use_ks_statistic=False
):
    """
    Create uncertainty analysis plots showing optimization metric and p-value distributions.

    Parameters
    ----------
    mc_results : tuple
        (mean_metric, std_metric, metric_values, mean_p, std_p, p_values,
        mean_psf_diameter, std_psf_diameter, psf_diameter_values) from the Monte Carlo analysis.
    output_file : Path
        Output file path; its suffix is replaced by ``_ks`` or ``_rmsd``.
    fraction : float, optional
        PSF containment fraction for labeling (default: 0.8).
    use_ks_statistic : bool, optional
        Whether KS statistic mode is being used (affects filename suffix).
    """
    (
        mean_metric,
        std_metric,
        metric_values,
        mean_p_value,
        _,
        p_values,
        mean_psf_diameter,
        std_psf_diameter,
        psf_diameter_values,
    ) = mc_results
    logger.info("Creating Monte Carlo uncertainty analysis plot...")

    valid_p_values = [p for p in p_values if p is not None] if p_values else []
    is_ks_mode = len(valid_p_values) > 0
    if is_ks_mode:
        _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10), tight_layout=True)
    else:
        _, (ax1, ax3) = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True)

    metric_name = "KS Statistic" if is_ks_mode else "RMSD"
    _plot_moment_histogram(
        ax1,
        metric_values,
        mean_metric,
        std_metric,
        "blue",
        metric_name,
        f"{metric_name} Distribution",
        ".6f",
    )
    _plot_moment_histogram(
        ax3,
        psf_diameter_values,
        mean_psf_diameter,
        std_psf_diameter,
        "green",
        get_psf_diameter_label(fraction),
        f"{get_psf_diameter_label(fraction, unit='')} Distribution",
        ".4f",
        unit=" cm",
    )

    if is_ks_mode:
        ax2.hist(valid_p_values, bins=20, alpha=0.7, color="magenta", edgecolor="black")
        if mean_p_value is not None:
            ax2.axvline(
                mean_p_value,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {mean_p_value:.6f}",
            )
        ax2.axvline(0.05, color="orange", linestyle="--", alpha=0.7, label="p = 0.05")
        ax2.axvline(0.01, color="red", linestyle="--", alpha=0.7, label="p = 0.01")
        ax2.set(xlabel="p-value", ylabel="Counts", title="p-value Distribution")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax4.scatter(metric_values, valid_p_values, alpha=0.6, color="purple")
        ax4.axhline(y=0.05, color="orange", linestyle="--", alpha=0.7, label="p = 0.05")
        ax4.axhline(y=0.01, color="red", linestyle="--", alpha=0.7, label="p = 0.01")
        ax4.set(
            xlabel=metric_name,
            ylabel="p-value",
            title=f"{metric_name} vs p-value Correlation",
            yscale="log",
        )
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    suffix = "_ks" if use_ks_statistic else "_rmsd"
    visualize.save_figure(
        plt, f"{output_file.with_suffix('')}{suffix}", log_title="Monte Carlo uncertainty", dpi=150
    )
    plt.close()


def plot_psf_histogram(measured, simulated, args_dict):
    """Write histogram comparing measured vs simulated PSF diameter distributions."""
    out_name = args_dict.get("psf_hist")
    if not out_name:
        return None
    out_path = Path(args_dict.get("output_path", ".")) / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    measured = np.asarray(measured, dtype=float)
    simulated = np.asarray(simulated, dtype=float)
    measured = measured[np.isfinite(measured)]
    simulated = simulated[np.isfinite(simulated)]
    if measured.size == 0 or simulated.size == 0:
        return None
    all_vals = np.concatenate([measured, simulated])
    x_min = float(np.nanmin(all_vals))
    x_max = float(np.nanmax(all_vals))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        return None
    bin_edges = np.linspace(x_min, x_max, 26)

    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    for values, name, color in (
        (measured, "Measured", "tab:red"),
        (simulated, "Simulated", "tab:blue"),
    ):
        mean = float(np.mean(values))
        rms = float(np.std(values, ddof=0))
        ax.hist(
            values,
            bins=bin_edges,
            alpha=0.55,
            color=color,
            edgecolor="white",
            label=f"{name} (mean={mean:.2f} mm, rms={rms:.2f} mm)",
        )
        ax.axvline(mean, color=color, linestyle="--", linewidth=1)

    tel = args_dict.get("telescope", "")
    model_version = args_dict.get("model_version", "")
    fraction = args_dict.get("fraction")
    label = get_psf_diameter_label(fraction, unit="mm") if fraction is not None else "PSF"
    suffix = " ".join([s for s in (tel, model_version) if s])
    ax.set(xlabel=label, ylabel="Count", title=f"{label} ({suffix})" if suffix else label)
    ax.legend(loc="best", fontsize=9, frameon=True)
    visualize.save_figure(
        fig, out_path.with_suffix(""), figure_format=[out_path.suffix[1:]], close=True
    )
    logger.info("PSF histogram written to %s", str(out_path))
    return str(out_path)
