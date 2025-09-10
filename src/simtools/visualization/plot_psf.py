"""
PSF plotting functions for parameter optimization visualization.

This module provides plotting functionality for PSF parameter optimization,
including parameter comparison plots, convergence plots, and D80 vs off-axis plots.
"""

import logging

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from simtools.ray_tracing.ray_tracing import RayTracing
from simtools.visualization import visualize

logger = logging.getLogger(__name__)

# Constants
RADIUS_CM = "Radius [cm]"
CUMULATIVE_PSF = "Cumulative PSF"
D80_CM_LABEL = "D80 (cm)"
MAX_OFFSET_DEFAULT = 4.5  # Maximum off-axis angle in degrees
OFFSET_STEPS_DEFAULT = 0.1  # Step size for off-axis angle sampling


def _get_significance_level(p_value):
    """Get significance level description for p-value."""
    if p_value > 0.05:
        return "GOOD"
    if p_value > 0.01:
        return "FAIR"
    return "POOR"


def _format_metric_text(d80, metric, p_value=None, use_ks_statistic=False, second_metric=None):
    """
    Format metric text for display in plots.

    Parameters
    ----------
    d80 : float
        D80 value
    metric : float
        Primary metric value (RMSD or KS statistic)
    p_value : float, optional
        P-value from KS test
    use_ks_statistic : bool
        If True, metric is KS statistic; if False, metric is RMSD
    second_metric : float, optional
        Second metric value to display alongside the primary metric

    Returns
    -------
    str
        Formatted metric text
    """
    d80_text = f"D80 = {d80:.5f} cm"

    # Create metric text based on the optimization method
    if second_metric is not None:
        # Special case: show both RMSD and KS statistic (for final best plot)
        metric_text = f"RMSD = {metric:.4f}"  # metric is RMSD in this case
        metric_text += f"\nKS statistic = {second_metric:.4f}"  # second_metric is KS statistic
        if p_value is not None:
            metric_text += f"\np-value = {p_value:.4f}"
    elif use_ks_statistic:
        metric_text = f"KS stat = {metric:.6f}"
        if p_value is not None:
            significance = _get_significance_level(p_value)
            metric_text += f"\np-value = {p_value:.6f} ({significance})"
    else:
        metric_text = f"RMSD = {metric:.4f}"

    return f"{d80_text}\n{metric_text}"


def _create_base_plot_figure(data_to_plot, simulated_data=None):
    """
    Create base figure for PSF parameter plots.

    Parameters
    ----------
    data_to_plot : dict
        Data dictionary for plotting
    simulated_data : array, optional
        Simulated data to add to the plot

    Returns
    -------
    tuple
        (fig, ax) - figure and axis objects
    """
    plot_data = data_to_plot.copy()

    if simulated_data is not None:
        plot_data["simulated"] = simulated_data

    fig = visualize.plot_1d(
        plot_data,
        plot_difference=True,
        no_markers=True,
    )
    ax = fig.get_axes()[0]
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(CUMULATIVE_PSF)
    return fig, ax


def _build_parameter_title(pars, is_best):
    """Build parameter title string for plots."""
    title_prefix = "* " if is_best else ""
    return (
        f"{title_prefix}reflection = "
        f"{pars['mirror_reflection_random_angle'][0]:.5f}, "
        f"{pars['mirror_reflection_random_angle'][1]:.5f}, "
        f"{pars['mirror_reflection_random_angle'][2]:.5f}\n"
        f"align_vertical = {pars['mirror_align_random_vertical'][0]:.5f}, "
        f"{pars['mirror_align_random_vertical'][1]:.5f}, "
        f"{pars['mirror_align_random_vertical'][2]:.5f}, "
        f"{pars['mirror_align_random_vertical'][3]:.5f}\n"
        f"align_horizontal = {pars['mirror_align_random_horizontal'][0]:.5f}, "
        f"{pars['mirror_align_random_horizontal'][1]:.5f}, "
        f"{pars['mirror_align_random_horizontal'][2]:.5f}, "
        f"{pars['mirror_align_random_horizontal'][3]:.5f}"
    )


def _add_metric_text_box(ax, metrics_text, is_best):
    """Add metric text box to plot."""
    d80_color = "red" if is_best else "black"
    d80_weight = "bold" if is_best else "normal"

    ax.text(
        0.5,
        0.3,
        metrics_text,
        verticalalignment="center",
        horizontalalignment="left",
        transform=ax.transAxes,
        color=d80_color,
        weight=d80_weight,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "yellow", "alpha": 0.7}
        if is_best
        else None,
    )


def _add_plot_annotations(
    ax, fig, pars, d80, metric, is_best, p_value=None, use_ks_statistic=False, second_metric=None
):
    """
    Add title, text annotations, and best parameter indicators to plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The plot axes
    fig : matplotlib.figure.Figure
        The plot figure
    pars : dict
        Parameter set dictionary
    d80 : float
        D80 value
    metric : float
        Primary metric value
    is_best : bool
        Whether this is the best parameter set
    p_value : float, optional
        P-value from KS test
    use_ks_statistic : bool
        If True, metric is KS statistic; if False, metric is RMSD
    second_metric : float, optional
        Second metric value to display
    """
    title = _build_parameter_title(pars, is_best)
    ax.set_title(title)

    metrics_text = _format_metric_text(d80, metric, p_value, use_ks_statistic, second_metric)
    _add_metric_text_box(ax, metrics_text, is_best)

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
    d80,
    metric,
    is_best,
    pdf_pages,
    p_value=None,
    use_ks_statistic=False,
    second_metric=None,
):
    """
    Create a plot for PSF simulation results.

    Parameters
    ----------
    data_to_plot : dict
        Data dictionary for plotting.
    pars : dict
        Parameter set dictionary.
    d80 : float
        D80 value.
    metric : float
        RMSD value (if use_ks_statistic=False) or KS statistic (if use_ks_statistic=True).
    is_best : bool
        Whether this is the best parameter set.
    pdf_pages : PdfPages
        PDF pages object for saving plots.
    p_value : float, optional
        P-value from KS test (only used when use_ks_statistic=True).
    use_ks_statistic : bool, optional
        If True, metric is KS statistic; if False, metric is RMSD.
    second_metric : float, optional
        Second metric value to display alongside the primary metric (for final best plot).
    """
    fig, ax = _create_base_plot_figure(data_to_plot)

    _add_plot_annotations(
        ax, fig, pars, d80, metric, is_best, p_value, use_ks_statistic, second_metric
    )

    pdf_pages.savefig(fig, bbox_inches="tight")
    plt.clf()


def create_detailed_parameter_plot(
    pars, ks_statistic, d80, simulated_data, data_to_plot, is_best, pdf_pages, p_value=None
):
    """
    Create a detailed plot for a parameter set showing all parameter values.

    Parameters
    ----------
    pars : dict
        Parameter set dictionary
    ks_statistic : float
        KS statistic value for this parameter set
    d80 : float
        D80 value for this parameter set
    simulated_data : array
        Simulated data for plotting
    data_to_plot : dict
        Data dictionary for plotting
    is_best : bool
        Whether this is the best parameter set
    pdf_pages : PdfPages
        PDF pages object to save the plot
    p_value : float, optional
        P-value from KS test for statistical significance
    """
    # Check if we have valid simulated data for plotting
    if simulated_data is None:
        logger.warning(
            "No simulated data available for plotting this parameter set, skipping plot creation"
        )
        return

    try:
        fig, ax = _create_base_plot_figure(data_to_plot, simulated_data)
    except (ValueError, RuntimeError, KeyError, TypeError) as e:
        logger.error(f"Failed to create plot for parameters: {e}")
        return

    _add_plot_annotations(
        ax,
        fig,
        pars,
        d80,
        ks_statistic,
        is_best,
        p_value,
        use_ks_statistic=True,
        second_metric=None,
    )

    pdf_pages.savefig(fig, bbox_inches="tight")
    plt.clf()


def create_parameter_progression_plots(results, best_pars, data_to_plot, pdf_pages):
    """
    Create plots for all parameter sets showing optimization progression.

    Parameters
    ----------
    results : list
        List of (pars, ks_statistic, p_value, d80, simulated_data) tuples
    best_pars : dict
        Best parameter set for highlighting
    data_to_plot : dict
        Data dictionary for plotting
    pdf_pages : PdfPages
        PDF pages object to save plots
    """
    logger.info("Creating plots for all parameter sets...")

    for i, (pars, ks_statistic, p_value, d80, simulated_data) in enumerate(results):
        if simulated_data is None:
            logger.warning(f"No simulated data for iteration {i}, skipping plot")
            continue

        is_best = pars is best_pars
        logger.info(f"Creating plot {i + 1}/{len(results)}{' (BEST)' if is_best else ''}")

        create_detailed_parameter_plot(
            pars, ks_statistic, d80, simulated_data, data_to_plot, is_best, pdf_pages, p_value
        )


def create_gradient_descent_convergence_plot(
    gd_results, threshold, output_file, use_ks_statistic=False
):
    """
    Create convergence plot showing optimization metric and D80 progression during gradient descent.

    Parameters
    ----------
    gd_results : list
        List of (params, metric, p_value, d80, simulated_data) tuples from gradient descent
    threshold : float
        Optimization metric threshold used for convergence
    output_file : Path
        Output file path for saving the plot
    use_ks_statistic : bool
        Whether to use KS statistic or RMSD labels and titles
    """
    logger.info("Creating gradient descent convergence plot...")

    # Check if results include p-values (for KS statistic mode)
    has_p_values = len(gd_results[0]) >= 4 and gd_results[0][2] is not None

    if has_p_values and use_ks_statistic:
        _, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), tight_layout=True)
    else:
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), tight_layout=True)

    iterations = list(range(len(gd_results)))
    metrics = [r[1] for r in gd_results]
    d80s = [r[3] for r in gd_results]

    metric_name = "KS Statistic" if use_ks_statistic else "RMSD"

    # Plot optimization metric progression
    ax1.plot(iterations, metrics, "b.-", linewidth=2, markersize=6)
    ax1.axhline(y=threshold, color="r", linestyle="--", label=f"Threshold: {threshold}")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel(metric_name)
    ax1.set_title(f"Gradient Descent Convergence - {metric_name}")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot D80 progression
    ax2.plot(iterations, d80s, "g.-", linewidth=2, markersize=6)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel(D80_CM_LABEL)
    ax2.set_title("Gradient Descent Convergence - D80")
    ax2.grid(True, alpha=0.3)

    # Plot p-value progression if available and using KS statistic
    if has_p_values and use_ks_statistic:
        p_values = [r[2] for r in gd_results if r[2] is not None]
        p_iterations = [i for i, r in enumerate(gd_results) if r[2] is not None]

        ax3.plot(p_iterations, p_values, "m.-", linewidth=2, markersize=6)
        ax3.axhline(
            y=0.05, color="orange", linestyle="--", alpha=0.7, label="p = 0.05 (significance)"
        )
        ax3.axhline(
            y=0.01, color="r", linestyle="--", alpha=0.7, label="p = 0.01 (high significance)"
        )
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("p-value")
        ax3.set_title("Gradient Descent Convergence - Statistical Significance")
        ax3.set_yscale("log")  # Log scale for p-values
        ax3.grid(True, alpha=0.3)
        ax3.legend()

    plt.savefig(output_file, bbox_inches="tight")
    plt.close()

    logger.info(f"Convergence plot saved to {output_file}")


def create_monte_carlo_uncertainty_plot(mc_results, output_file, use_ks_statistic=False):
    """
    Create uncertainty analysis plots showing optimization metric and p-value distributions.

    Parameters
    ----------
    mc_results : tuple
        Results from Monte Carlo analysis: (mean_metric, std_metric, metric_values,
        mean_p, std_p, p_values, mean_d80, std_d80, d80_values)
    output_file : Path
        Output file path for saving the plot
    use_ks_statistic : bool, optional
        Whether KS statistic mode is being used (affects filename suffix)
    """
    (
        mean_metric,
        std_metric,
        metric_values,
        mean_p_value,
        _,  # std_p_value (unused)
        p_values,
        mean_d80,
        std_d80,
        d80_values,
    ) = mc_results

    logger.info("Creating Monte Carlo uncertainty analysis plot...")

    # Check if we have valid p-values to determine if this is KS statistic or RMSD mode
    valid_p_values = [p for p in p_values if p is not None] if p_values else []
    is_ks_mode = len(valid_p_values) > 0

    # Create subplot layout based on mode
    if is_ks_mode:
        # KS mode: 2x2 layout with all 4 plots
        _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10), tight_layout=True)
    else:
        # RMSD mode: 1x2 layout with only metric and D80 plots
        _, (ax1, ax3) = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True)

    # Metric histogram (KS statistic or RMSD)
    metric_name = "KS Statistic" if is_ks_mode else "RMSD"
    ax1.hist(metric_values, bins=20, alpha=0.7, color="blue", edgecolor="black")
    ax1.axvline(
        mean_metric, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_metric:.6f}"
    )
    ax1.axvline(
        mean_metric - std_metric,
        color="orange",
        linestyle=":",
        alpha=0.7,
        label=f"$\\sigma$: {std_metric:.6f}",
    )
    ax1.axvline(mean_metric + std_metric, color="orange", linestyle=":", alpha=0.7)
    ax1.set_xlabel(metric_name)
    ax1.set_ylabel("Counts")
    ax1.set_title(f"{metric_name} Distribution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # p-value histogram (only for KS statistic mode)
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
        ax2.set_xlabel("p-value")
        ax2.set_ylabel("Counts")
        ax2.set_title("p-value Distribution")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # D80 histogram
    ax3.hist(d80_values, bins=20, alpha=0.7, color="green", edgecolor="black")
    ax3.axvline(
        mean_d80, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_d80:.4f} cm"
    )
    ax3.axvline(
        mean_d80 - std_d80,
        color="orange",
        linestyle=":",
        alpha=0.7,
        label=f"$\\sigma$: {std_d80:.4f} cm",
    )
    ax3.axvline(mean_d80 + std_d80, color="orange", linestyle=":", alpha=0.7)
    ax3.set_xlabel(D80_CM_LABEL)
    ax3.set_ylabel("Counts")
    ax3.set_title("D80 Distribution")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Scatter plot: Metric vs p-value (only for KS statistic mode)
    if is_ks_mode:
        ax4.scatter(metric_values, valid_p_values, alpha=0.6, color="purple")
        ax4.axhline(y=0.05, color="orange", linestyle="--", alpha=0.7, label="p = 0.05")
        ax4.axhline(y=0.01, color="red", linestyle="--", alpha=0.7, label="p = 0.01")
        ax4.set_xlabel(metric_name)
        ax4.set_ylabel("p-value")
        ax4.set_title(f"{metric_name} vs p-value Correlation")
        ax4.set_yscale("log")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    # Save plot in both PDF and PNG formats with appropriate suffix
    suffix = "_ks" if use_ks_statistic else "_rmsd"

    # Generate base filename without extension
    base_path = output_file.with_suffix("")
    base_name = str(base_path)

    # Add suffix and save in both formats
    pdf_file = f"{base_name}{suffix}.pdf"
    png_file = f"{base_name}{suffix}.png"

    plt.savefig(pdf_file, bbox_inches="tight")
    plt.savefig(png_file, bbox_inches="tight", dpi=150)
    plt.close()

    logger.info(f"Monte Carlo uncertainty plot saved to {pdf_file}")
    logger.info(f"Monte Carlo uncertainty plot saved to {png_file}")


def create_d80_vs_offaxis_plot(tel_model, site_model, args_dict, best_pars, output_dir):
    """
    Create D80 vs off-axis angle plot using the best parameters.

    Parameters
    ----------
    tel_model : TelescopeModel
        Telescope model object.
    site_model : SiteModel
        Site model object.
    args_dict : dict
        Dictionary containing parsed command-line arguments.
    best_pars : dict
        Best parameter set.
    output_dir : Path
        Output directory for saving plots.
    """
    logger.info("Creating D80 vs off-axis angle plot with best parameters...")

    # Apply best parameters to telescope model
    tel_model.change_multiple_parameters(**best_pars)

    # Create off-axis angle array
    max_offset = args_dict.get("max_offset", MAX_OFFSET_DEFAULT)
    offset_steps = args_dict.get("offset_steps", OFFSET_STEPS_DEFAULT)
    off_axis_angles = np.linspace(
        0,
        max_offset,
        int(max_offset / offset_steps) + 1,
    )

    ray = RayTracing(
        telescope_model=tel_model,
        site_model=site_model,
        simtel_path=args_dict["simtel_path"],
        zenith_angle=args_dict["zenith"] * u.deg,
        source_distance=args_dict["src_distance"] * u.km,
        off_axis_angle=off_axis_angles * u.deg,
    )

    logger.info(f"Running ray tracing for {len(off_axis_angles)} off-axis angles...")
    ray.simulate(test=args_dict.get("test", False), force=True)
    ray.analyze(force=True)

    for key in ["d80_cm", "d80_deg"]:
        plt.figure(figsize=(10, 6), tight_layout=True)

        ray.plot(key, marker="o", linestyle="-", color="blue", linewidth=2, markersize=6)

        plt.title(
            f"PSF D80 vs Off-axis Angle - {tel_model.name}\n"
            f"Best Parameters: \n"
            f"reflection=[{best_pars['mirror_reflection_random_angle'][0]:.4f}],\n"
            f"align_horizontal={best_pars['mirror_align_random_horizontal'][0]:.4f}\n"
            f"align_vertical={best_pars['mirror_align_random_vertical'][0]:.4f}\n"
        )
        plt.xlabel("Off-axis Angle (degrees)")
        plt.ylabel(D80_CM_LABEL if key == "d80_cm" else "D80 (degrees)")
        plt.ylim(bottom=0)
        plt.xticks(rotation=45)
        plt.xlim(0, max_offset)
        plt.grid(True, alpha=0.3)

        plot_file_name = f"tune_psf_{tel_model.name}_best_params_{key}.png"
        plot_file = output_dir.joinpath(plot_file_name)
        visualize.save_figure(
            plt, plot_file, figure_format=["png"], log_title=f"D80 vs off-axis ({key})"
        )

    plt.close("all")
