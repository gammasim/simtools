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
MAX_OFFSET_DEFAULT = 4.5  # Maximum off-axis angle in degrees
OFFSET_STEPS_DEFAULT = 0.1  # Step size for off-axis angle sampling


def create_psf_parameter_plot(data_to_plot, pars, d80, rmsd, is_best, pdf_pages):
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
    rmsd : float
        RMSD value.
    is_best : bool
        Whether this is the best parameter set.
    pdf_pages : PdfPages
        PDF pages object for saving plots.
    """
    fig = visualize.plot_1d(
        data_to_plot,
        plot_difference=True,
        no_markers=True,
    )
    ax = fig.get_axes()[0]
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(CUMULATIVE_PSF)

    title_prefix = "* " if is_best else ""
    ax.set_title(
        f"{title_prefix}refl_rnd = "
        f"{pars['mirror_reflection_random_angle'][0]:.5f}, "
        f"{pars['mirror_reflection_random_angle'][1]:.5f}, "
        f"{pars['mirror_reflection_random_angle'][2]:.5f}\n"
        f"align_rnd = {pars['mirror_align_random_vertical'][0]:.5f}, "
        f"{pars['mirror_align_random_vertical'][1]:.5f}, "
        f"{pars['mirror_align_random_vertical'][2]:.5f}, "
        f"{pars['mirror_align_random_vertical'][3]:.5f}"
    )

    d80_color = "red" if is_best else "black"
    d80_weight = "bold" if is_best else "normal"
    d80_text = f"D80 = {d80:.5f} cm"

    ax.text(
        0.5,
        0.3,
        f"{d80_text}\nRMSD = {rmsd:.4f}",
        verticalalignment="center",
        horizontalalignment="left",
        transform=ax.transAxes,
        color=d80_color,
        weight=d80_weight,
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

    pdf_pages.savefig(fig, bbox_inches="tight")
    plt.clf()


def create_detailed_parameter_plot(
    pars, rmsd, d80, simulated_data, data_to_plot, is_best, pdf_pages
):
    """
    Create a detailed plot for a parameter set showing all parameter values.

    Parameters
    ----------
    pars : dict
        Parameter set dictionary
    rmsd : float
        RMSD value for this parameter set
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
    """
    original_simulated = data_to_plot.get("simulated")

    # Check if we have valid simulated data for plotting
    if simulated_data is None:
        logger.warning(
            "No simulated data available for plotting this parameter set, skipping plot creation"
        )
        return

    data_to_plot["simulated"] = simulated_data

    try:
        fig = visualize.plot_1d(
            data_to_plot,
            plot_difference=True,
            no_markers=True,
        )
    except (ValueError, RuntimeError, KeyError, TypeError) as e:
        logger.error(f"Failed to create plot for parameters: {e}")
        # Restore original simulated data before returning
        if original_simulated is not None:
            data_to_plot["simulated"] = original_simulated
        elif "simulated" in data_to_plot:
            del data_to_plot["simulated"]
        return
    ax = fig.get_axes()[0]
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(CUMULATIVE_PSF)

    title_prefix = "* " if is_best else ""

    ax.set_title(
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

    d80_color = "red" if is_best else "black"
    d80_weight = "bold" if is_best else "normal"

    ax.text(
        0.5,
        0.3,
        f"D80 = {d80:.5f} cm\nRMSD = {rmsd:.4f}",
        verticalalignment="center",
        horizontalalignment="left",
        transform=ax.transAxes,
        color=d80_color,
        weight=d80_weight,
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

    pdf_pages.savefig(fig, bbox_inches="tight")
    plt.clf()

    if original_simulated is not None:
        data_to_plot["simulated"] = original_simulated


def create_parameter_progression_plots(results, best_pars, data_to_plot, pdf_pages):
    """
    Create plots for all parameter sets showing optimization progression.

    Parameters
    ----------
    results : list
        List of (pars, rmsd, d80, simulated_data) tuples
    best_pars : dict
        Best parameter set for highlighting
    data_to_plot : dict
        Data dictionary for plotting
    pdf_pages : PdfPages
        PDF pages object to save plots
    """
    logger.info("Creating plots for all parameter sets...")

    for i, (pars, rmsd, d80, simulated_data) in enumerate(results):
        if simulated_data is None:
            logger.warning(f"No simulated data for iteration {i}, skipping plot")
            continue

        is_best = pars is best_pars
        logger.info(f"Creating plot {i + 1}/{len(results)}{' (BEST)' if is_best else ''}")

        create_detailed_parameter_plot(
            pars, rmsd, d80, simulated_data, data_to_plot, is_best, pdf_pages
        )


def create_gradient_descent_convergence_plot(gd_results, rmsd_threshold, output_file):
    """
    Create convergence plot showing RMSD and D80 progression during gradient descent.

    Parameters
    ----------
    gd_results : list
        List of (params, rmsd, d80, simulated_data) tuples from gradient descent
    rmsd_threshold : float
        RMSD threshold used for convergence
    output_file : Path
        Output file path for saving the plot
    """
    logger.info("Creating gradient descent convergence plot...")

    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), tight_layout=True)

    iterations = list(range(len(gd_results)))
    rmsds = [r[1] for r in gd_results]
    d80s = [r[2] for r in gd_results]

    ax1.plot(iterations, rmsds, "b.-", linewidth=2, markersize=6)
    ax1.axhline(y=rmsd_threshold, color="r", linestyle="--", label=f"Threshold: {rmsd_threshold}")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("RMSD")
    ax1.set_title("Gradient Descent Convergence - RMSD")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(iterations, d80s, "g.-", linewidth=2, markersize=6)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("D80 (cm)")
    ax2.set_title("Gradient Descent Convergence - D80")
    ax2.grid(True, alpha=0.3)

    plt.savefig(output_file, bbox_inches="tight")
    plt.close()

    logger.info(f"Convergence plot saved to {output_file}")


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
        plt.ylabel("D80 (cm)" if key == "d80_cm" else "D80 (degrees)")
        plt.ylim(bottom=0)
        plt.xticks(rotation=45)
        plt.xlim(0, max_offset)
        plt.grid(True, alpha=0.3)

        plot_file_name = f"tune_psf_{tel_model.name}_best_params_{key}.pdf"
        plot_file = output_dir.joinpath(plot_file_name)
        visualize.save_figure(plt, plot_file, log_title=f"D80 vs off-axis ({key})")

    plt.close("all")
