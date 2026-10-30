"""
Output of the PSF parameter optimisation.

Writes the gradient descent and Monte Carlo log files, the optimisation plots, the PSF
diameter versus off-axis angle plot, and the PDF reports.
"""

import logging
from contextlib import nullcontext

import astropy.units as u
import numpy as np

from simtools.ray_tracing.ray_tracing import RayTracing
from simtools.visualization import plot_psf, visualize
from simtools.visualization.matplotlib_backend import pyplot as plt
from simtools.visualization.plot_psf import DEFAULT_FRACTION, get_psf_diameter_label

logger = logging.getLogger(__name__)


MAX_OFFSET_DEFAULT = 4.5  # Maximum off-axis angle in degrees


OFFSET_STEPS_DEFAULT = 0.1  # Step size for off-axis angle sampling


def _format_log_value(value):
    """Format a parameter value for the log files."""
    if isinstance(value, list):
        return "[" + ", ".join([f"{v:.6f}" for v in value]) + "]"
    if isinstance(value, int | float):
        return f"{value:.6f}"
    return str(value)


def _log_header(title, tel_model, additional_info=None):
    """Return the commented header of a log file."""
    header_lines = [f"# {title}", f"# Telescope: {tel_model.name}"]
    for key, val in (additional_info or {}).items():
        header_lines.append(f"# {key}: {val}")
    header_lines.extend(["#" + "=" * 65, ""])
    return "\n".join(header_lines) + "\n"


def write_gradient_descent_log(
    gd_results,
    best_pars,
    best_psf_diameter,
    output_dir,
    tel_model,
    use_ks_statistic=False,
    fraction=DEFAULT_FRACTION,
):
    """
    Write the gradient descent progression to a log file.

    Parameters
    ----------
    gd_results : list
        (params, metric, p_value, psf_diameter, simulated_data) per iteration.
    best_pars : dict
        Best parameter values found.
    best_psf_diameter : float
        PSF containment diameter in cm for the best parameter set.
    output_dir : Path
        Directory where the log file is written.
    tel_model : TelescopeModel
        Telescope model used to name the output file.
    use_ks_statistic : bool, optional
        Log the KS statistic instead of the RMSD.
    fraction : float, optional
        PSF containment fraction for labeling.

    Returns
    -------
    Path
        Path to the created log file.
    """
    metric_name = "KS Statistic" if use_ks_statistic else "RMSD"
    file_suffix = "ks" if use_ks_statistic else "rmsd"
    label = get_psf_diameter_label(fraction)
    param_file = output_dir.joinpath(f"psf_gradient_descent_{file_suffix}_{tel_model.name}.log")
    total = len(gd_results)

    with open(param_file, "w", encoding="utf-8") as f:
        f.write(
            _log_header(
                f"PSF Parameter Optimization - Gradient Descent Progression ({metric_name})",
                tel_model,
                {"Total iterations": total},
            )
        )
        f.write(
            "GRADIENT DESCENT PROGRESSION:\n(Each entry shows the parameters chosen "
            "at each iteration)\n\n"
        )
        if use_ks_statistic:
            f.write(
                "P-VALUE INTERPRETATION:\n  p > 0.05: Distributions are statistically similar "
                "(good fit)\n"
                "  p < 0.05: Distributions are significantly different (poor fit)\n"
                "  p < 0.01: Very significant difference (very poor fit)\n\n"
            )
        else:
            f.write(
                "RMSD INTERPRETATION:\n  Lower RMSD values indicate better agreement between "
                "measured and simulated PSF curves\n\n"
            )

        for iteration, (pars, metric, p_value, psf_diameter, _) in enumerate(gd_results):
            status = "FINAL" if iteration == total - 1 else f"ITER-{iteration:02d}"
            if use_ks_statistic and p_value is not None:
                significance = plot_psf.get_significance_label(p_value)
                f.write(
                    f"[{status}] Iteration {iteration}: KS_stat={metric:.6f}, "
                    f"p_value={p_value:.6f} ({significance}), {label}={psf_diameter:.6f} cm\n"
                )
            else:
                f.write(
                    f"[{status}] Iteration {iteration}: {metric_name}={metric:.6f}, "
                    f"{label}={psf_diameter:.6f} cm\n"
                )
            for par, value in pars.items():
                f.write(f"    {par}: {_format_log_value(value)}\n")
            f.write("\n")

        best_metric = min(metric for _, metric, _, _, _ in gd_results)
        f.write("OPTIMIZATION SUMMARY:\n")
        f.write(f"Best {metric_name.lower()}: {best_metric:.6f}\n")
        f.write(
            f"Best {label}: {best_psf_diameter:.6f} cm\n"
            if best_psf_diameter is not None
            else f"Best {label}: N/A\n"
        )
        f.write(f"Total iterations: {total}\n\nFINAL OPTIMIZED PARAMETERS:\n")
        for par, value in best_pars.items():
            f.write(f"{par}: {_format_log_value(value)}\n")

    return param_file


def write_monte_carlo_analysis(
    mc_results, output_dir, tel_model, use_ks_statistic=False, fraction=DEFAULT_FRACTION
):
    """
    Write the Monte Carlo uncertainty analysis to a log file.

    Parameters
    ----------
    mc_results : tuple
        Results of ``PSFParameterOptimizer.analyze_monte_carlo_error``.
    output_dir : Path
        Directory where the log file is written.
    tel_model : TelescopeModel
        Telescope model used to name the output file.
    use_ks_statistic : bool, optional
        Analyze the KS statistic instead of the RMSD.
    fraction : float, optional
        PSF containment fraction for labeling.

    Returns
    -------
    Path
        Path to the created log file.
    """
    (
        mean_metric,
        std_metric,
        metric_values,
        mean_p_value,
        std_p_value,
        p_values,
        mean_psf_diameter,
        std_psf_diameter,
        psf_diameter_values,
    ) = mc_results

    metric_name = "KS Statistic" if use_ks_statistic else "RMSD"
    file_suffix = "ks" if use_ks_statistic else "rmsd"
    mc_file = output_dir.joinpath(f"monte_carlo_{file_suffix}_analysis_{tel_model.name}.log")
    psf_label = get_psf_diameter_label(fraction)

    with open(mc_file, "w", encoding="utf-8") as f:
        f.write(
            _log_header(
                f"Monte Carlo {metric_name} Error Analysis",
                tel_model,
                {"Number of simulations": len(metric_values)},
            )
        )
        f.write(
            f"MONTE CARLO SIMULATION RESULTS:\nNumber of successful simulations: "
            f"{len(metric_values)}\n\n"
        )
        f.write(f"{metric_name.upper()} STATISTICS:\n")
        f.write(
            f"Mean {metric_name.lower()}: {mean_metric:.6f}\n"
            f"Standard deviation: {std_metric:.6f}\n"
            f"Minimum {metric_name.lower()}: {min(metric_values):.6f}\n"
            f"Maximum {metric_name.lower()}: {max(metric_values):.6f}\n"
            f"Relative error: {(std_metric / mean_metric) * 100:.2f}%\n\n"
        )

        if use_ks_statistic and mean_p_value is not None:
            valid_p_values = [p for p in p_values if p is not None]
            n_valid = len(valid_p_values)
            good_fits = sum(1 for p in valid_p_values if p > 0.05)
            fair_fits = sum(1 for p in valid_p_values if 0.01 < p <= 0.05)
            poor_fits = sum(1 for p in valid_p_values if p <= 0.01)
            f.write(
                f"P-VALUE STATISTICS:\nMean p-value: {mean_p_value:.6f}\n"
                f"Standard deviation: {std_p_value:.6f}\n"
                f"Minimum p-value: {min(valid_p_values):.6f}\n"
                f"Maximum p-value: {max(valid_p_values):.6f}\n"
                f"Relative error: {(std_p_value / mean_p_value) * 100:.2f}%\n"
                f"Good fits (p > 0.05): {good_fits}/{n_valid} "
                f"({100 * good_fits / n_valid:.1f}%)\n"
                f"Fair fits (0.01 < p <= 0.05): {fair_fits}/{n_valid} "
                f"({100 * fair_fits / n_valid:.1f}%)\n"
                f"Poor fits (p <= 0.01): {poor_fits}/{n_valid} "
                f"({100 * poor_fits / n_valid:.1f}%)\n\n"
            )

        f.write(
            f"{psf_label} STATISTICS:\nMean {psf_label}: {mean_psf_diameter:.6f} cm\n"
            f"Standard deviation: {std_psf_diameter:.6f} cm\n"
            f"Minimum {psf_label}: {min(psf_diameter_values):.6f} cm\n"
            f"Maximum {psf_label}: {max(psf_diameter_values):.6f} cm\n"
            f"Relative error: {(std_psf_diameter / mean_psf_diameter) * 100:.2f}%\n\n"
        )

        f.write("INDIVIDUAL SIMULATION RESULTS:\n")
        for i, (metric_val, p_value, psf_diameter) in enumerate(
            zip(metric_values, p_values, psf_diameter_values)
        ):
            if use_ks_statistic and p_value is not None:
                significance = plot_psf.get_significance_label(p_value)
                f.write(
                    f"Simulation {i + 1:2d}: {metric_name}={metric_val:.6f}, "
                    f"p_value={p_value:.6f} ({significance}), {psf_label}={psf_diameter:.6f} cm\n"
                )
            else:
                f.write(
                    f"Simulation {i + 1:2d}: {metric_name}={metric_val:.6f}, "
                    f"{psf_label}={psf_diameter:.6f} cm\n"
                )

    return mc_file


def create_psf_vs_offaxis_plot(
    tel_model, site_model, args_dict, best_pars, output_dir, context=None
):
    """
    Plot the PSF diameter versus off-axis angle using the best parameters.

    Parameters
    ----------
    tel_model : TelescopeModel
        Telescope model object.
    site_model : SiteModel
        Site model object.
    args_dict : dict
        Parsed command-line arguments.
    best_pars : dict
        Best parameter set.
    output_dir : Path
        Output directory for plots.
    context : RunContext, optional
        Run context for ray tracing.
    """
    fraction = args_dict.get("fraction", DEFAULT_FRACTION)
    psf_label_cm = get_psf_diameter_label(fraction, unit="cm")
    psf_label_deg = get_psf_diameter_label(fraction, unit="degrees")
    logger.info(f"Creating {psf_label_cm} vs off-axis angle plot with best parameters...")

    tel_model.overwrite_parameters(best_pars, flat_dict=True)

    max_offset = args_dict.get("max_offset", MAX_OFFSET_DEFAULT * u.deg)
    offset_step = args_dict.get("offset_step", OFFSET_STEPS_DEFAULT * u.deg)
    max_offset_deg = (
        max_offset.to_value(u.deg) if isinstance(max_offset, u.Quantity) else max_offset
    )
    offset_step_deg = (
        offset_step.to_value(u.deg) if isinstance(offset_step, u.Quantity) else offset_step
    )
    if offset_step_deg <= 0:
        raise ValueError(f"offset_step must be positive, got {offset_step_deg} deg.")
    if max_offset_deg < 0:
        raise ValueError(f"max_offset must be non-negative, got {max_offset_deg} deg.")
    off_axis_angles = np.linspace(0, max_offset_deg, int(max_offset_deg / offset_step_deg) + 1)

    ray = RayTracing(
        telescope_model=tel_model,
        site_model=site_model,
        label=args_dict.get("label") or getattr(tel_model, "label", None),
        zenith_angle=args_dict["zenith_angle"],
        source_distance=args_dict["source_distance"],
        off_axis_angle=off_axis_angles * u.deg,
        context=context,
    )
    logger.info(f"Running ray tracing for {len(off_axis_angles)} off-axis angles...")
    ray.simulate(test=args_dict.get("test", False), force=True)
    ray.analyze(force=True, containment_fraction=fraction, keep_images=False)

    parameters_text = (
        f"Best Parameters: \n"
        f"align_horizontal=["
        f"{', '.join(f'{x:.4f}' for x in best_pars['mirror_align_random_horizontal'])}]\n"
        f"align_vertical=["
        f"{', '.join(f'{x:.4f}' for x in best_pars['mirror_align_random_vertical'])}]\n"
    )
    psf_identifier = get_psf_diameter_label(fraction, unit="").lower()  # e.g., "d80", "d95"
    for key in ["psf_cm", "psf_deg"]:
        plt.figure(figsize=(10, 6), tight_layout=True)
        ray.plot(key, marker="o", linestyle="-", color="blue", linewidth=2, markersize=6)
        plt.title(parameters_text)
        plt.xlabel("Off-axis Angle (degrees)")
        plt.ylabel(psf_label_cm if "_cm" in key else psf_label_deg)
        plt.ylim(bottom=0)
        plt.xticks(rotation=45)
        plt.xlim(0, max_offset_deg)
        plt.grid(True, alpha=0.3)

        unit_suffix = "cm" if "_cm" in key else "deg"
        plot_file = output_dir.joinpath(
            f"{tel_model.name}_best_params_{psf_identifier}_{unit_suffix}.png"
        )
        visualize.save_figure(plt, plot_file, log_title=f"{psf_label_cm} vs off-axis ({key})")

    plt.close("all")


def setup_pdf_plotting(args_dict, output_dir, tel_model_name):
    """
    Set up a PDF report for the gradient descent plots if ``plot_all`` is requested.

    Returns
    -------
    context manager
        PDF report context manager, or a null context if plotting is not requested.
    """
    if not args_dict.get("plot_all", False):
        return nullcontext()
    pdf_filename = output_dir / f"psf_gradient_descent_plots_{tel_model_name}.pdf"
    logger.info(f"Creating cumulative PSF plots for each iteration (saving to {pdf_filename})")
    return visualize.pdf_report(pdf_filename)


def create_optimization_plots(args_dict, gd_results, tel_model, data_to_plot, output_dir):
    """
    Create PSF plots for every fifth optimisation iteration and the final one.

    Parameters
    ----------
    args_dict : dict
        Command-line arguments; plots are created only if ``save_plots`` is set.
    gd_results : list
        (params, rmsd, _, psf_diameter, _) tuples from gradient descent.
    tel_model : TelescopeModel
        Telescope model object for naming files.
    data_to_plot : dict
        Measured PSF data.
    output_dir : Path
        Directory where the PDF file is saved.
    """
    if not args_dict.get("save_plots", False):
        return

    fraction = args_dict.get("fraction", DEFAULT_FRACTION)
    pdf_filename = output_dir.joinpath(f"psf_optimization_results_{tel_model.name}.pdf")
    logger.info(f"Creating PSF plots for each optimization iteration (saving to {pdf_filename})")
    with visualize.pdf_report(pdf_filename) as report:
        for i, (params, rmsd, _, psf_diameter, _) in enumerate(gd_results):
            if i % 5 == 0 or i == len(gd_results) - 1:
                plot_psf.create_psf_parameter_plot(
                    data_to_plot,
                    params,
                    psf_diameter,
                    rmsd,
                    is_best=(i == len(gd_results) - 1),
                    pdf_pages=report,
                    fraction=fraction,
                    use_ks_statistic=False,
                )
