"""
PSF parameter optimisation and fitting routines for mirror alignment and reflection parameters.

This module provides functions for loading PSF data, generating random parameter sets,
running PSF simulations, calculating RMSD, and finding the best-fit parameters for a given
telescope model.
PSF (Point Spread Function) describes how a point source of light is spread out by the
optical system, and RMSD (Root Mean Squared Deviation) is used as the optimization metric
to quantify the difference between measured and simulated PSF curves.
"""

import logging
from collections import OrderedDict

import astropy.units as u
import numpy as np
from astropy.table import Table
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

from simtools.data_model import model_data_writer as writer
from simtools.ray_tracing.ray_tracing import RayTracing
from simtools.utils import general as gen
from simtools.visualization import plot_psf

logger = logging.getLogger(__name__)

# Constants
RADIUS_CM = "Radius [cm]"
CUMULATIVE_PSF = "Cumulative PSF"
KS_STATISTIC_NAME = "KS statistic"


def _create_log_header_and_format_value(title, tel_model, additional_info=None, value=None):
    """Create log header and format parameter values."""
    if value is not None:  # Format value mode
        if isinstance(value, list):
            return "[" + ", ".join([f"{v:.6f}" for v in value]) + "]"
        if isinstance(value, int | float):
            return f"{value:.6f}"
        return str(value)

    # Create header mode
    header_lines = [f"# {title}", f"# Telescope: {tel_model.name}"]
    if additional_info:
        for key, val in additional_info.items():
            header_lines.append(f"# {key}: {val}")
    header_lines.extend(["#" + "=" * 60, ""])
    return "\n".join(header_lines) + "\n"


def calculate_rmsd(data, sim):
    """Calculate RMSD between measured and simulated cumulative PSF curves."""
    return np.sqrt(np.mean((data - sim) ** 2))


def calculate_ks_statistic(data, sim):
    """Calculate the KS statistic between measured and simulated cumulative PSF curves."""
    return stats.ks_2samp(data, sim, method="asymp")


def get_previous_values(tel_model):
    """Retrieve previous parameter values from the telescope model."""
    return {
        "mirror_reflection_random_angle": tel_model.get_parameter_value(
            "mirror_reflection_random_angle"
        ),
        "mirror_align_random_horizontal": tel_model.get_parameter_value(
            "mirror_align_random_horizontal"
        ),
        "mirror_align_random_vertical": tel_model.get_parameter_value(
            "mirror_align_random_vertical"
        ),
    }


def _run_ray_tracing_simulation(tel_model, site_model, args_dict, pars):
    """Run a ray tracing simulation with the given telescope parameters."""
    if pars is None:
        raise ValueError("No best parameters found")

    tel_model.change_multiple_parameters(**pars)
    ray = RayTracing(
        telescope_model=tel_model,
        site_model=site_model,
        simtel_path=args_dict["simtel_path"],
        zenith_angle=args_dict["zenith"] * u.deg,
        source_distance=args_dict["src_distance"] * u.km,
        off_axis_angle=[0.0] * u.deg,
    )
    ray.simulate(test=args_dict.get("test", False), force=True)
    ray.analyze(force=True, use_rx=False)
    im = ray.images()[0]
    return im.get_psf(), im


def run_psf_simulation(
    tel_model,
    site_model,
    args_dict,
    pars,
    data_to_plot,
    radius,
    pdf_pages=None,
    is_best=False,
    use_ks_statistic=False,
):
    """Run simulation for one parameter set and return D80, metric, p_value, and simulated data."""
    d80, im = _run_ray_tracing_simulation(tel_model, site_model, args_dict, pars)

    if radius is None:
        raise ValueError("Radius data is not available.")

    simulated_data = im.get_cumulative_data(radius * u.cm)

    if use_ks_statistic:
        ks_statistic, p_value = calculate_ks_statistic(
            data_to_plot["measured"][CUMULATIVE_PSF], simulated_data[CUMULATIVE_PSF]
        )
        metric = ks_statistic
    else:
        metric = calculate_rmsd(
            data_to_plot["measured"][CUMULATIVE_PSF], simulated_data[CUMULATIVE_PSF]
        )
        p_value = None

    # Handle plotting if requested
    if pdf_pages is not None and args_dict.get("plot_all", False):
        data_to_plot["simulated"] = simulated_data
        plot_psf.create_psf_parameter_plot(
            data_to_plot,
            pars,
            d80,
            metric,
            is_best,
            pdf_pages,
            p_value=p_value,
            use_ks_statistic=use_ks_statistic,
        )
        del data_to_plot["simulated"]

    return d80, metric, p_value, simulated_data


def load_and_process_data(args_dict):
    """Load and process data if specified in the command-line arguments."""
    if args_dict["data"] is None:
        raise FileNotFoundError("No data file specified for PSF optimization.")

    data_file = gen.find_file(args_dict["data"], args_dict["model_path"])
    table = Table.read(data_file, format="ascii.ecsv")

    d_type = {"names": (RADIUS_CM, CUMULATIVE_PSF), "formats": ("f8", "f8")}
    data = np.zeros(len(table), dtype=d_type)

    data[RADIUS_CM] = table["radius_mm"].quantity.to(u.cm).value
    data[CUMULATIVE_PSF] = table["cumulative_psf"]
    data[CUMULATIVE_PSF] /= np.max(np.abs(data[CUMULATIVE_PSF]))  # Normalize to max = 1.0

    return OrderedDict([("measured", data)]), data[RADIUS_CM]


def write_tested_parameters_to_file(results, best_pars, best_d80, output_dir, tel_model):
    """Write all tested parameters and their metrics to a text file."""
    param_file = output_dir.joinpath(f"psf_optimization_{tel_model.name}.log")
    with open(param_file, "w", encoding="utf-8") as f:
        header = _create_log_header_and_format_value(
            "PSF Parameter Optimization Log",
            tel_model,
            {"Total parameter sets tested": len(results)},
        )
        f.write(header)

        f.write("PARAMETER TESTING RESULTS:\n")
        for i, (pars, ks_statistic, p_value, d80, _) in enumerate(results):
            status = "BEST" if pars is best_pars else "TESTED"
            f.write(
                f"[{status}] Set {i + 1:03d}: KS_stat={ks_statistic:.5f}, "
                f"p_value={p_value:.5f}, D80={d80:.5f} cm\n"
            )
            for par, value in pars.items():
                f.write(f"    {par}: {value}\n")
            f.write("\n")

        f.write("OPTIMIZATION SUMMARY:\n")
        f.write(f"Best KS statistic: {min(result[1] for result in results):.5f}\n")
        f.write(f"Best D80: {best_d80:.5f} cm\n")
        f.write("\nOPTIMIZED PARAMETERS:\n")
        for par, value in best_pars.items():
            f.write(f"{par}: {value}\n")
    return param_file


def _add_units_to_psf_parameters(best_pars):
    """Add proper astropy units to PSF parameters based on their schemas."""
    psf_pars_with_units = {}
    for param_name, param_values in best_pars.items():
        if param_name == "mirror_reflection_random_angle":
            psf_pars_with_units[param_name] = [
                param_values[0] * u.deg,
                param_values[1] * u.dimensionless_unscaled,
                param_values[2] * u.deg,
            ]
        elif param_name in ["mirror_align_random_horizontal", "mirror_align_random_vertical"]:
            psf_pars_with_units[param_name] = [
                param_values[0] * u.deg,
                param_values[1] * u.deg,
                param_values[2] * u.dimensionless_unscaled,
                param_values[3] * u.dimensionless_unscaled,
            ]
        else:
            psf_pars_with_units[param_name] = param_values
    return psf_pars_with_units


def export_psf_parameters(best_pars, telescope, parameter_version, output_dir):
    """Export PSF parameters as simulation model parameter files."""
    try:
        psf_pars_with_units = _add_units_to_psf_parameters(best_pars)
        parameter_output_path = output_dir / telescope
        for parameter_name, parameter_value in psf_pars_with_units.items():
            writer.ModelDataWriter.dump_model_parameter(
                parameter_name=parameter_name,
                value=parameter_value,
                instrument=telescope,
                parameter_version=parameter_version,
                output_file=f"{parameter_name}-{parameter_version}.json",
                output_path=parameter_output_path,
                use_plain_output_path=True,
            )
        logger.info(f"simulation model parameter files exported to {output_dir}")

    except (ValueError, KeyError, OSError) as e:
        logger.error(f"Error exporting simulation parameters: {e}")


def _calculate_param_gradient(
    tel_model,
    site_model,
    args_dict,
    current_params,
    data_to_plot,
    radius,
    current_rmsd,
    param_name,
    param_values,
    epsilon,
    use_ks_statistic,
):
    """Calculate gradient for a single parameter."""
    param_gradients = []
    values_list = param_values if isinstance(param_values, list) else [param_values]

    for i, value in enumerate(values_list):
        perturbed_params = {
            k: v.copy() if isinstance(v, list) else v for k, v in current_params.items()
        }

        if isinstance(param_values, list):
            perturbed_params[param_name][i] = value + epsilon
        else:
            perturbed_params[param_name] = value + epsilon

        try:
            _, perturbed_rmsd, _, _ = run_psf_simulation(
                tel_model,
                site_model,
                args_dict,
                perturbed_params,
                data_to_plot,
                radius,
                pdf_pages=None,
                is_best=False,
                use_ks_statistic=use_ks_statistic,
            )
            param_gradients.append((perturbed_rmsd - current_rmsd) / epsilon)
        except (ValueError, RuntimeError):
            param_gradients.append(0.0)

    return param_gradients[0] if not isinstance(param_values, list) else param_gradients


def calculate_gradient(
    tel_model,
    site_model,
    args_dict,
    current_params,
    data_to_plot,
    radius,
    current_rmsd,
    epsilon=0.0005,
    use_ks_statistic=False,
):
    """Calculate numerical gradient of RMSD with respect to parameters."""
    gradients = {}

    for param_name, param_values in current_params.items():
        gradients[param_name] = _calculate_param_gradient(
            tel_model,
            site_model,
            args_dict,
            current_params,
            data_to_plot,
            radius,
            current_rmsd,
            param_name,
            param_values,
            epsilon,
            use_ks_statistic,
        )

    return gradients


def apply_gradient_step(current_params, gradients, learning_rate):
    """Move parameters in the direction of negative gradient."""
    new_params = {}
    for param_name, param_values in current_params.items():
        param_gradients = gradients[param_name]

        if isinstance(param_values, list):
            new_params[param_name] = [
                value - learning_rate * gradient
                for value, gradient in zip(param_values, param_gradients)
            ]
        else:
            new_params[param_name] = param_values - learning_rate * param_gradients

    return new_params


def _perform_gradient_step_with_retries(
    tel_model,
    site_model,
    args_dict,
    current_params,
    current_metric,
    data_to_plot,
    radius,
    learning_rate,
    max_retries=3,
):
    """Try gradient step with learning rate reduction on rejection."""
    current_lr = learning_rate

    for attempt in range(max_retries):
        try:
            gradients = calculate_gradient(
                tel_model,
                site_model,
                args_dict,
                current_params,
                data_to_plot,
                radius,
                current_metric,
                use_ks_statistic=False,
            )
            new_params = apply_gradient_step(current_params, gradients, current_lr)

            new_d80, new_metric, new_p_value, new_simulated_data = run_psf_simulation(
                tel_model,
                site_model,
                args_dict,
                new_params,
                data_to_plot,
                radius,
                pdf_pages=None,
                is_best=False,
                use_ks_statistic=False,
            )

            if new_metric < current_metric:
                return (
                    new_params,
                    new_d80,
                    new_metric,
                    new_p_value,
                    new_simulated_data,
                    True,
                    current_lr,
                )

            logger.info(
                f"Step rejected (RMSD {current_metric:.6f} -> {new_metric:.6f}), "
                f"reducing learning rate {current_lr:.6f} -> {current_lr * 0.7:.6f}"
            )
            current_lr *= 0.7

            if current_lr < 1e-5:
                current_lr = 0.001

        except (ValueError, RuntimeError, KeyError) as e:
            logger.warning(f"Simulation failed on attempt {attempt + 1}: {e}")
            continue

    return None, None, None, None, None, False, current_lr


def _setup_pdf_plotting(args_dict, output_dir, tel_model_name):
    """Set up PDF plotting if requested."""
    if not args_dict.get("plot_all", False):
        return None
    pdf_filename = output_dir / f"psf_gradient_descent_plots_{tel_model_name}.pdf"
    pdf_pages = PdfPages(pdf_filename)
    logger.info(f"Creating cumulative PSF plots for each iteration (saving to {pdf_filename})")
    return pdf_pages


def _create_step_plot(
    pdf_pages,
    args_dict,
    data_to_plot,
    current_params,
    new_d80,
    new_metric,
    new_p_value,
    new_simulated_data,
):
    """Create plot for an accepted gradient step."""
    if pdf_pages is None or not args_dict.get("plot_all", False) or new_simulated_data is None:
        return

    data_to_plot["simulated"] = new_simulated_data
    plot_psf.create_psf_parameter_plot(
        data_to_plot,
        current_params,
        new_d80,
        new_metric,
        False,
        pdf_pages,
        p_value=new_p_value,
        use_ks_statistic=False,
    )
    del data_to_plot["simulated"]


def _create_final_plot(
    pdf_pages, tel_model, site_model, args_dict, best_params, data_to_plot, radius, best_d80
):
    """Create final plot for best parameters."""
    if pdf_pages is None or best_params is None:
        return

    logger.info("Creating final plot for best parameters with both RMSD and KS statistic...")
    _, best_ks_stat, best_p_value, best_simulated_data = run_psf_simulation(
        tel_model,
        site_model,
        args_dict,
        best_params,
        data_to_plot,
        radius,
        pdf_pages=None,
        is_best=False,
        use_ks_statistic=True,
    )
    best_rmsd = calculate_rmsd(
        data_to_plot["measured"][CUMULATIVE_PSF], best_simulated_data[CUMULATIVE_PSF]
    )

    data_to_plot["simulated"] = best_simulated_data
    plot_psf.create_psf_parameter_plot(
        data_to_plot,
        best_params,
        best_d80,
        best_rmsd,
        True,
        pdf_pages,
        p_value=best_p_value,
        use_ks_statistic=False,
        second_metric=best_ks_stat,
    )
    del data_to_plot["simulated"]
    pdf_pages.close()
    logger.info("Cumulative PSF plots saved")


def run_gradient_descent_optimization(
    tel_model,
    site_model,
    args_dict,
    data_to_plot,
    radius,
    rmsd_threshold,
    learning_rate,
    output_dir,
):
    """Run gradient descent optimization to minimize RMSD."""
    if data_to_plot is None or radius is None:
        logger.error("No PSF measurement data provided. Cannot run optimization.")
        return None, None, []

    current_params = get_previous_values(tel_model)
    pdf_pages = _setup_pdf_plotting(args_dict, output_dir, tel_model.name)
    results = []

    # Evaluate initial parameters
    current_d80, current_metric, current_p_value, simulated_data = run_psf_simulation(
        tel_model,
        site_model,
        args_dict,
        current_params,
        data_to_plot,
        radius,
        pdf_pages=pdf_pages if args_dict.get("plot_all", False) else None,
        is_best=False,
        use_ks_statistic=False,
    )

    results.append(
        (current_params.copy(), current_metric, current_p_value, current_d80, simulated_data)
    )
    best_metric, best_params, best_d80 = current_metric, current_params.copy(), current_d80

    logger.info(f"Initial RMSD: {current_metric:.6f}, D80: {current_d80:.6f} cm")

    iteration = 0
    max_total_iterations = 100

    while iteration < max_total_iterations:
        if current_metric <= rmsd_threshold:
            logger.info(
                f"Optimization converged: RMSD {current_metric:.6f} <= "
                f"threshold {rmsd_threshold:.6f}"
            )
            break

        iteration += 1
        logger.info(f"Gradient descent iteration {iteration}")

        step_result = _perform_gradient_step_with_retries(
            tel_model,
            site_model,
            args_dict,
            current_params,
            current_metric,
            data_to_plot,
            radius,
            learning_rate,
        )
        (
            new_params,
            new_d80,
            new_metric,
            new_p_value,
            new_simulated_data,
            step_accepted,
            learning_rate,
        ) = step_result

        if not step_accepted or new_params is None:
            learning_rate *= 2.0
            logger.info(f"No step accepted, increasing learning rate to {learning_rate:.6f}")
            continue

        # Step was accepted - update state
        current_params, current_metric, current_d80 = new_params, new_metric, new_d80
        results.append(
            (current_params.copy(), current_metric, None, current_d80, new_simulated_data)
        )

        if current_metric < best_metric:
            best_metric, best_params, best_d80 = current_metric, current_params.copy(), current_d80

        _create_step_plot(
            pdf_pages,
            args_dict,
            data_to_plot,
            current_params,
            new_d80,
            new_metric,
            new_p_value,
            new_simulated_data,
        )
        logger.info(f"  Accepted step: improved to {new_metric:.6f}")

    _create_final_plot(
        pdf_pages, tel_model, site_model, args_dict, best_params, data_to_plot, radius, best_d80
    )
    return best_params, best_d80, results


def _write_log_interpretation(f, use_ks_statistic):
    """Write interpretation section for the log file."""
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


def _get_significance_label(p_value):
    """Get significance label for p-value."""
    if p_value > 0.05:
        return "GOOD"
    if p_value > 0.01:
        return "FAIR"
    return "POOR"


def _write_iteration_entry(
    f, iteration, pars, metric, p_value, d80, use_ks_statistic, metric_name, total_iterations
):
    """Write a single iteration entry."""
    status = "FINAL" if iteration == total_iterations - 1 else f"ITER-{iteration:02d}"

    if use_ks_statistic and p_value is not None:
        significance = _get_significance_label(p_value)
        f.write(
            f"[{status}] Iteration {iteration}: KS_stat={metric:.6f}, "
            f"p_value={p_value:.6f} ({significance}), D80={d80:.6f} cm\n"
        )
    else:
        f.write(f"[{status}] Iteration {iteration}: {metric_name}={metric:.6f}, D80={d80:.6f} cm\n")

    for par, value in pars.items():
        f.write(f"    {par}: {_create_log_header_and_format_value(None, None, None, value)}\n")
    f.write("\n")


def _write_optimization_summary(f, gd_results, best_pars, best_d80, metric_name):
    """Write optimization summary section."""
    f.write("OPTIMIZATION SUMMARY:\n")
    best_metric_from_results = min(metric for _, metric, _, _, _ in gd_results)
    f.write(f"Best {metric_name.lower()}: {best_metric_from_results:.6f}\n")
    f.write(f"Best D80: {best_d80:.6f} cm\n" if best_d80 is not None else "Best D80: N/A\n")
    f.write(f"Total iterations: {len(gd_results)}\n\nFINAL OPTIMIZED PARAMETERS:\n")
    for par, value in best_pars.items():
        f.write(f"{par}: {_create_log_header_and_format_value(None, None, None, value)}\n")


def write_gradient_descent_log(
    gd_results, best_pars, best_d80, output_dir, tel_model, use_ks_statistic=False
):
    """Write gradient descent progression to a log file."""
    metric_name = "KS Statistic" if use_ks_statistic else "RMSD"
    file_suffix = "ks" if use_ks_statistic else "rmsd"
    param_file = output_dir.joinpath(f"psf_gradient_descent_{file_suffix}_{tel_model.name}.log")

    with open(param_file, "w", encoding="utf-8") as f:
        header = _create_log_header_and_format_value(
            f"PSF Parameter Optimization - Gradient Descent Progression ({metric_name})",
            tel_model,
            {"Total iterations": len(gd_results)},
        )
        f.write(header)

        f.write(
            "GRADIENT DESCENT PROGRESSION:\n(Each entry shows the parameters chosen "
            "at each iteration)\n\n"
        )
        _write_log_interpretation(f, use_ks_statistic)

        for iteration, (pars, metric, p_value, d80, _) in enumerate(gd_results):
            _write_iteration_entry(
                f,
                iteration,
                pars,
                metric,
                p_value,
                d80,
                use_ks_statistic,
                metric_name,
                len(gd_results),
            )

        _write_optimization_summary(f, gd_results, best_pars, best_d80, metric_name)

    return param_file


def analyze_monte_carlo_error(
    tel_model, site_model, args_dict, data_to_plot, radius, n_simulations=500
):
    """Analyze Monte Carlo error on the optimization metric by running multiple simulations."""
    if data_to_plot is None or radius is None:
        logger.error("No PSF measurement data provided. Cannot analyze Monte Carlo error.")
        return None, None, []

    initial_params = get_previous_values(tel_model)
    for param_name, param_values in initial_params.items():
        logger.info(f"  {param_name}: {param_values}")

    use_ks_statistic = args_dict.get("ks_statistic", False)
    metric_values, p_values, d80_values = [], [], []

    for i in range(n_simulations):
        try:
            d80, metric, p_value, _ = run_psf_simulation(
                tel_model,
                site_model,
                args_dict,
                initial_params,
                data_to_plot,
                radius,
                use_ks_statistic=use_ks_statistic,
            )
            metric_values.append(metric)
            d80_values.append(d80)
            p_values.append(p_value)
        except (ValueError, RuntimeError) as e:
            logger.warning(f"WARNING: Simulation {i + 1} failed: {e}")

    if not metric_values:
        logger.error("All Monte Carlo simulations failed.")
        return None, None, [], None, None, []

    mean_metric, std_metric = np.mean(metric_values), np.std(metric_values, ddof=1)
    mean_d80, std_d80 = np.mean(d80_values), np.std(d80_values, ddof=1)

    if use_ks_statistic:
        valid_p_values = [p for p in p_values if p is not None]
        mean_p_value = np.mean(valid_p_values) if valid_p_values else None
        std_p_value = np.std(valid_p_values, ddof=1) if valid_p_values else None
    else:
        mean_p_value = std_p_value = None

    return (
        mean_metric,
        std_metric,
        metric_values,
        mean_p_value,
        std_p_value,
        p_values,
        mean_d80,
        std_d80,
        d80_values,
    )


def write_monte_carlo_analysis(mc_results, output_dir, tel_model, use_ks_statistic=False):
    """Write Monte Carlo analysis results to a log file."""
    (
        mean_metric,
        std_metric,
        metric_values,
        mean_p_value,
        std_p_value,
        p_values,
        mean_d80,
        std_d80,
        d80_values,
    ) = mc_results

    metric_name = "KS Statistic" if use_ks_statistic else "RMSD"
    file_suffix = "ks" if use_ks_statistic else "rmsd"
    mc_file = output_dir.joinpath(f"monte_carlo_{file_suffix}_analysis_{tel_model.name}.log")

    with open(mc_file, "w", encoding="utf-8") as f:
        header = _create_log_header_and_format_value(
            f"Monte Carlo {metric_name} Error Analysis",
            tel_model,
            {"Number of simulations": len(metric_values)},
        )
        f.write(header)

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
            f.write(
                f"P-VALUE STATISTICS:\nMean p-value: {mean_p_value:.6f}\n"
                f"Standard deviation: {std_p_value:.6f}\n"
                f"Minimum p-value: {min(valid_p_values):.6f}\n"
                f"Maximum p-value: {max(valid_p_values):.6f}\n"
                f"Relative error: {(std_p_value / mean_p_value) * 100:.2f}%\n"
            )

            good_fits = sum(1 for p in valid_p_values if p > 0.05)
            fair_fits = sum(1 for p in valid_p_values if 0.01 < p <= 0.05)
            poor_fits = sum(1 for p in valid_p_values if p <= 0.01)
            f.write(
                f"Good fits (p > 0.05): {good_fits}/{len(valid_p_values)} "
                f"({100 * good_fits / len(valid_p_values):.1f}%)\n"
                f"Fair fits (0.01 < p <= 0.05): {fair_fits}/{len(valid_p_values)} "
                f"({100 * fair_fits / len(valid_p_values):.1f}%)\n"
                f"Poor fits (p <= 0.01): {poor_fits}/{len(valid_p_values)} "
                f"({100 * poor_fits / len(valid_p_values):.1f}%)\n\n"
            )

        f.write(
            f"D80 STATISTICS:\nMean D80: {mean_d80:.6f} cm\n"
            f"Standard deviation: {std_d80:.6f} cm\n"
            f"Minimum D80: {min(d80_values):.6f} cm\n"
            f"Maximum D80: {max(d80_values):.6f} cm\n"
            f"Relative error: {(std_d80 / mean_d80) * 100:.2f}%\n\n"
        )

        f.write("INDIVIDUAL SIMULATION RESULTS:\n")
        for i, (metric_val, p_value, d80) in enumerate(zip(metric_values, p_values, d80_values)):
            if use_ks_statistic and p_value is not None:
                if p_value > 0.05:
                    significance = "GOOD"
                elif p_value > 0.01:
                    significance = "FAIR"
                else:
                    significance = "POOR"
                f.write(
                    f"Simulation {i + 1:2d}: {metric_name}={metric_val:.6f}, "
                    f"p_value={p_value:.6f} ({significance}), D80={d80:.6f} cm\n"
                )
            else:
                f.write(
                    f"Simulation {i + 1:2d}: {metric_name}={metric_val:.6f}, D80={d80:.6f} cm\n"
                )

    return mc_file


def _handle_monte_carlo_analysis(
    tel_model, site_model, args_dict, data_to_plot, radius, output_dir, use_ks_statistic
):
    """Handle Monte Carlo analysis if requested."""
    if not args_dict.get("monte_carlo_analysis", False):
        return False

    mc_results = analyze_monte_carlo_error(tel_model, site_model, args_dict, data_to_plot, radius)
    if mc_results[0] is not None:
        mc_file = write_monte_carlo_analysis(mc_results, output_dir, tel_model, use_ks_statistic)
        logger.info(f"Monte Carlo analysis results written to {mc_file}")
        mc_plot_file = output_dir.joinpath(f"monte_carlo_uncertainty_{tel_model.name}.pdf")
        plot_psf.create_monte_carlo_uncertainty_plot(mc_results, mc_plot_file, use_ks_statistic)
    return True


def _create_optimization_plots(args_dict, gd_results, tel_model, data_to_plot, output_dir):
    """Create optimization plots if requested."""
    if not args_dict.get("save_plots", False):
        return

    pdf_filename = output_dir.joinpath(f"psf_optimization_results_{tel_model.name}.pdf")
    pdf_pages = PdfPages(pdf_filename)
    logger.info(f"Creating PSF plots for each optimization iteration (saving to {pdf_filename})")

    for i, (params, rmsd, _, d80, _) in enumerate(gd_results):
        if i % 5 == 0 or i == len(gd_results) - 1:
            plot_psf.create_psf_parameter_plot(
                data_to_plot,
                params,
                d80,
                rmsd,
                is_best=(i == len(gd_results) - 1),
                pdf_pages=pdf_pages,
                use_ks_statistic=False,
            )
    pdf_pages.close()


def run_psf_optimization_workflow(tel_model, site_model, args_dict, output_dir):
    """Run the complete PSF parameter optimization workflow using gradient descent."""
    data_to_plot, radius = load_and_process_data(args_dict)
    use_ks_statistic = args_dict.get("ks_statistic", False)

    if _handle_monte_carlo_analysis(
        tel_model, site_model, args_dict, data_to_plot, radius, output_dir, use_ks_statistic
    ):
        return

    # Run gradient descent optimization
    threshold = args_dict.get("rmsd_threshold")
    learning_rate = args_dict.get("learning_rate")

    best_pars, best_d80, gd_results = run_gradient_descent_optimization(
        tel_model,
        site_model,
        args_dict,
        data_to_plot,
        radius,
        rmsd_threshold=threshold,
        learning_rate=learning_rate,
        output_dir=output_dir,
    )

    # Check if optimization was successful
    if not gd_results or best_pars is None:
        logger.error("Gradient descent optimization failed. No valid results found.")
        if radius is None:
            logger.error(
                "Possible cause: No PSF measurement data provided. "
                "Use --data argument to provide PSF data."
            )
        return

    _create_optimization_plots(args_dict, gd_results, tel_model, data_to_plot, output_dir)

    convergence_plot_file = output_dir.joinpath(
        f"gradient_descent_convergence_{tel_model.name}.png"
    )
    plot_psf.create_gradient_descent_convergence_plot(
        gd_results, threshold, convergence_plot_file, use_ks_statistic
    )

    param_file = write_gradient_descent_log(
        gd_results, best_pars, best_d80, output_dir, tel_model, use_ks_statistic
    )
    logger.info(f"\nGradient descent progression written to {param_file}")

    plot_psf.create_d80_vs_offaxis_plot(tel_model, site_model, args_dict, best_pars, output_dir)

    if args_dict.get("write_psf_parameters", False):
        logger.info("Exporting best parameters as model files...")
        export_psf_parameters(
            best_pars, args_dict.get("telescope"), args_dict.get("parameter_version"), output_dir
        )
