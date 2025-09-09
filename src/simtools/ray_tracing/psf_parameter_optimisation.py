"""
PSF parameter optimisation and fitting routines for mirror alignment and reflection parameters.

This module provides functions for loading PSF data, generating random parameter sets,
running PSF simulations, calculating KS statistic, and finding the best-fit parameters for a given
telescope model.

PSF (Point Spread Function) describes how a point source of light is spread out by the
optical system, and KS statistic (Kolmogorov-Smirnov) is used as the optimization metric
to quantify the difference between measured and simulated PSF curves.
"""

import logging
from collections import OrderedDict

import astropy.units as u
import numpy as np
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


def _create_log_file_header(title, tel_model, additional_info=None):
    """
    Create a standardized header for log files.

    Parameters
    ----------
    title : str
        Main title for the log file
    tel_model : TelescopeModel
        Telescope model object
    additional_info : dict, optional
        Additional information to include in the header

    Returns
    -------
    str
        Formatted header string
    """
    header_lines = [
        f"# {title}",
        f"# Telescope: {tel_model.name}",
    ]

    if additional_info:
        for key, value in additional_info.items():
            header_lines.append(f"# {key}: {value}")

    header_lines.append("#" + "=" * 60)
    header_lines.append("")  # Empty line after header

    return "\n".join(header_lines) + "\n"


def _format_parameter_value(value):
    """
    Format parameter values for consistent display in log files.

    Parameters
    ----------
    value : various
        Parameter value to format

    Returns
    -------
    str
        Formatted value string
    """
    if isinstance(value, list):
        return "[" + ", ".join([f"{v:.6f}" for v in value]) + "]"
    if isinstance(value, int | float):
        return f"{value:.6f}"
    return str(value)


def _validate_psf_data(data_to_plot, radius):
    """
    Validate that PSF data and radius are available for analysis.

    Parameters
    ----------
    data_to_plot : dict or None
        PSF data dictionary
    radius : array or None
        Radius data array

    Raises
    ------
    ValueError
        If data is not available for analysis
    """
    if data_to_plot is None or radius is None:
        raise ValueError(
            "PSF data and radius are required for analysis. "
            "Please provide measured PSF data using the --data argument."
        )


def calculate_rmsd(data, sim):
    """Calculate RMSD between measured and simulated cumulative PSF curves."""
    return np.sqrt(np.mean((data - sim) ** 2))


def calculate_ks_statistic(data, sim):
    """Calculate the KS statistic between measured and simulated cumulative PSF curves."""
    # Use asymptotic method to avoid warnings with small sample sizes
    ks_statistic, p_value = stats.ks_2samp(data, sim, method="asymp")
    return ks_statistic, p_value


def get_previous_values(tel_model):
    """
    Retrieve previous parameter values from the telescope model.

    Parameters
    ----------
    tel_model : TelescopeModel
        Telescope model object.

    Returns
    -------
    dict
        Dictionary containing all PSF parameters needed for optimization:
        - mirror_reflection_random_angle: [mrra_0, mfr_0, mrra2_0]
        - mirror_align_random_horizontal: full 4-element array
        - mirror_align_random_vertical: full 4-element array
    """
    # Get mirror reflection parameters
    split_par = tel_model.get_parameter_value("mirror_reflection_random_angle")
    mrra_0, mfr_0, mrra2_0 = split_par[0], split_par[1], split_par[2]

    # Get full mirror alignment arrays
    mirror_align_h = tel_model.get_parameter_value("mirror_align_random_horizontal")
    mirror_align_v = tel_model.get_parameter_value("mirror_align_random_vertical")

    return {
        "mirror_reflection_random_angle": [mrra_0, mfr_0, mrra2_0],
        "mirror_align_random_horizontal": mirror_align_h,
        "mirror_align_random_vertical": mirror_align_v,
    }


def _run_ray_tracing_simulation(tel_model, site_model, args_dict, pars):
    """
    Run a ray tracing simulation with the given telescope parameters.

    Parameters
    ----------
    tel_model : TelescopeModel
        Telescope model object.
    site_model : SiteModel
        Site model object.
    args_dict : dict
        Dictionary containing parsed command-line arguments.
    pars : dict
        Parameter set dictionary.

    Returns
    -------
    tuple
        (d80, simulated_data) - D80 value and simulated data from ray tracing.
    """
    if pars is not None:
        tel_model.change_multiple_parameters(**pars)
    else:
        raise ValueError("No best parameters found")

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
    d80 = im.get_psf()

    return d80, im


def run_psf_simulation(
    tel_model,
    site_model,
    args_dict,
    pars,
    data_to_plot,
    radius,
    pdf_pages=None,
    is_best=False,
    return_simulated_data=False,
    use_ks_statistic=False,
):
    """
    Run the simulation for one set of parameters and return D80, RMSD or KS statistic.

    Parameters
    ----------
    tel_model : TelescopeModel
        Telescope model object.
    site_model : SiteModel
        Site model object.
    args_dict : dict
        Dictionary containing parsed command-line arguments.
    pars : dict
        Parameter set dictionary.
    data_to_plot : dict
        Data dictionary for plotting.
    radius : array-like
        Radius data.
    pdf_pages : PdfPages, optional
        PDF pages object for plotting. If None, no plotting is done.
    is_best : bool, optional
        Whether this is the best parameter set for highlighting in plots.
    return_simulated_data : bool, optional
        If True, returns simulated data as third element in return tuple.
    use_ks_statistic : bool, optional
        If True, returns KS statistic and p-value instead of RMSD.

    Returns
    -------
    tuple
        (d80, rmsd) if use_ks_statistic=False and return_simulated_data=False
        (d80, rmsd, simulated_data) if use_ks_statistic=False and return_simulated_data=True
        (d80, ks_statistic, p_value) if use_ks_statistic=True and return_simulated_data=False
        (d80, ks_statistic, p_value, simulated_data) if use_ks_statistic=True and
        return_simulated_data=True
    """
    d80, im = _run_ray_tracing_simulation(tel_model, site_model, args_dict, pars)

    if radius is None:
        raise ValueError("Radius data is not available.")

    simulated_data = im.get_cumulative_data(radius * u.cm)

    if use_ks_statistic:
        ks_statistic, p_value = calculate_ks_statistic(
            data_to_plot["measured"][CUMULATIVE_PSF], simulated_data[CUMULATIVE_PSF]
        )
        metric = ks_statistic
        extra_return = p_value
    else:
        rmsd = calculate_rmsd(
            data_to_plot["measured"][CUMULATIVE_PSF], simulated_data[CUMULATIVE_PSF]
        )
        metric = rmsd
        extra_return = None

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
            p_value=extra_return,
            use_ks_statistic=use_ks_statistic,
        )
        del data_to_plot["simulated"]

    if use_ks_statistic:
        return (
            (d80, metric, extra_return, simulated_data)
            if return_simulated_data
            else (d80, metric, extra_return)
        )
    return (d80, metric, simulated_data) if return_simulated_data else (d80, metric)


def load_and_process_data(args_dict):
    """
    Load and process data if specified in the command-line arguments.

    Parameters
    ----------
    args_dict : dict
        Dictionary containing parsed command-line arguments with 'data' and 'model_path' keys.

    Returns
    -------
    tuple
        (data_to_plot, radius) where:
        - data_to_plot: OrderedDict containing loaded and processed data
        - radius: Radius data from loaded data (if available)
    """
    data_to_plot = OrderedDict()
    radius = None
    if args_dict["data"] is not None:
        data_file = gen.find_file(args_dict["data"], args_dict["model_path"])

        # Load data from text file containing cumulative PSF measurements
        d_type = {"names": (RADIUS_CM, CUMULATIVE_PSF), "formats": ("f8", "f8")}
        data = np.loadtxt(data_file, dtype=d_type, usecols=(0, 2))
        data[RADIUS_CM] *= 0.1  # Convert from mm to cm
        data[CUMULATIVE_PSF] /= np.max(np.abs(data[CUMULATIVE_PSF]))  # Normalize to max = 1.0

        data_to_plot["measured"] = data
        radius = data[RADIUS_CM]
    return data_to_plot, radius


def write_tested_parameters_to_file(results, best_pars, best_d80, output_dir, tel_model):
    """
    Write all tested parameters and their metrics to a text file.

    Parameters
    ----------
    results : list
        List of (pars, ks_statistic, p_value, d80, simulated_data) tuples
    best_pars : dict
        Best parameter set
    best_d80 : float
        Best D80 value
    output_dir : Path
        Output directory path
    tel_model : TelescopeModel
        Telescope model object for filename generation
    """
    param_file = output_dir.joinpath(f"psf_optimization_{tel_model.name}.log")
    with open(param_file, "w", encoding="utf-8") as f:
        header = _create_log_file_header(
            "PSF Parameter Optimization Log",
            tel_model,
            {"Total parameter sets tested": len(results)},
        )
        f.write(header)

        f.write("PARAMETER TESTING RESULTS:\n")
        for i, (pars, ks_statistic, p_value, d80, _) in enumerate(results):
            is_best = pars is best_pars
            status = "BEST" if is_best else "TESTED"
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
    """
    Add proper astropy units to PSF parameters based on their schemas.

    Parameters
    ----------
    best_pars : dict
        Dictionary with PSF parameter names as keys and values as lists

    Returns
    -------
    dict
        Dictionary with same keys but values converted to astropy quantities with units
    """
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


def export_psf_parameters(best_pars, tel_model, parameter_version, output_dir):
    """
    Export PSF parameters as simulation model parameter files.

    Parameters
    ----------
    best_pars : dict
        Best parameter set
    tel_model : TelescopeModel
        Telescope model object
    parameter_version : str
        Parameter version string
    output_dir : Path
        Output directory path
    """
    try:
        psf_pars_with_units = _add_units_to_psf_parameters(best_pars)
        parameter_output_path = output_dir / tel_model.name
        for parameter_name, parameter_value in psf_pars_with_units.items():
            writer.ModelDataWriter.dump_model_parameter(
                parameter_name=parameter_name,
                value=parameter_value,
                instrument=tel_model.name,
                parameter_version=parameter_version,
                output_file=f"{parameter_name}-{parameter_version}.json",
                output_path=parameter_output_path,
                use_plain_output_path=True,
            )
        logger.info(f"simulation model parameter files exported to {output_dir}")
    except ImportError as e:
        logger.warning(f"Could not export simulation parameters: {e}")
    except (ValueError, KeyError, OSError) as e:
        logger.error(f"Error exporting simulation parameters: {e}")


def _calculate_parameter_gradient(
    tel_model,
    site_model,
    args_dict,
    param_name,
    param_values,
    current_params,
    current_rmsd,
    data_to_plot,
    radius,
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
            result = run_psf_simulation(
                tel_model,
                site_model,
                args_dict,
                perturbed_params,
                data_to_plot,
                radius,
                pdf_pages=None,
                is_best=False,
                return_simulated_data=False,
                use_ks_statistic=use_ks_statistic,
            )
            # Extract the metric value based on whether we're using KS statistic
            if use_ks_statistic:
                _, perturbed_rmsd, _ = (
                    result  # (d80, ks_statistic, p_value)  # pylint: disable=unbalanced-tuple-unpacking
                )
            else:
                # pylint: disable=unbalanced-tuple-unpacking
                _, perturbed_rmsd = (  # pylint: disable=unbalanced-tuple-unpacking
                    result  # (d80, rmsd)  # pylint: disable=unbalanced-tuple-unpacking
                )

            gradient = (perturbed_rmsd - current_rmsd) / epsilon
            param_gradients.append(gradient)
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
    """
    Calculate numerical gradient of RMSD with respect to parameters.

    Parameters
    ----------
    tel_model : TelescopeModel
        Telescope model object
    site_model : SiteModel
        Site model object
    args_dict : dict
        Arguments dictionary
    current_params : dict
        Current parameter values
    data_to_plot : array
        Measured PSF data
    radius : array
        Radius array
    current_rmsd : float
        Current RMSD value (pre-computed to avoid duplication)
    epsilon : float
        Step size for numerical differentiation

    Returns
    -------
    dict
        Gradient dictionary with same structure as current_params
    """
    gradients = {}

    for param_name, param_values in current_params.items():
        gradients[param_name] = _calculate_parameter_gradient(
            tel_model,
            site_model,
            args_dict,
            param_name,
            param_values,
            current_params,
            current_rmsd,
            data_to_plot,
            radius,
            epsilon,
            use_ks_statistic,
        )

    return gradients


def apply_gradient_step(current_params, gradients, learning_rate):
    """
    Apply gradient descent step.

    Parameters
    ----------
    current_params : dict
        Current parameter values
    gradients : dict
        Parameter gradients
    learning_rate : float
        Learning rate for gradient descent

    Returns
    -------
    dict
        Updated parameters after gradient step
    """
    new_params = {}

    for param_name, param_values in current_params.items():
        param_gradients = gradients[param_name]

        if isinstance(param_values, list):
            new_params[param_name] = []
            for value, gradient in zip(param_values, param_gradients):
                new_value = value - learning_rate * gradient
                new_params[param_name].append(new_value)
        else:
            new_value = param_values - learning_rate * param_gradients
            new_params[param_name] = new_value

    return new_params


def _setup_optimization_plotting(args_dict, output_dir, tel_model):
    """Set up PDF plotting for optimization if requested."""
    pdf_pages = None
    if args_dict.get("plot_all", False):
        pdf_filename = output_dir / f"psf_gradient_descent_plots_{tel_model.name}.pdf"
        pdf_pages = PdfPages(pdf_filename)
        logger.info(f"Creating cumulative PSF plots for each iteration (saving to {pdf_filename})")
    return pdf_pages


def _evaluate_initial_parameters(
    tel_model, site_model, args_dict, current_params, data_to_plot, radius, pdf_pages
):
    """Evaluate initial parameters and return results."""
    use_ks_statistic = args_dict.get("ks_statistic", False)
    plot_all = args_dict.get("plot_all", False)

    if plot_all:
        if use_ks_statistic:
            current_d80, current_metric, current_p_value, simulated_data = run_psf_simulation(  # pylint: disable=unbalanced-tuple-unpacking
                tel_model,
                site_model,
                args_dict,
                current_params,
                data_to_plot,
                radius,
                pdf_pages=pdf_pages,
                is_best=False,
                return_simulated_data=True,
                use_ks_statistic=True,
            )
        else:
            current_d80, current_metric, simulated_data = run_psf_simulation(  # pylint: disable=unbalanced-tuple-unpacking
                tel_model,
                site_model,
                args_dict,
                current_params,
                data_to_plot,
                radius,
                pdf_pages=pdf_pages,
                is_best=False,
                return_simulated_data=True,
                use_ks_statistic=False,
            )
            current_p_value = None
    else:
        if use_ks_statistic:
            current_d80, current_metric, current_p_value = run_psf_simulation(  # pylint: disable=unbalanced-tuple-unpacking
                tel_model,
                site_model,
                args_dict,
                current_params,
                data_to_plot,
                radius,
                pdf_pages=None,
                is_best=False,
                return_simulated_data=False,
                use_ks_statistic=True,
            )
            simulated_data = None
        else:
            current_d80, current_metric = run_psf_simulation(  # pylint: disable=unbalanced-tuple-unpacking
                tel_model,
                site_model,
                args_dict,
                current_params,
                data_to_plot,
                radius,
                pdf_pages=None,
                is_best=False,
                return_simulated_data=False,
                use_ks_statistic=False,
            )
            simulated_data = None
            current_p_value = None

    return current_d80, current_metric, current_p_value, simulated_data


def _perform_gradient_step(
    tel_model,
    site_model,
    args_dict,
    current_params,
    current_metric,
    data_to_plot,
    radius,
    learning_rate,
    use_ks_statistic,
):
    """Perform a single gradient descent step."""
    try:
        gradients = calculate_gradient(
            tel_model,
            site_model,
            args_dict,
            current_params,
            data_to_plot,
            radius,
            current_metric,
            use_ks_statistic=use_ks_statistic,
        )

        new_params = apply_gradient_step(current_params, gradients, learning_rate)

        plot_all = args_dict.get("plot_all", False)

        if plot_all:
            if use_ks_statistic:
                new_d80, new_metric, new_p_value, new_simulated_data = run_psf_simulation(  # pylint: disable=unbalanced-tuple-unpacking
                    tel_model,
                    site_model,
                    args_dict,
                    new_params,
                    data_to_plot,
                    radius,
                    pdf_pages=None,
                    is_best=False,
                    use_ks_statistic=True,
                    return_simulated_data=True,
                )
            else:
                new_d80, new_metric, new_simulated_data = run_psf_simulation(  # pylint: disable=unbalanced-tuple-unpacking
                    tel_model,
                    site_model,
                    args_dict,
                    new_params,
                    data_to_plot,
                    radius,
                    pdf_pages=None,
                    is_best=False,
                    use_ks_statistic=False,
                    return_simulated_data=True,
                )
                new_p_value = None
        else:
            if use_ks_statistic:
                new_d80, new_metric, new_p_value = run_psf_simulation(  # pylint: disable=unbalanced-tuple-unpacking
                    tel_model,
                    site_model,
                    args_dict,
                    new_params,
                    data_to_plot,
                    radius,
                    pdf_pages=None,
                    is_best=False,
                    use_ks_statistic=True,
                    return_simulated_data=False,
                )
                new_simulated_data = None
            else:
                new_d80, new_metric = run_psf_simulation(  # pylint: disable=unbalanced-tuple-unpacking
                    tel_model,
                    site_model,
                    args_dict,
                    new_params,
                    data_to_plot,
                    radius,
                    pdf_pages=None,
                    is_best=False,
                    use_ks_statistic=False,
                    return_simulated_data=False,
                )
                new_simulated_data = None
                new_p_value = None

        return new_params, new_d80, new_metric, new_p_value, new_simulated_data, True

    except (ValueError, RuntimeError, KeyError) as e:
        logger.error(f"Error in gradient step: {e}")
        return None, None, None, None, None, False


def _should_accept_step(current_metric, new_metric):
    """Determine if a gradient step should be accepted."""
    improvement = current_metric - new_metric
    relative_improvement = improvement / current_metric if current_metric > 0 else 0
    return improvement > 0 and (relative_improvement > 0.001 or improvement > 0.0001)


def _create_accepted_step_plot(
    pdf_pages,
    args_dict,
    data_to_plot,
    new_simulated_data,
    current_params,
    new_d80,
    new_metric,
    new_p_value,
    use_ks_statistic,
):
    """Create plot for an accepted gradient step."""
    if (
        pdf_pages is not None
        and args_dict.get("plot_all", False)
        and new_simulated_data is not None
    ):
        data_to_plot["simulated"] = new_simulated_data
        plot_psf.create_psf_parameter_plot(
            data_to_plot,
            current_params,
            new_d80,
            new_metric,
            False,
            pdf_pages,
            p_value=new_p_value if use_ks_statistic else None,
            use_ks_statistic=use_ks_statistic,
        )
        del data_to_plot["simulated"]


def _log_iteration_metrics(use_ks_statistic, current_metric, current_d80):
    """Log initial metrics for the gradient descent optimization."""
    metric_name = KS_STATISTIC_NAME if use_ks_statistic else "RMSD"
    logger.info(f"Initial {metric_name}: {current_metric:.6f}, D80: {current_d80:.6f} cm")


def _should_stop_optimization(current_metric, rmsd_threshold, use_ks_statistic):
    """Check if optimization should stop based on threshold."""
    if current_metric <= rmsd_threshold:
        metric_name = KS_STATISTIC_NAME if use_ks_statistic else "RMSD"
        logger.info(
            f"Optimization converged: {metric_name} {current_metric:.6f} <= "
            f"threshold {rmsd_threshold:.6f}"
        )
        return True
    return False


def _update_learning_rate_on_rejection(learning_rate, current_metric, new_metric, use_ks_statistic):
    """Update learning rate when step is rejected."""
    learning_rate *= 0.7
    improvement = current_metric - new_metric
    relative_improvement = improvement / current_metric if current_metric > 0 else 0
    metric_name = KS_STATISTIC_NAME if use_ks_statistic else "RMSD"

    logger.info(
        f"  Rejected step: {metric_name} would change from {current_metric:.6f} to "
        f"{new_metric:.6f} (improvement: {improvement:.6f}, {relative_improvement * 100:.3f}%)"
    )

    if learning_rate < 1e-6:
        logger.info("Learning rate getting too small for this iteration, moving to next iteration.")
        return learning_rate, True  # Should break

    return learning_rate, False


def _log_accepted_step(current_metric, new_metric, use_ks_statistic):
    """Log information about an accepted gradient step."""
    improvement = current_metric - new_metric
    relative_improvement = improvement / current_metric if current_metric > 0 else 0
    metric_name = KS_STATISTIC_NAME if use_ks_statistic else "RMSD"

    logger.info(
        f"  Accepted step: {metric_name} improved by {improvement:.6f} "
        f"({relative_improvement * 100:.3f}%) to {new_metric:.6f}"
    )


def _handle_iteration_failure(step_accepted, learning_rate, iteration):
    """Handle case where no step was accepted in an iteration."""
    if not step_accepted:
        learning_rate *= 1.2
        logger.info(f"No step accepted, increasing learning rate to {learning_rate:.6f}")

        if learning_rate > 0.1:
            logger.warning("Learning rate getting very large - optimization may be stuck")
            return learning_rate, True  # Should break

    if iteration % 10 == 0 and iteration > 0:
        original_lr = 0.0001  # Default learning rate
        logger.info(f"Resetting learning rate to original value: {original_lr}")
        return original_lr, False

    return learning_rate, False


def _create_final_best_plot(
    pdf_pages, best_params, tel_model, site_model, args_dict, data_to_plot, radius, best_d80
):
    """Create final plot for best parameters with both RMSD and KS statistic."""
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
        return_simulated_data=True,
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


def _try_gradient_step_with_retries(
    tel_model,
    site_model,
    args_dict,
    current_params,
    current_metric,
    data_to_plot,
    radius,
    learning_rate,
    use_ks_statistic,
    max_retries=3,
):
    """Try gradient step with retries, return result or None if failed."""
    for retries in range(max_retries):
        step_result = _perform_gradient_step(
            tel_model,
            site_model,
            args_dict,
            current_params,
            current_metric,
            data_to_plot,
            radius,
            learning_rate,
            use_ks_statistic,
        )

        _, _, _, _, _, step_success = step_result

        if step_success:
            return step_result

        if retries >= max_retries - 1:
            logger.warning("Too many retries, moving to next iteration")

    return None


def _update_best_parameters(
    current_metric, best_metric, current_params, best_params, current_d80, best_d80
):
    """Update best parameters if current is better."""
    if current_metric < best_metric:
        return current_metric, current_params.copy(), current_d80
    return best_metric, best_params, best_d80


def _execute_single_iteration(
    tel_model,
    site_model,
    args_dict,
    optimization_state,
    data_to_plot,
    radius,
    learning_rate,
    use_ks_statistic,
    pdf_pages,
    results,
):
    """Execute a single gradient descent iteration with retry logic."""
    current_params, current_metric, current_d80, best_params, best_metric, best_d80 = (
        optimization_state
    )

    step_result = _try_gradient_step_with_retries(
        tel_model,
        site_model,
        args_dict,
        current_params,
        current_metric,
        data_to_plot,
        radius,
        learning_rate,
        use_ks_statistic,
    )

    if step_result is None:
        return optimization_state, learning_rate, False

    new_params, new_d80, new_metric, new_p_value, new_simulated_data, _ = step_result

    if _should_accept_step(current_metric, new_metric):
        # Update current parameters
        current_params = new_params
        current_metric = new_metric
        current_d80 = new_d80
        current_p_value = new_p_value if use_ks_statistic else None

        results.append(
            (
                current_params.copy(),
                current_metric,
                current_p_value,
                current_d80,
                new_simulated_data,
            )
        )

        best_metric, best_params, best_d80 = _update_best_parameters(
            current_metric, best_metric, current_params, best_params, current_d80, best_d80
        )

        _create_accepted_step_plot(
            pdf_pages,
            args_dict,
            data_to_plot,
            new_simulated_data,
            current_params,
            new_d80,
            new_metric,
            new_p_value,
            use_ks_statistic,
        )

        _log_accepted_step(
            best_metric if best_metric != current_metric else current_metric,
            new_metric,
            use_ks_statistic,
        )

        updated_state = (
            current_params,
            current_metric,
            current_d80,
            best_params,
            best_metric,
            best_d80,
        )
        return updated_state, learning_rate, True
    learning_rate, should_break = _update_learning_rate_on_rejection(
        learning_rate, current_metric, new_metric, use_ks_statistic
    )
    if should_break:
        return optimization_state, learning_rate, False
    return optimization_state, learning_rate, False


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
    """
    Run gradient descent optimization to minimize RMSD.

    Parameters
    ----------
    tel_model : TelescopeModel
        Telescope model object
    site_model : SiteModel
        Site model object
    args_dict : dict
        Arguments dictionary
    data_to_plot : array
        Measured PSF data
    radius : array
        Radius array
    rmsd_threshold : float
        RMSD threshold to stop optimization
    learning_rate : float
        Initial learning rate
    output_dir : Path
        Output directory for saving plots

    Returns
    -------
    tuple
        (best_params, best_d80, results_list)
        results_list contains (params, rmsd, d80, simulated_data) tuples
    """
    if data_to_plot is None or radius is None:
        logger.error("No PSF measurement data provided. Cannot run optimization.")
        return None, None, []

    current_params = get_previous_values(tel_model)
    pdf_pages = _setup_optimization_plotting(args_dict, output_dir, tel_model)

    results = []
    use_ks_statistic = args_dict.get("ks_statistic", False)

    # Evaluate initial parameters
    current_d80, current_metric, current_p_value, simulated_data = _evaluate_initial_parameters(
        tel_model, site_model, args_dict, current_params, data_to_plot, radius, pdf_pages
    )

    results.append(
        (current_params.copy(), current_metric, current_p_value, current_d80, simulated_data)
    )
    best_metric = current_metric
    best_params = current_params.copy()
    best_d80 = current_d80

    _log_iteration_metrics(use_ks_statistic, current_metric, current_d80)

    iteration = 0
    max_total_iterations = 10

    while iteration < max_total_iterations:
        if _should_stop_optimization(current_metric, rmsd_threshold, use_ks_statistic):
            break

        iteration += 1
        logger.info(f"Gradient descent iteration {iteration}")

        optimization_state = (
            current_params,
            current_metric,
            current_d80,
            best_params,
            best_metric,
            best_d80,
        )
        optimization_state, learning_rate, step_accepted = _execute_single_iteration(
            tel_model,
            site_model,
            args_dict,
            optimization_state,
            data_to_plot,
            radius,
            learning_rate,
            use_ks_statistic,
            pdf_pages,
            results,
        )
        current_params, current_metric, current_d80, best_params, best_metric, best_d80 = (
            optimization_state
        )

        learning_rate, should_break = _handle_iteration_failure(
            step_accepted, learning_rate, iteration
        )
        if should_break:
            break

    _create_final_best_plot(
        pdf_pages, best_params, tel_model, site_model, args_dict, data_to_plot, radius, best_d80
    )

    if pdf_pages is not None:
        pdf_pages.close()
        logger.info("Cumulative PSF plots saved")

    return best_params, best_d80, results


def _get_significance_level(p_value):
    """Get significance level description for p-value."""
    if p_value > 0.05:
        return "GOOD"
    if p_value > 0.01:
        return "FAIR"
    return "POOR"


def write_gradient_descent_log(
    gd_results, best_pars, best_d80, output_dir, tel_model, use_ks_statistic=False
):
    """
    Write gradient descent progression to a log file.

    Parameters
    ----------
    gd_results : list
        List of (params, metric, p_value, d80, simulated_data) tuples from gradient descent
    best_pars : dict
        Best parameter set
    best_d80 : float
        Best D80 value
    output_dir : Path
        Output directory path
    tel_model : TelescopeModel
        Telescope model object for filename generation
    use_ks_statistic : bool
        Whether KS statistic or RMSD was used

    Returns
    -------
    Path
        Path to the created log file
    """
    metric_name = "KS Statistic" if use_ks_statistic else "RMSD"
    file_suffix = "ks" if use_ks_statistic else "rmsd"
    param_file = output_dir.joinpath(f"psf_gradient_descent_{file_suffix}_{tel_model.name}.log")

    with open(param_file, "w", encoding="utf-8") as f:
        header = _create_log_file_header(
            f"PSF Parameter Optimization - Gradient Descent Progression ({metric_name})",
            tel_model,
            {"Total iterations": len(gd_results)},
        )
        f.write(header)

        f.write("GRADIENT DESCENT PROGRESSION:\n")
        f.write("(Each entry shows the parameters chosen at each iteration)\n\n")

        _write_interpretation_section(f, use_ks_statistic)
        _write_iteration_results(f, gd_results, use_ks_statistic, metric_name)
        _write_optimization_summary(f, gd_results, best_d80, best_pars, metric_name)

    return param_file


def _write_interpretation_section(f, use_ks_statistic):
    """Write the interpretation section for the log file."""
    if use_ks_statistic:
        f.write("P-VALUE INTERPRETATION:\n")
        f.write("  p > 0.05: Distributions are statistically similar (good fit)\n")
        f.write("  p < 0.05: Distributions are significantly different (poor fit)\n")
        f.write("  p < 0.01: Very significant difference (very poor fit)\n\n")
    else:
        f.write("RMSD INTERPRETATION:\n")
        f.write(
            "  Lower RMSD values indicate better agreement between measured and "
            "simulated PSF curves\n\n"
        )


def _write_iteration_results(f, gd_results, use_ks_statistic, metric_name):
    """Write iteration results to the log file."""
    for iteration, (pars, metric, p_value, d80, _) in enumerate(gd_results):
        is_final = iteration == len(gd_results) - 1
        status = "FINAL" if is_final else f"ITER-{iteration:02d}"

        if use_ks_statistic and p_value is not None:
            significance = _get_significance_level(p_value)
            f.write(
                f"[{status}] Iteration {iteration}: KS_stat={metric:.6f}, "
                f"p_value={p_value:.6f} ({significance}), D80={d80:.6f} cm\n"
            )
        else:
            f.write(
                f"[{status}] Iteration {iteration}: {metric_name}={metric:.6f}, D80={d80:.6f} cm\n"
            )

        for par, value in pars.items():
            f.write(f"    {par}: {_format_parameter_value(value)}\n")
        f.write("\n")


def _write_optimization_summary(f, gd_results, best_d80, best_pars, metric_name):
    """Write optimization summary to the log file."""
    f.write("OPTIMIZATION SUMMARY:\n")
    best_metric_from_results = min(metric for _, metric, _, _, _ in gd_results)

    f.write(f"Best {metric_name.lower()}: {best_metric_from_results:.6f}\n")
    f.write(f"Best D80: {best_d80:.6f} cm\n" if best_d80 is not None else "Best D80: N/A\n")
    f.write(f"Total iterations: {len(gd_results)}\n")
    f.write("\nFINAL OPTIMIZED PARAMETERS:\n")
    for par, value in best_pars.items():
        f.write(f"{par}: {_format_parameter_value(value)}\n")


def _run_monte_carlo_simulation(
    tel_model, site_model, args_dict, initial_params, data_to_plot, radius, use_ks_statistic, i
):
    """Run a single Monte Carlo simulation."""
    try:
        if use_ks_statistic:
            d80, metric, p_value = run_psf_simulation(
                tel_model,
                site_model,
                args_dict,
                initial_params,
                data_to_plot,
                radius,
                return_simulated_data=False,
                use_ks_statistic=True,
            )
            return d80, metric, p_value, True
        # When use_ks_statistic=False and return_simulated_data=False, returns (d80, rmsd)
        d80, metric = run_psf_simulation(  # pylint: disable=unbalanced-tuple-unpacking
            tel_model,
            site_model,
            args_dict,
            initial_params,
            data_to_plot,
            radius,
            return_simulated_data=False,
            use_ks_statistic=False,
        )
        return d80, metric, None, True
    except (ValueError, RuntimeError) as e:
        logger.warning(f"WARNING: Simulation {i + 1} failed: {e}")
        return None, None, None, False


def _calculate_monte_carlo_statistics(metric_values, d80_values, p_values, use_ks_statistic):
    """Calculate Monte Carlo statistics from simulation results."""
    if not metric_values:
        logger.error("All Monte Carlo simulations failed.")
        return None, None, [], None, None, []

    mean_metric = np.mean(metric_values)
    std_metric = np.std(metric_values, ddof=1)
    mean_d80 = np.mean(d80_values)
    std_d80 = np.std(d80_values, ddof=1)

    if use_ks_statistic:
        valid_p_values = [p for p in p_values if p is not None]
        mean_p_value = np.mean(valid_p_values) if valid_p_values else None
        std_p_value = np.std(valid_p_values, ddof=1) if valid_p_values else None
    else:
        mean_p_value = None
        std_p_value = None

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


def analyze_monte_carlo_error(
    tel_model, site_model, args_dict, data_to_plot, radius, n_simulations=500
):
    """
    Analyze Monte Carlo error on the optimization metric by running multiple simulations.

    This function runs multiple simulations with the same parameters to analyze error.

    Parameters
    ----------
    tel_model : TelescopeModel
        Telescope model object
    site_model : SiteModel
        Site model object
    args_dict : dict
        Arguments dictionary
    data_to_plot : array
        Measured PSF data
    radius : array
        Radius array
    n_simulations : int
        Number of Monte Carlo simulations to run

    Returns
    -------
    tuple
        (mean_metric, std_metric, metric_values, mean_p_value, std_p_value, p_values,
         mean_d80, std_d80, d80_values) -
        Mean optimization metric, standard deviation, and all metric values, plus p-value
        statistics (if using KS statistic)
    """
    if data_to_plot is None or radius is None:
        logger.error("No PSF measurement data provided. Cannot analyze Monte Carlo error.")
        return None, None, []

    # Get initial parameters from the database
    initial_params = get_previous_values(tel_model)
    for param_name, param_values in initial_params.items():
        logger.info(f"  {param_name}: {param_values}")

    use_ks_statistic = args_dict.get("ks_statistic", False)
    metric_values = []
    p_values = []
    d80_values = []

    for i in range(n_simulations):
        d80, metric, p_value, success = _run_monte_carlo_simulation(
            tel_model,
            site_model,
            args_dict,
            initial_params,
            data_to_plot,
            radius,
            use_ks_statistic,
            i,
        )

        if success:
            metric_values.append(metric)
            d80_values.append(d80)
            p_values.append(p_value)

    return _calculate_monte_carlo_statistics(metric_values, d80_values, p_values, use_ks_statistic)


def write_monte_carlo_analysis(mc_results, output_dir, tel_model, use_ks_statistic=False):
    """
    Write Monte Carlo analysis results to a log file.

    Parameters
    ----------
    mc_results : tuple
        Results from analyze_monte_carlo_error
    output_dir : Path
        Output directory path
    tel_model : TelescopeModel
        Telescope model object for filename generation
    use_ks_statistic : bool
        Whether KS statistic or RMSD was used

    Returns
    -------
    Path
        Path to the created log file
    """
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
        header = _create_log_file_header(
            f"Monte Carlo {metric_name} Error Analysis",
            tel_model,
            {"Number of simulations": len(metric_values)},
        )
        f.write(header)

        f.write("MONTE CARLO SIMULATION RESULTS:\n")
        f.write(f"Number of successful simulations: {len(metric_values)}\n\n")

        f.write(f"{metric_name.upper()} STATISTICS:\n")
        f.write(f"Mean {metric_name.lower()}: {mean_metric:.6f}\n")
        f.write(f"Standard deviation: {std_metric:.6f}\n")
        f.write(f"Minimum {metric_name.lower()}: {min(metric_values):.6f}\n")
        f.write(f"Maximum {metric_name.lower()}: {max(metric_values):.6f}\n")
        f.write(f"Relative error: {(std_metric / mean_metric) * 100:.2f}%\n\n")

        if use_ks_statistic and mean_p_value is not None:
            f.write("P-VALUE STATISTICS:\n")
            f.write(f"Mean p-value: {mean_p_value:.6f}\n")
            f.write(f"Standard deviation: {std_p_value:.6f}\n")
            valid_p_values = [p for p in p_values if p is not None]
            f.write(f"Minimum p-value: {min(valid_p_values):.6f}\n")
            f.write(f"Maximum p-value: {max(valid_p_values):.6f}\n")
            f.write(f"Relative error: {(std_p_value / mean_p_value) * 100:.2f}%\n")

            # Statistical significance analysis
            good_fits = sum(1 for p in valid_p_values if p > 0.05)
            fair_fits = sum(1 for p in valid_p_values if 0.01 < p <= 0.05)
            poor_fits = sum(1 for p in valid_p_values if p <= 0.01)
            f.write(
                f"Good fits (p > 0.05): {good_fits}/{len(valid_p_values)} "
                f"({100 * good_fits / len(valid_p_values):.1f}%)\n"
            )
            f.write(
                f"Fair fits (0.01 < p <= 0.05): {fair_fits}/{len(valid_p_values)} "
                f"({100 * fair_fits / len(valid_p_values):.1f}%)\n"
            )
            f.write(
                f"Poor fits (p <= 0.01): {poor_fits}/{len(valid_p_values)} "
                f"({100 * poor_fits / len(valid_p_values):.1f}%)\n\n"
            )

        f.write("D80 STATISTICS:\n")
        f.write(f"Mean D80: {mean_d80:.6f} cm\n")
        f.write(f"Standard deviation: {std_d80:.6f} cm\n")
        f.write(f"Minimum D80: {min(d80_values):.6f} cm\n")
        f.write(f"Maximum D80: {max(d80_values):.6f} cm\n")
        f.write(f"Relative error: {(std_d80 / mean_d80) * 100:.2f}%\n\n")

        f.write("INDIVIDUAL SIMULATION RESULTS:\n")
        for i, (metric_val, p_value, d80) in enumerate(zip(metric_values, p_values, d80_values)):
            if use_ks_statistic and p_value is not None:
                significance = _get_significance_level(p_value)
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
    mc_results = analyze_monte_carlo_error(tel_model, site_model, args_dict, data_to_plot, radius)

    if mc_results[0] is not None:
        mc_file = write_monte_carlo_analysis(mc_results, output_dir, tel_model, use_ks_statistic)
        logger.info(f"Monte Carlo analysis results written to {mc_file}")

        mc_plot_file = output_dir.joinpath(f"monte_carlo_uncertainty_{tel_model.name}.pdf")
        plot_psf.create_monte_carlo_uncertainty_plot(mc_results, mc_plot_file, use_ks_statistic)


def _create_iteration_plots(args_dict, output_dir, tel_model, gd_results, data_to_plot):
    """Create individual plots for each iteration if enabled."""
    if not args_dict.get("save_plots", False):
        return

    pdf_filename = output_dir.joinpath(f"psf_optimization_results_{tel_model.name}.pdf")
    pdf_pages = PdfPages(pdf_filename)

    logger.info(f"Creating PSF plots for each optimization iteration (saving to {pdf_filename})")

    for i, (params, rmsd, d80) in enumerate(gd_results):
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


def _create_convergence_and_reports(
    gd_results,
    threshold,
    output_dir,
    tel_model,
    best_pars,
    best_d80,
    use_ks_statistic,
    site_model,
    args_dict,
):
    """Create convergence plots and write reports."""
    convergence_plot_file = output_dir.joinpath(
        f"gradient_descent_convergence_{tel_model.name}.pdf"
    )
    plot_psf.create_gradient_descent_convergence_plot(
        gd_results, threshold, convergence_plot_file, use_ks_statistic
    )

    param_file = write_gradient_descent_log(
        gd_results, best_pars, best_d80, output_dir, tel_model, use_ks_statistic
    )
    print(f"\nGradient descent progression written to {param_file}")

    plot_psf.create_d80_vs_offaxis_plot(tel_model, site_model, args_dict, best_pars, output_dir)
    print("D80 vs off-axis angle plots created successfully")


def run_psf_optimization_workflow(tel_model, site_model, args_dict, output_dir):
    """
    Run the complete PSF parameter optimization workflow using gradient descent.

    Parameters
    ----------
    tel_model : TelescopeModel
        Telescope model object
    site_model : SiteModel
        Site model object
    args_dict : dict
        Dictionary containing parsed command-line arguments
    output_dir : Path
        Output directory path

    Returns
    -------
    None
        All results are saved to files and printed to console
    """
    data_to_plot, radius = load_and_process_data(args_dict)
    use_ks_statistic = args_dict.get("ks_statistic", False)

    if args_dict.get("monte_carlo_analysis", False):
        _handle_monte_carlo_analysis(
            tel_model, site_model, args_dict, data_to_plot, radius, output_dir, use_ks_statistic
        )
        return

    # Get gradient descent parameters with defaults
    threshold = args_dict.get("rmsd_threshold")
    learning_rate = args_dict.get("learning_rate")

    # Run gradient descent optimization
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
                "Possible cause: No PSF measurement data provided. Use --data argument to "
                "provide PSF data."
            )
        return

    _create_iteration_plots(args_dict, output_dir, tel_model, gd_results, data_to_plot)
    _create_convergence_and_reports(
        gd_results,
        threshold,
        output_dir,
        tel_model,
        best_pars,
        best_d80,
        use_ks_statistic,
        site_model,
        args_dict,
    )

    print("\nBest parameters:")
    for par, value in best_pars.items():
        print(f"{par} = {value}")

    # Calculate and display KS statistic and p-value for best parameters
    logger.info("Calculating KS statistic and p-value for best parameters...")
    _, best_ks_stat, best_p_value, best_simulated_data = run_psf_simulation(
        tel_model,
        site_model,
        args_dict,
        best_pars,
        data_to_plot,
        radius,
        pdf_pages=None,
        is_best=True,
        use_ks_statistic=True,
        return_simulated_data=True,
    )

    print("\nFinal performance metrics for best parameters:")
    print(f"KS statistic: {best_ks_stat:.6f}")
    print(f"p-value: {best_p_value:.6f}")

    # Also calculate RMSD for comparison if we were optimizing using KS statistic
    if use_ks_statistic:
        best_rmsd = calculate_rmsd(
            data_to_plot["measured"][CUMULATIVE_PSF], best_simulated_data[CUMULATIVE_PSF]
        )
        print(f"RMSD: {best_rmsd:.6f}")

    # Export best parameters as simulation model parameter files (if flag is provided)
    if args_dict.get("export_parameter_files", False):
        logger.info("Exporting best parameters as model files...")
        logger.warning("Parameter file export not yet implemented")

    print(f"\nOptimal D80: {best_d80:.3f} cm")
    optimization_method = f"gradient_descent ({'KS_statistic' if use_ks_statistic else 'RMSD'})"
    print(f"Optimization method: {optimization_method}")
    print(f"Total iterations: {len(gd_results)}")
