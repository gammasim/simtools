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
from scipy import stats

from simtools.data_model import model_data_writer as writer
from simtools.ray_tracing.ray_tracing import RayTracing
from simtools.utils import general as gen
from simtools.visualization import plot_psf
from simtools.visualization.plot_psf import DEFAULT_FRACTION, get_psf_diameter_label

logger = logging.getLogger(__name__)


# Constants
RADIUS = "Radius"
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
    header_lines.extend(["#" + "=" * 65, ""])
    return "\n".join(header_lines) + "\n"


def calculate_rmsd(data, sim):
    """Calculate RMSD between measured and simulated cumulative PSF curves."""
    return np.sqrt(np.mean((data - sim) ** 2))


def calculate_ks_statistic(data, sim):
    """Calculate the KS statistic between measured and simulated cumulative PSF curves."""
    return stats.ks_2samp(data, sim)


def get_previous_values(tel_model):
    """
    Retrieve current PSF parameter values from the telescope model.

    Parameters
    ----------
    tel_model : TelescopeModel
        Telescope model object containing parameter configurations.

    Returns
    -------
    dict
        Dictionary containing current values of PSF optimization parameters:
        - 'mirror_reflection_random_angle': Random reflection angle parameters
        - 'mirror_align_random_horizontal': Horizontal alignment parameters
        - 'mirror_align_random_vertical': Vertical alignment parameters
    """
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

    tel_model.overwrite_parameters(**pars)
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
    fraction = args_dict.get("fraction", DEFAULT_FRACTION)
    return im.get_psf(fraction=fraction), im


def run_psf_simulation(
    tel_model,
    site,
    args_dict,
    pars,
    data_to_plot,
    radius,
    pdf_pages=None,
    is_best=False,
    use_ks_statistic=False,
):
    """
    Run PSF simulation for given parameters and calculate optimization metric.

    Parameters
    ----------
    tel_model : TelescopeModel
        Telescope model object to be configured with the test parameters.
    site : Site
        Site model object with environmental conditions.
    args_dict : dict
        Dictionary containing simulation configuration arguments.
    pars : dict
        Dictionary of parameter values to test in the simulation.
    data_to_plot : dict
        Dictionary containing measured PSF data under "measured" key.
    radius : array-like
        Radius values in cm for PSF evaluation.
    pdf_pages : PdfPages, optional
        PDF pages object for saving plots (default: None).
    is_best : bool, optional
        Flag indicating if this is the best parameter set (default: False).
    use_ks_statistic : bool, optional
        If True, use KS statistic as metric; if False, use RMSD (default: False).

    Returns
    -------
    tuple of (float, float, float or None, array)
        - psf_diameter: PSF containment diameter of the simulated PSF in cm
        - metric: RMSD or KS statistic value
        - p_value: p-value from KS test (None if using RMSD)
        - simulated_data: Structured array with simulated cumulative PSF data
    """
    psf_diameter, im = _run_ray_tracing_simulation(tel_model, site, args_dict, pars)

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
            psf_diameter,
            metric,
            is_best,
            pdf_pages,
            fraction=args_dict.get("fraction", DEFAULT_FRACTION),
            p_value=p_value,
            use_ks_statistic=use_ks_statistic,
        )
        del data_to_plot["simulated"]

    return psf_diameter, metric, p_value, simulated_data


def load_and_process_data(args_dict):
    """
    Load and process PSF measurement data from ECSV file.

    Parameters
    ----------
    args_dict : dict
        Dictionary containing command-line arguments with 'data' and 'model_path' keys.

    Returns
    -------
    tuple of (OrderedDict, array)
        - data_dict: OrderedDict with "measured" key containing structured array
          of radius and cumulative PSF data
        - radius: Array of radius values in cm

    Raises
    ------
    FileNotFoundError
        If no data file is specified in args_dict.
    """
    if args_dict["data"] is None:
        raise FileNotFoundError("No data file specified for PSF optimization.")

    data_file = gen.find_file(args_dict["data"], args_dict["model_path"])
    table = Table.read(data_file, format="ascii.ecsv")

    radius_column = next((col for col in table.colnames if "radius" in col.lower()), None)
    integral_psf_column = next((col for col in table.colnames if "integral" in col.lower()), None)

    # Create structured array with converted data
    d_type = {"names": (RADIUS, CUMULATIVE_PSF), "formats": ("f8", "f8")}
    data = np.zeros(len(table), dtype=d_type)

    data[RADIUS] = table[radius_column].to(u.cm).value
    data[CUMULATIVE_PSF] = table[integral_psf_column]
    data[CUMULATIVE_PSF] /= np.max(np.abs(data[CUMULATIVE_PSF]))  # Normalize to max = 1.0

    return OrderedDict([("measured", data)]), data[RADIUS]


def write_tested_parameters_to_file(
    results, best_pars, best_psf_diameter, output_dir, tel_model, fraction=DEFAULT_FRACTION
):
    """
    Write optimization results and tested parameters to a log file.

    Parameters
    ----------
    results : list
        List of tuples containing (parameters, ks_statistic, p_value, psf_diameter, simulated_data)
        for each tested parameter set.
    best_pars : dict
        Dictionary containing the best parameter values found.
    best_psf_diameter : float
        PSF containment diameter in cm for the best parameter set.
    output_dir : Path
        Directory where the log file will be written.
    tel_model : TelescopeModel
        Telescope model object for naming the output file.
    fraction : float, optional
        PSF containment fraction for labeling (default: 0.8).

    Returns
    -------
    Path
        Path to the created log file.
    """
    param_file = output_dir.joinpath(f"psf_optimization_{tel_model.name}.log")
    psf_label = get_psf_diameter_label(fraction)

    with open(param_file, "w", encoding="utf-8") as f:
        header = _create_log_header_and_format_value(
            "PSF Parameter Optimization Log",
            tel_model,
            {"Total parameter sets tested": len(results)},
        )
        f.write(header)

        f.write("PARAMETER TESTING RESULTS:\n")
        for i, (pars, ks_statistic, p_value, psf_diameter, _) in enumerate(results):
            status = "BEST" if pars is best_pars else "TESTED"
            f.write(
                f"[{status}] Set {i + 1:03d}: KS_stat={ks_statistic:.5f}, "
                f"p_value={p_value:.5f}, {psf_label}={psf_diameter:.5f} cm\n"
            )
            for par, value in pars.items():
                f.write(f"    {par}: {value}\n")
            f.write("\n")

        f.write("OPTIMIZATION SUMMARY:\n")
        f.write(f"Best KS statistic: {min(result[1] for result in results):.5f}\n")
        f.write(f"Best {psf_label}: {best_psf_diameter:.5f} cm\n")
        f.write("\nOPTIMIZED PARAMETERS:\n")
        for par, value in best_pars.items():
            f.write(f"{par}: {value}\n")
    return param_file


def _add_units_to_psf_parameters(best_pars):
    """Add astropy units to PSF parameters based on their schemas."""
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
    """
    Export optimized PSF parameters as simulation model parameter files.

    Parameters
    ----------
    best_pars : dict
        Dictionary containing the optimized parameter values.
    telescope : str
        Telescope name for the parameter files.
    parameter_version : str
        Version string for the parameter files.
    output_dir : Path
        Base directory for parameter file output.

    Notes
    -----
    Creates individual JSON files for each optimized parameter with
    units. Files are saved in the format:
    {output_dir}/{telescope}/{parameter_name}-{parameter_version}.json

    Raises
    ------
    ValueError, KeyError, OSError
        If parameter export fails due to invalid values, missing keys, or file I/O errors.
    """
    try:
        psf_pars_with_units = _add_units_to_psf_parameters(best_pars)
        parameter_output_path = output_dir.parent / telescope
        for parameter_name, parameter_value in psf_pars_with_units.items():
            writer.ModelDataWriter.dump_model_parameter(
                parameter_name=parameter_name,
                value=parameter_value,
                instrument=telescope,
                parameter_version=parameter_version,
                output_file=f"{parameter_name}-{parameter_version}.json",
                output_path=parameter_output_path,
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
    """
    Calculate numerical gradient for a single parameter using finite differences.

    The gradient is calculated using forward finite differences:
    gradient = (f(x + epsilon) - f(x)) / epsilon

    Parameters
    ----------
    tel_model : TelescopeModel
        The telescope model object containing the current parameter configuration.
    site_model : SiteModel
        The site model object with environmental conditions.
    args_dict : dict
        Dictionary containing simulation arguments and configuration options.
    current_params : dict
        Dictionary of current parameter values for all optimization parameters.
    data_to_plot : dict
        Dictionary containing measured PSF data with "measured" key.
    radius : array-like
        Radius values in cm for PSF evaluation.
    current_rmsd : float
        Current RMSD at the current parameter configuration.
    param_name : str
        Name of the parameter for which to calculate the gradient.
    param_values : float or list
        Current value(s) of the parameter. Can be a single value or list of values.
    epsilon : float
        Small perturbation value for finite difference calculation.
    use_ks_statistic : bool
        If True, calculate gradient with respect to KS statistic; if False, use RMSD.

    Returns
    -------
    float or list
        Gradient value(s) for the parameter. Returns a single float if param_values
        is a single value, or a list of gradients if param_values is a list.

    If a simulation fails during gradient calculation, a gradient of 0.0 is assigned
    for that component to ensure the optimization can continue.
    """
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
    """
    Calculate numerical gradients for all optimization parameters.

    Parameters
    ----------
    tel_model : TelescopeModel
        Telescope model object for simulations.
    site_model : SiteModel
        Site model object with environmental conditions.
    args_dict : dict
        Dictionary containing simulation configuration arguments.
    current_params : dict
        Dictionary of current parameter values for all optimization parameters.
    data_to_plot : dict
        Dictionary containing measured PSF data.
    radius : array-like
        Radius values in cm for PSF evaluation.
    current_rmsd : float
        Current RMSD or KS statistic value.
    epsilon : float, optional
        Perturbation value for finite difference calculation (default: 0.0005).
    use_ks_statistic : bool, optional
        If True, calculate gradients for KS statistic; if False, use RMSD (default: False).

    Returns
    -------
    dict
        Dictionary mapping parameter names to their gradient values.
        For parameters with multiple components, gradients are returned as lists.
    """
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
    """
    Apply gradient descent step to update parameters.

    Parameters
    ----------
    current_params : dict
        Dictionary of current parameter values.
    gradients : dict
        Dictionary of gradient values for each parameter.
    learning_rate : float
        Step size for the gradient descent update.

    Returns
    -------
    dict
        Dictionary of updated parameter values after applying the gradient step.
    """
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
    """
    Attempt gradient descent step with adaptive learning rate reduction on rejection.

    The learning rate reduction strategy follows these rules:
    - If step is rejected: learning_rate *= 0.7
    - If attempt number < number of max retries then try again
    - If learning_rate drops below 1e-5: reset to 0.001
    - If all retries fail: returns None values with step_accepted=False

    This adaptive approach helps navigate local minima and ensures robust convergence
    by automatically adjusting the step size based on optimization progress.

    Parameters
    ----------
    tel_model : TelescopeModel
        Telescope model object containing the current parameter configuration.
    site_model : SiteModel
        Site model object with environmental conditions for ray tracing simulations.
    args_dict : dict
        Dictionary containing simulation configuration arguments and settings.
    current_params : dict
        Dictionary of current parameter values for all optimization parameters.
    current_metric : float
        Current optimization metric value (RMSD or KS statistic) to improve upon.
    data_to_plot : dict
        Dictionary containing measured PSF data under "measured" key for comparison.
    radius : array-like
        Radius values in cm for PSF evaluation and comparison.
    learning_rate : float
        Initial learning rate for the gradient descent step.
    max_retries : int, optional
        Maximum number of attempts with learning rate reduction (default: 3).

    Returns
    -------
    tuple of (dict, float, float, float or None, array, bool, float)
        - new_params: Updated parameter dictionary if step accepted, None if rejected
        - new_psf_diameter: PSF containment diameter in cm for new parameters, None if step rejected
        - new_metric: New optimization metric value, None if step rejected
        - new_p_value: p-value from KS test if applicable, None otherwise
        - new_simulated_data: Simulated PSF data array, None if step rejected
        - step_accepted: Boolean indicating if any step was accepted
        - final_learning_rate: Learning rate after potential reductions

    """
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

            new_psf_diameter, new_metric, new_p_value, new_simulated_data = run_psf_simulation(
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
                    new_psf_diameter,
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


def _create_step_plot(
    pdf_pages,
    args_dict,
    data_to_plot,
    current_params,
    new_psf_diameter,
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
        new_psf_diameter,
        new_metric,
        False,
        pdf_pages,
        fraction=args_dict.get("fraction", DEFAULT_FRACTION),
        p_value=new_p_value,
        use_ks_statistic=False,
    )
    del data_to_plot["simulated"]


def _create_final_plot(
    pdf_pages,
    tel_model,
    site_model,
    args_dict,
    best_params,
    data_to_plot,
    radius,
    best_psf_diameter,
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
        best_psf_diameter,
        best_rmsd,
        True,
        pdf_pages,
        fraction=args_dict.get("fraction", DEFAULT_FRACTION),
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
    """
    Run gradient descent optimization to minimize PSF fitting metric.

    Parameters
    ----------
    tel_model : TelescopeModel
        Telescope model object to be optimized.
    site_model : SiteModel
        Site model object with environmental conditions.
    args_dict : dict
        Dictionary containing simulation configuration arguments.
    data_to_plot : dict
        Dictionary containing measured PSF data under "measured" key.
    radius : array-like
        Radius values in cm for PSF evaluation.
    rmsd_threshold : float
        Convergence threshold for RMSD improvement.
    learning_rate : float
        Initial learning rate for gradient descent steps.
    output_dir : Path
        Directory for saving optimization plots and results.

    Returns
    -------
    tuple of (dict, float, list)
        - best_params: Dictionary of optimized parameter values
        - best_psf_diameter: PSF containment diameter in cm for the best parameters
        - results: List of (params, metric, p_value, psf_diameter, simulated_data)
          for each iteration

    Returns None values if optimization fails or no measurement data is provided.
    """
    if data_to_plot is None or radius is None:
        logger.error("No PSF measurement data provided. Cannot run optimization.")
        return None, None, []

    current_params = get_previous_values(tel_model)
    pdf_pages = plot_psf.setup_pdf_plotting(args_dict, output_dir, tel_model.name)
    results = []

    # Evaluate initial parameters
    current_psf_diameter, current_metric, current_p_value, simulated_data = run_psf_simulation(
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
        (
            current_params.copy(),
            current_metric,
            current_p_value,
            current_psf_diameter,
            simulated_data,
        )
    )
    best_metric, best_params, best_psf_diameter = (
        current_metric,
        current_params.copy(),
        current_psf_diameter,
    )

    logger.info(f"Initial RMSD: {current_metric:.6f}, PSF diameter: {current_psf_diameter:.6f} cm")

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
            new_psf_diameter,
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
        current_params, current_metric, current_psf_diameter = (
            new_params,
            new_metric,
            new_psf_diameter,
        )
        results.append(
            (current_params.copy(), current_metric, None, current_psf_diameter, new_simulated_data)
        )

        if current_metric < best_metric:
            best_metric, best_params, best_psf_diameter = (
                current_metric,
                current_params.copy(),
                current_psf_diameter,
            )

        _create_step_plot(
            pdf_pages,
            args_dict,
            data_to_plot,
            current_params,
            new_psf_diameter,
            new_metric,
            new_p_value,
            new_simulated_data,
        )
        logger.info(f"  Accepted step: improved to {new_metric:.6f}")

    _create_final_plot(
        pdf_pages,
        tel_model,
        site_model,
        args_dict,
        best_params,
        data_to_plot,
        radius,
        best_psf_diameter,
    )
    return best_params, best_psf_diameter, results


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


def _write_iteration_entry(
    f,
    iteration,
    pars,
    metric,
    p_value,
    psf_diameter,
    use_ks_statistic,
    metric_name,
    total_iterations,
    fraction=DEFAULT_FRACTION,
):
    """Write a single iteration entry."""
    status = "FINAL" if iteration == total_iterations - 1 else f"ITER-{iteration:02d}"

    if use_ks_statistic and p_value is not None:
        significance = plot_psf.get_significance_label(p_value)
        label = get_psf_diameter_label(fraction)
        f.write(
            f"[{status}] Iteration {iteration}: KS_stat={metric:.6f}, "
            f"p_value={p_value:.6f} ({significance}), {label}={psf_diameter:.6f} cm\n"
        )
    else:
        label = get_psf_diameter_label(fraction)
        f.write(
            f"[{status}] Iteration {iteration}: {metric_name}={metric:.6f}, "
            f"{label}={psf_diameter:.6f} cm\n"
        )

    for par, value in pars.items():
        f.write(f"    {par}: {_create_log_header_and_format_value(None, None, None, value)}\n")
    f.write("\n")


def _write_optimization_summary(
    f, gd_results, best_pars, best_psf_diameter, metric_name, fraction=DEFAULT_FRACTION
):
    """Write optimization summary section."""
    f.write("OPTIMIZATION SUMMARY:\n")
    best_metric_from_results = min(metric for _, metric, _, _, _ in gd_results)
    f.write(f"Best {metric_name.lower()}: {best_metric_from_results:.6f}\n")

    label = get_psf_diameter_label(fraction)
    f.write(
        f"Best {label}: {best_psf_diameter:.6f} cm\n"
        if best_psf_diameter is not None
        else f"Best {label}: N/A\n"
    )
    f.write(f"Total iterations: {len(gd_results)}\n\nFINAL OPTIMIZED PARAMETERS:\n")
    for par, value in best_pars.items():
        f.write(f"{par}: {_create_log_header_and_format_value(None, None, None, value)}\n")


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
    Write gradient descent optimization progression to a log file.

    Parameters
    ----------
    gd_results : list
        List of tuples containing (params, metric, p_value, psf_diameter, simulated_data)
        for each optimization iteration.
    best_pars : dict
        Dictionary containing the best parameter values found.
    best_psf_diameter : float
        PSF containment diameter in cm for the best parameter set.
    output_dir : Path
        Directory where the log file will be written.
    tel_model : TelescopeModel
        Telescope model object for naming the output file.
    use_ks_statistic : bool, optional
        If True, log KS statistic values; if False, log RMSD values (default: False).
    fraction : float, optional
        PSF containment fraction for labeling (default: 0.8).

    Returns
    -------
    Path
        Path to the created log file.
    """
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

        for iteration, (pars, metric, p_value, psf_diameter, _) in enumerate(gd_results):
            _write_iteration_entry(
                f,
                iteration,
                pars,
                metric,
                p_value,
                psf_diameter,
                use_ks_statistic,
                metric_name,
                len(gd_results),
                fraction,
            )

        _write_optimization_summary(
            f, gd_results, best_pars, best_psf_diameter, metric_name, fraction
        )

    return param_file


def analyze_monte_carlo_error(
    tel_model, site_model, args_dict, data_to_plot, radius, n_simulations=500
):
    """
    Analyze Monte Carlo uncertainty in PSF optimization metrics.

    Runs multiple simulations with the same parameters to quantify the
    statistical uncertainty in the optimization metric due to Monte Carlo
    noise in the ray tracing simulations. Returns None values if no
    measurement data is provided or all simulations fail.

    Parameters
    ----------
    tel_model : TelescopeModel
        Telescope model object with current parameter configuration.
    site_model : SiteModel
        Site model object with environmental conditions.
    args_dict : dict
        Dictionary containing simulation configuration arguments.
    data_to_plot : dict
        Dictionary containing measured PSF data under "measured" key.
    radius : array-like
        Radius values in cm for PSF evaluation.
    n_simulations : int, optional
        Number of Monte Carlo simulations to run (default: 500).

    Returns
    -------
    tuple of (float, float, list, float, float, list, float, float, list)
        - mean_metric: Mean RMSD or KS statistic value
        - std_metric: Standard deviation of metric values
        - metric_values: List of all metric values from simulations
        - mean_p_value: Mean p-value (None if using RMSD)
        - std_p_value: Standard deviation of p-values (None if using RMSD)
        - p_values: List of all p-values from simulations
        - mean_psf_diameter: Mean PSF containment diameter in cm
        - std_psf_diameter: Standard deviation of PSF diameter values
        - psf_diameter_values: List of all PSF diameter values from simulations
    """
    if data_to_plot is None or radius is None:
        logger.error("No PSF measurement data provided. Cannot analyze Monte Carlo error.")
        return None, None, []

    initial_params = get_previous_values(tel_model)
    for param_name, param_values in initial_params.items():
        logger.info(f"  {param_name}: {param_values}")

    use_ks_statistic = args_dict.get("ks_statistic", False)
    metric_values, p_values, psf_diameter_values = [], [], []

    for i in range(n_simulations):
        try:
            psf_diameter, metric, p_value, _ = run_psf_simulation(
                tel_model,
                site_model,
                args_dict,
                initial_params,
                data_to_plot,
                radius,
                use_ks_statistic=use_ks_statistic,
            )
            metric_values.append(metric)
            psf_diameter_values.append(psf_diameter)
            p_values.append(p_value)
        except (ValueError, RuntimeError) as e:
            logger.warning(f"WARNING: Simulation {i + 1} failed: {e}")

    if not metric_values:
        logger.error("All Monte Carlo simulations failed.")
        return None, None, [], None, None, []

    mean_metric, std_metric = np.mean(metric_values), np.std(metric_values, ddof=1)
    mean_psf_diameter, std_psf_diameter = (
        np.mean(psf_diameter_values),
        np.std(psf_diameter_values, ddof=1),
    )

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
        mean_psf_diameter,
        std_psf_diameter,
        psf_diameter_values,
    )


def write_monte_carlo_analysis(
    mc_results, output_dir, tel_model, use_ks_statistic=False, fraction=DEFAULT_FRACTION
):
    """
    Write Monte Carlo uncertainty analysis results to a log file.

    Parameters
    ----------
    mc_results : tuple
        Tuple of Monte Carlo analysis results from analyze_monte_carlo_error().
    output_dir : Path
        Directory where the log file will be written.
    tel_model : TelescopeModel
        Telescope model object for naming the output file.
    use_ks_statistic : bool, optional
        If True, analyze KS statistic results; if False, analyze RMSD results (default: False).
    fraction : float, optional
        PSF containment fraction for labeling (default: 0.8).

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
                if p_value > 0.05:
                    significance = "GOOD"
                elif p_value > 0.01:
                    significance = "FAIR"
                else:
                    significance = "POOR"
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


def _handle_monte_carlo_analysis(
    tel_model, site_model, args_dict, data_to_plot, radius, output_dir, use_ks_statistic
):
    """Handle Monte Carlo analysis if requested."""
    if not args_dict.get("monte_carlo_analysis", False):
        return False

    mc_results = analyze_monte_carlo_error(tel_model, site_model, args_dict, data_to_plot, radius)
    if mc_results[0] is not None:
        mc_file = write_monte_carlo_analysis(
            mc_results,
            output_dir,
            tel_model,
            use_ks_statistic,
            args_dict.get("fraction", DEFAULT_FRACTION),
        )
        logger.info(f"Monte Carlo analysis results written to {mc_file}")
        mc_plot_file = output_dir.joinpath(f"monte_carlo_uncertainty_{tel_model.name}.pdf")
        plot_psf.create_monte_carlo_uncertainty_plot(
            mc_results, mc_plot_file, args_dict.get("fraction", DEFAULT_FRACTION), use_ks_statistic
        )
    return True


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

    best_pars, best_psf_diameter, gd_results = run_gradient_descent_optimization(
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

    plot_psf.create_optimization_plots(args_dict, gd_results, tel_model, data_to_plot, output_dir)

    convergence_plot_file = output_dir.joinpath(
        f"gradient_descent_convergence_{tel_model.name}.png"
    )
    plot_psf.create_gradient_descent_convergence_plot(
        gd_results,
        threshold,
        convergence_plot_file,
        args_dict.get("fraction", DEFAULT_FRACTION),
        use_ks_statistic,
    )

    param_file = write_gradient_descent_log(
        gd_results,
        best_pars,
        best_psf_diameter,
        output_dir,
        tel_model,
        use_ks_statistic,
        args_dict.get("fraction", DEFAULT_FRACTION),
    )
    logger.info(f"\nGradient descent progression written to {param_file}")

    plot_psf.create_psf_vs_offaxis_plot(tel_model, site_model, args_dict, best_pars, output_dir)

    if args_dict.get("write_psf_parameters", False):
        logger.info("Exporting best parameters as model files...")
        export_psf_parameters(
            best_pars, args_dict.get("telescope"), args_dict.get("parameter_version"), output_dir
        )
