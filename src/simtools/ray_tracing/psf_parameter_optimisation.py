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
from matplotlib.backends.backend_pdf import PdfPages

from simtools.data_model import model_data_writer as writer
from simtools.ray_tracing.ray_tracing import RayTracing
from simtools.utils import general as gen
from simtools.visualization import plot_psf

logger = logging.getLogger(__name__)

# Constants
RADIUS_CM = "Radius [cm]"
CUMULATIVE_PSF = "Cumulative PSF"


def calculate_rmsd(data, sim):
    """Calculate RMSD between measured and simulated cumulative PSF curves."""
    return np.sqrt(np.mean((data - sim) ** 2))


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

    parameters = {
        "mirror_reflection_random_angle": [mrra_0, mfr_0, mrra2_0],
        "mirror_align_random_horizontal": mirror_align_h,
        "mirror_align_random_vertical": mirror_align_v,
    }

    logger.debug(
        "Previous parameter values:\n"
        f"mirror_reflection_random_angle = {parameters['mirror_reflection_random_angle']}\n"
        f"mirror_align_random_horizontal = {parameters['mirror_align_random_horizontal']}\n"
        f"mirror_align_random_vertical = {parameters['mirror_align_random_vertical']}\n"
    )
    return parameters


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
):
    """
    Run the simulation for one set of parameters and return D80, RMSD.

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

    Returns
    -------
    tuple
        (d80, rmsd) if return_simulated_data=False
        (d80, rmsd, simulated_data) if return_simulated_data=True
    """
    d80, im = _run_ray_tracing_simulation(tel_model, site_model, args_dict, pars)

    if radius is None:
        raise ValueError("Radius data is not available.")

    simulated_data = im.get_cumulative_data(radius * u.cm)
    rmsd = calculate_rmsd(data_to_plot["measured"][CUMULATIVE_PSF], simulated_data[CUMULATIVE_PSF])

    # Handle plotting if requested
    if pdf_pages is not None and args_dict.get("plot_all", False):
        data_to_plot["simulated"] = simulated_data
        plot_psf.create_psf_parameter_plot(data_to_plot, pars, d80, rmsd, is_best, pdf_pages)
        del data_to_plot["simulated"]

    return (d80, rmsd, simulated_data) if return_simulated_data else (d80, rmsd)


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
        # TODO - change to astropy
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
        List of (pars, rmsd, d80, simulated_data) tuples
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
        f.write("# PSF Parameter Optimization Log\n")
        f.write(f"# Telescope: {tel_model.name}\n")
        f.write(f"# Total parameter sets tested: {len(results)}\n")
        f.write("#" + "=" * 60 + "\n\n")

        f.write("PARAMETER TESTING RESULTS:\n")
        for i, (pars, rmsd, d80, _) in enumerate(results):
            is_best = pars is best_pars
            status = "BEST" if is_best else "TESTED"
            f.write(f"[{status}] Set {i + 1:03d}: RMSD={rmsd:.5f}, D80={d80:.5f} cm\n")
            for par, value in pars.items():
                f.write(f"    {par}: {value}\n")
            f.write("\n")

        f.write("OPTIMIZATION SUMMARY:\n")
        f.write(f"Best RMSD: {min(result[1] for result in results):.5f}\n")
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
        logger.info("Exporting best PSF parameters as simulation model parameter files")
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


def calculate_gradient(
    tel_model,
    site_model,
    args_dict,
    current_params,
    data_to_plot,
    radius,
    current_rmsd,
    epsilon=0.0005,
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
        gradients[param_name] = []

        if isinstance(param_values, list):
            values_list = param_values
        else:
            values_list = [param_values]

        for i, value in enumerate(values_list):
            perturbed_params = {
                k: v.copy() if isinstance(v, list) else v for k, v in current_params.items()
            }

            if isinstance(param_values, list):
                perturbed_params[param_name][i] = value + epsilon
            else:
                perturbed_params[param_name] = value + epsilon

            try:
                _, perturbed_rmsd = run_psf_simulation(
                    tel_model, site_model, args_dict, perturbed_params, data_to_plot, radius
                )
                gradient = (perturbed_rmsd - current_rmsd) / epsilon
                gradients[param_name].append(gradient)
            except (ValueError, RuntimeError):
                # If simulation fails, assume zero gradient
                gradients[param_name].append(0.0)

        if not isinstance(param_values, list):
            gradients[param_name] = gradients[param_name][0]

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


def run_gradient_descent_optimization(
    tel_model, site_model, args_dict, data_to_plot, radius, rmsd_threshold, learning_rate
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

    Returns
    -------
    tuple
        (best_params, best_d80, results_list)
        results_list contains (iteration, params, rmsd, d80) tuples
    """
    if data_to_plot is None or radius is None:
        logger.error("No PSF measurement data provided. Cannot run optimization.")
        return None, None, []

    current_params = get_previous_values(tel_model)

    results = []
    best_rmsd = float("inf")
    best_params = current_params.copy()
    best_d80 = None

    # Evaluate initial parameters and store as first result
    if args_dict.get("plot_all", False):
        current_d80, current_rmsd, simulated_data = run_psf_simulation(
            tel_model,
            site_model,
            args_dict,
            current_params,
            data_to_plot,
            radius,
            return_simulated_data=True,
        )
    else:
        current_d80, current_rmsd = run_psf_simulation(
            tel_model,
            site_model,
            args_dict,
            current_params,
            data_to_plot,
            radius,
            return_simulated_data=False,
        )
        simulated_data = None  # Save memory when plotting is disabled

    results.append((current_params.copy(), current_rmsd, current_d80, simulated_data))
    best_rmsd = current_rmsd
    best_params = current_params.copy()
    best_d80 = current_d80

    logger.info(f"Initial RMSD: {current_rmsd:.6f}, D80: {current_d80:.6f} cm")

    iteration = 0
    max_total_iterations = 100 if not args_dict.get("monte_carlo_analysis", False) else 1

    while iteration < max_total_iterations:
        iteration += 1
        logger.info(f"Gradient descent iteration {iteration}")

        # Check convergence first
        if current_rmsd <= rmsd_threshold:
            logger.info(f"RMSD threshold {rmsd_threshold} reached! Stopping optimization.")
            break

        # Try gradient step with current learning rate
        step_accepted = False
        max_retries = 3  # Limit retries for failed simulations
        retries = 0

        while not step_accepted and retries < max_retries:
            try:
                gradients = calculate_gradient(
                    tel_model,
                    site_model,
                    args_dict,
                    current_params,
                    data_to_plot,
                    radius,
                    current_rmsd,
                )

                # Apply gradient step
                new_params = apply_gradient_step(current_params, gradients, learning_rate)

                if args_dict.get("plot_all", False):
                    new_d80, new_rmsd, new_simulated_data = run_psf_simulation(
                        tel_model,
                        site_model,
                        args_dict,
                        new_params,
                        data_to_plot,
                        radius,
                        return_simulated_data=True,
                    )
                else:
                    new_d80, new_rmsd = run_psf_simulation(
                        tel_model,
                        site_model,
                        args_dict,
                        new_params,
                        data_to_plot,
                        radius,
                        return_simulated_data=False,
                    )
                    new_simulated_data = None

                # Accept step if RMSD reduces, otherwise reduce learning rate and retry
                if new_rmsd < current_rmsd:
                    current_params = new_params
                    current_rmsd = new_rmsd
                    current_d80 = new_d80
                    results.append(
                        (current_params.copy(), current_rmsd, current_d80, new_simulated_data)
                    )

                    if current_rmsd < best_rmsd:
                        best_rmsd = current_rmsd
                        best_params = current_params.copy()
                        best_d80 = current_d80

                    logger.info(f"  Accepted step: RMSD improved to {new_rmsd:.6f}")
                    step_accepted = True
                else:
                    learning_rate *= 0.9
                    retries += 1
                    logger.info(
                        f"  Rejected step: RMSD would increase from {current_rmsd:.6f} to "
                        f"{new_rmsd:.6f}"
                    )

                    if learning_rate < 1e-4:
                        logger.info(
                            "Learning rate getting too small for this iteration, "
                            "moving to next iteration."
                        )
                        break

            except (ValueError, RuntimeError, KeyError) as e:
                logger.error(f"Error in gradient descent iteration {iteration}: {e}")
                retries += 1
                if retries >= max_retries:
                    logger.warning("Too many retries, moving to next iteration")
                    break

        # Check for convergence based on RMSD threshold
        if best_rmsd <= rmsd_threshold:
            logger.info(
                f"Convergence achieved: RMSD {best_rmsd:.6f} <= threshold {rmsd_threshold:.6f}"
            )
            break

        # Increase when no step accepted after multiple attempts (escape local minimum)
        if not step_accepted:
            learning_rate *= 1.5

            if learning_rate > 0.5:  # Prevent runaway learning rate
                logger.warning("Learning rate getting very large - optimization may be stuck")
                break

    return best_params, best_d80, results


def write_gradient_descent_log(gd_results, best_pars, best_d80, output_dir, tel_model):
    """
    Write gradient descent progression to a log file.

    Parameters
    ----------
    gd_results : list
        List of (iteration, params, rmsd, d80) tuples from gradient descent
    best_pars : dict
        Best parameter set
    best_d80 : float
        Best D80 value
    output_dir : Path
        Output directory path
    tel_model : TelescopeModel
        Telescope model object for filename generation

    Returns
    -------
    Path
        Path to the created log file
    """
    param_file = output_dir.joinpath(f"psf_gradient_descent_{tel_model.name}.log")
    with open(param_file, "w", encoding="utf-8") as f:
        f.write("# PSF Parameter Optimization - Gradient Descent Progression\n")
        f.write(f"# Telescope: {tel_model.name}\n")
        f.write(f"# Total iterations: {len(gd_results)}\n")
        f.write("#" + "=" * 60 + "\n\n")

        f.write("GRADIENT DESCENT PROGRESSION:\n")
        f.write("(Each entry shows the parameters chosen at each iteration)\n\n")

        for iteration, (pars, rmsd, d80, _) in enumerate(gd_results):
            is_final = iteration == len(gd_results) - 1
            status = "FINAL" if is_final else f"ITER-{iteration:02d}"

            f.write(f"[{status}] Iteration {iteration}: RMSD={rmsd:.6f}, D80={d80:.6f} cm\n")
            for par, value in pars.items():
                if isinstance(value, list):
                    value_str = "[" + ", ".join([f"{v:.6f}" for v in value]) + "]"
                else:
                    value_str = f"{value:.6f}" if isinstance(value, int | float) else str(value)
                f.write(f"    {par}: {value_str}\n")
            f.write("\n")

        f.write("OPTIMIZATION SUMMARY:\n")
        # Calculate best RMSD from results
        best_rmsd_from_results = min(rmsd for _, rmsd, _, _ in gd_results)

        f.write(f"Best RMSD achieved: {best_rmsd_from_results:.6f}\n")
        f.write(f"Best D80: {best_d80:.6f} cm\n" if best_d80 is not None else "Best D80: N/A\n")
        f.write(f"Total iterations: {len(gd_results)}\n")
        f.write("\nFINAL OPTIMIZED PARAMETERS:\n")
        for par, value in best_pars.items():
            if isinstance(value, list):
                value_str = "[" + ", ".join([f"{v:.6f}" for v in value]) + "]"
            else:
                value_str = f"{value:.6f}" if isinstance(value, int | float) else str(value)
            f.write(f"{par}: {value_str}\n")
    return param_file


def analyze_monte_carlo_rmsd_error(
    tel_model, site_model, args_dict, data_to_plot, radius, n_simulations=500
):
    """
    Analyze Monte Carlo error on RMSD by running multiple simulations with same parameters.

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
        (mean_rmsd, std_rmsd, rmsd_values) - Mean RMSD, standard deviation, and all RMSD values
    """
    if data_to_plot is None or radius is None:
        logger.error("No PSF measurement data provided. Cannot analyze Monte Carlo error.")
        return None, None, []

    # Get initial parameters from the database
    initial_params = get_previous_values(tel_model)
    for param_name, param_values in initial_params.items():
        logger.info(f"  {param_name}: {param_values}")

    rmsd_values = []
    d80_values = []

    for i in range(n_simulations):
        try:
            d80, rmsd = run_psf_simulation(
                tel_model,
                site_model,
                args_dict,
                initial_params,
                data_to_plot,
                radius,
                return_simulated_data=False,
            )
            rmsd_values.append(rmsd)
            d80_values.append(d80)
        except (ValueError, RuntimeError) as e:
            logger.warning(f"WARNING: Simulation {i + 1} failed: {e}")
            continue

    if not rmsd_values:
        logger.error("All Monte Carlo simulations failed.")
        return None, None, []

    mean_rmsd = np.mean(rmsd_values)
    std_rmsd = np.std(rmsd_values, ddof=1)
    mean_d80 = np.mean(d80_values)
    std_d80 = np.std(d80_values, ddof=1)

    return mean_rmsd, std_rmsd, rmsd_values, mean_d80, std_d80, d80_values


def write_monte_carlo_analysis(mc_results, output_dir, tel_model):
    """
    Write Monte Carlo analysis results to a log file.

    Parameters
    ----------
    mc_results : tuple
        Results from analyze_monte_carlo_rmsd_error
    output_dir : Path
        Output directory path
    tel_model : TelescopeModel
        Telescope model object for filename generation

    Returns
    -------
    Path
        Path to the created log file
    """
    mean_rmsd, std_rmsd, rmsd_values, mean_d80, std_d80, d80_values = mc_results

    mc_file = output_dir.joinpath(f"monte_carlo_rmsd_analysis_{tel_model.name}.log")
    with open(mc_file, "w", encoding="utf-8") as f:
        f.write("# Monte Carlo RMSD Error Analysis\n")
        f.write(f"# Telescope: {tel_model.name}\n")
        f.write(f"# Number of simulations: {len(rmsd_values)}\n")
        f.write("#" + "=" * 60 + "\n\n")

        f.write("MONTE CARLO SIMULATION RESULTS:\n")
        f.write(f"Number of successful simulations: {len(rmsd_values)}\n\n")

        f.write("RMSD STATISTICS:\n")
        f.write(f"Mean RMSD: {mean_rmsd:.6f}\n")
        f.write(f"Standard deviation: {std_rmsd:.6f}\n")
        f.write(f"Minimum RMSD: {min(rmsd_values):.6f}\n")
        f.write(f"Maximum RMSD: {max(rmsd_values):.6f}\n")
        f.write(f"Relative error: {(std_rmsd / mean_rmsd) * 100:.2f}%\n\n")

        f.write("D80 STATISTICS:\n")
        f.write(f"Mean D80: {mean_d80:.6f} cm\n")
        f.write(f"Standard deviation: {std_d80:.6f} cm\n")
        f.write(f"Minimum D80: {min(d80_values):.6f} cm\n")
        f.write(f"Maximum D80: {max(d80_values):.6f} cm\n")
        f.write(f"Relative error: {(std_d80 / mean_d80) * 100:.2f}%\n\n")

        f.write("INDIVIDUAL SIMULATION RESULTS:\n")
        for i, (rmsd, d80) in enumerate(zip(rmsd_values, d80_values)):
            f.write(f"Simulation {i + 1:2d}: RMSD={rmsd:.6f}, D80={d80:.6f} cm\n")

    return mc_file


def run_psf_optimization_workflow(tel_model, site_model, args_dict, output_dir):
    """
    Run the complete PSF parameter optimization workflow using gradient descent.

    This function consolidates the gradient descent optimization logic.

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

    if args_dict.get("monte_carlo_analysis", False):
        logger.info("=" * 60)
        logger.info("MONTE CARLO RMSD ERROR ANALYSIS")
        logger.info("=" * 60)

        mc_results = analyze_monte_carlo_rmsd_error(
            tel_model, site_model, args_dict, data_to_plot, radius
        )

        if mc_results[0] is not None:  # If Monte Carlo analysis was successful
            mc_file = write_monte_carlo_analysis(mc_results, output_dir, tel_model)
            logger.info(f"Monte Carlo analysis results written to {mc_file}")

        logger.info("=" * 60)
        logger.info("PSF PARAMETER OPTIMIZATION")
        logger.info("=" * 60)

    logger.info("Running PSF optimization using gradient descent")

    # Get gradient descent parameters with defaults
    rmsd_threshold = args_dict.get("rmsd_threshold")
    learning_rate = args_dict.get("learning_rate")

    # Run gradient descent optimization
    best_pars, best_d80, gd_results = run_gradient_descent_optimization(
        tel_model,
        site_model,
        args_dict,
        data_to_plot,
        radius,
        rmsd_threshold=rmsd_threshold,
        learning_rate=learning_rate,
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

    results = gd_results

    # Create plots showing gradient descent progression
    plot_file_name = f"psf_gradient_descent_progression_{tel_model.name}.pdf"
    plot_file = output_dir.joinpath(plot_file_name)

    if args_dict.get("plot_all", False):
        pdf_pages = PdfPages(plot_file)

        for i, (pars, rmsd, d80, simulated_data) in enumerate(results):
            is_best = i == len(results) - 1

            plot_psf.create_detailed_parameter_plot(
                pars, rmsd, d80, simulated_data, data_to_plot, is_best, pdf_pages
            )

        pdf_pages.close()

    convergence_plot_file = output_dir.joinpath(
        f"gradient_descent_convergence_{tel_model.name}.pdf"
    )
    plot_psf.create_gradient_descent_convergence_plot(
        gd_results, rmsd_threshold, convergence_plot_file
    )

    param_file = write_gradient_descent_log(gd_results, best_pars, best_d80, output_dir, tel_model)
    print(f"\nGradient descent progression written to {param_file}")

    # Automatically create D80 vs off-axis angle plot for best parameters
    plot_psf.create_d80_vs_offaxis_plot(tel_model, site_model, args_dict, best_pars, output_dir)
    print("D80 vs off-axis angle plots created successfully")

    print("\nBest parameters:")
    for par, value in best_pars.items():
        print(f"{par} = {value}")

    # Export best parameters as simulation model parameter files (if flag is provided)
    if args_dict.get("write_psf_parameters", False):
        parameter_version = args_dict.get("parameter_version") or "0.0.0"
        export_psf_parameters(
            best_pars,
            tel_model,
            parameter_version,
            output_dir.parent,
        )
