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
from simtools.utils import names
from simtools.visualization import plot_psf
from simtools.visualization.plot_psf import DEFAULT_FRACTION, get_psf_diameter_label

logger = logging.getLogger(__name__)


# Constants
RADIUS = "Radius"
CUMULATIVE_PSF = "Cumulative PSF"
KS_STATISTIC_NAME = "KS statistic"


class PSFParameterOptimizer:
    """
    Gradient descent optimizer for PSF parameters.

    Parameters
    ----------
    tel_model : TelescopeModel
        Telescope model object containing parameter configurations.
    site_model : SiteModel
        Site model object with environmental conditions.
    args_dict : dict
        Dictionary containing simulation configuration arguments.
    data_to_plot : dict
        Dictionary containing measured PSF data under "measured" key.
    radius : numpy.ndarray
        Radius values in cm for PSF evaluation.
    output_dir : Path
        Directory for saving optimization results and plots.

    Attributes
    ----------
    simulation_cache : dict
        Cache for simulation results to avoid redundant ray tracing simulations.
        Key: frozenset of (param_name, tuple(values)) items
        Value: (psf_diameter, metric, p_value, simulated_data)
    """

    # Learning rate adjustment constants
    LR_REDUCTION_FACTOR = 0.7
    LR_INCREASE_FACTOR = 2.0
    LR_MINIMUM_THRESHOLD = 1e-6
    LR_RESET_VALUE = 0.0001

    def __init__(self, tel_model, site_model, args_dict, data_to_plot, radius, output_dir):
        """Initialize the PSF parameter optimizer."""
        self.tel_model = tel_model
        self.site_model = site_model
        self.args_dict = args_dict
        self.data_to_plot = data_to_plot
        self.radius = radius
        self.output_dir = output_dir
        self.use_ks_statistic = args_dict.get("ks_statistic", False)
        self.fraction = args_dict.get("fraction", DEFAULT_FRACTION)
        self.simulation_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def _params_to_cache_key(self, params):
        """Convert parameters dict to a hashable cache key."""
        items = []
        for key in sorted(params.keys()):
            value = params[key]
            if isinstance(value, list):
                items.append((key, tuple(value)))
            else:
                items.append((key, value))
        return frozenset(items)

    def _reduce_learning_rate(self, current_lr):
        """
        Reduce learning rate with minimum threshold and reset.

        Parameters
        ----------
        current_lr : float
            Current learning rate.

        Returns
        -------
        float
            Reduced learning rate, reset to LR_RESET_VALUE if below threshold.
        """
        new_lr = current_lr * self.LR_REDUCTION_FACTOR
        if new_lr < self.LR_MINIMUM_THRESHOLD:
            return self.LR_RESET_VALUE
        return new_lr

    def _increase_learning_rate(self, current_lr):
        """
        Increase learning rate.

        Parameters
        ----------
        current_lr : float
            Current learning rate.

        Returns
        -------
        float
            Increased learning rate.
        """
        return current_lr * self.LR_INCREASE_FACTOR

    def get_initial_parameters(self):
        """
        Get current PSF parameter values from the telescope model.

        Returns
        -------
        dict
            Dictionary of current parameter values.
        """
        return get_previous_values(self.tel_model)

    def run_simulation(
        self, pars, pdf_pages=None, is_best=False, use_cache=True, use_ks_statistic=None
    ):
        """
        Run PSF simulation for given parameters with optional caching.

        Parameters
        ----------
        pars : dict
            Dictionary of parameter values to test.
        pdf_pages : PdfPages, optional
            PDF pages object for saving plots.
        is_best : bool, optional
            Flag indicating if this is the best parameter set.
        use_cache : bool, optional
            If True, use cached results if available.
        use_ks_statistic : bool, optional
            If provided, override self.use_ks_statistic for this simulation.

        Returns
        -------
        tuple
            (psf_diameter, metric, p_value, simulated_data)
        """
        # Determine which statistic to use
        ks_stat = use_ks_statistic if use_ks_statistic is not None else self.use_ks_statistic

        if use_cache and pdf_pages is None and not is_best:
            cache_key = self._params_to_cache_key(pars)
            if cache_key in self.simulation_cache:
                self.cache_hits += 1
                return self.simulation_cache[cache_key]
            self.cache_misses += 1

        result = run_psf_simulation(
            self.tel_model,
            self.site_model,
            self.args_dict,
            pars,
            self.data_to_plot,
            self.radius,
            pdf_pages,
            is_best,
            ks_stat,
        )

        # Cache the result if caching is enabled and not plotting
        if use_cache and pdf_pages is None and not is_best:
            cache_key = self._params_to_cache_key(pars)
            self.simulation_cache[cache_key] = result

        return result

    def calculate_gradient(self, current_params, current_metric, epsilon=0.0005):
        """
        Calculate numerical gradients for all optimization parameters.

        Parameters
        ----------
        current_params : dict
            Dictionary of current parameter values.
        current_metric : float
            Current RMSD or KS statistic value.
        epsilon : float, optional
            Perturbation value for finite difference calculation.

        Returns
        -------
        dict or None
            Dictionary mapping parameter names to their gradient values.
            Returns None if gradient calculation fails for any parameter.
        """
        gradients = {}

        for param_name, param_values in current_params.items():
            param_gradient = self._calculate_param_gradient(
                current_params,
                current_metric,
                param_name,
                param_values,
                epsilon,
            )
            if param_gradient is None:
                return None
            gradients[param_name] = param_gradient

        return gradients

    def apply_gradient_step(self, current_params, gradients, learning_rate):
        """
        Apply gradient descent step to update parameters while preserving constraints.

        This function applies the standard gradient descent update and preserves
        zenith angle components (index 1) for mirror alignment parameters.

        Note: Use _are_all_parameters_within_allowed_range() to validate the result before
        accepting the step.

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
                updated_values = []
                for i, (value, gradient) in enumerate(zip(param_values, param_gradients)):
                    # Apply gradient descent update
                    new_value = value - learning_rate * gradient

                    # Enforce constraint: preserve zenith angle (index 1) for mirror alignment
                    if (
                        param_name
                        in ["mirror_align_random_horizontal", "mirror_align_random_vertical"]
                        and i == 1
                    ):
                        new_value = value  # Keep original zenith angle value

                    updated_values.append(new_value)

                new_params[param_name] = updated_values
            else:
                new_value = param_values - learning_rate * param_gradients
                new_params[param_name] = new_value

        return new_params

    def _calculate_param_gradient(
        self, current_params, current_metric, param_name, param_values, epsilon
    ):
        """
        Calculate numerical gradient for a single parameter using finite differences.

        Parameters
        ----------
        current_params : dict
            Dictionary of current parameter values for all optimization parameters.
        current_metric : float
            Current RMSD or KS statistic value.
        param_name : str
            Name of the parameter for which to calculate the gradient.
        param_values : float or list
            Current value(s) of the parameter.
        epsilon : float
            Small perturbation value for finite difference calculation.

        Returns
        -------
        float or list or None
            Gradient value(s) for the parameter.
            Returns None if simulation fails for any component.
        """
        param_gradients = []
        values_list = param_values if isinstance(param_values, list) else [param_values]

        for i, value in enumerate(values_list):
            perturbed_params = _create_perturbed_params(
                current_params, param_name, param_values, i, value, epsilon
            )

            # Calculate gradient for this parameter component using cached simulations
            try:
                _, perturbed_metric, _, _ = self.run_simulation(perturbed_params, use_cache=True)
                gradient = (perturbed_metric - current_metric) / epsilon
            except (ValueError, RuntimeError) as e:
                logger.warning(f"Simulation failed for {param_name}[{i}] gradient calculation: {e}")
                return None

            param_gradients.append(gradient)

        return param_gradients[0] if not isinstance(param_values, list) else param_gradients

    def perform_gradient_step_with_retries(
        self, current_params, current_metric, learning_rate, max_retries=3
    ):
        """
        Attempt gradient descent step with adaptive learning rate reduction.

        Parameters
        ----------
        current_params : dict
            Dictionary of current parameter values.
        current_metric : float
            Current optimization metric value.
        learning_rate : float
            Initial learning rate for the gradient descent step.
        max_retries : int, optional
            Maximum number of attempts with learning rate reduction.

        Returns
        -------
        tuple
            (new_params, new_psf_diameter, new_metric, new_p_value,
             new_simulated_data, step_accepted, final_learning_rate)
        """
        current_lr = learning_rate

        for attempt in range(max_retries):
            try:
                gradients = self.calculate_gradient(current_params, current_metric)

                if gradients is None:
                    logger.warning(
                        f"Gradient calculation failed on attempt {attempt + 1}, skipping step"
                    )
                    return None, None, None, None, None, False, current_lr

                new_params = self.apply_gradient_step(current_params, gradients, current_lr)

                # Validate that all parameters are within allowed ranges
                if not _are_all_parameters_within_allowed_range(new_params):
                    logger.info(
                        f"Step rejected: parameters would go out of bounds with learning rate "
                        f"{current_lr:.6f}, reducing to {current_lr * self.LR_REDUCTION_FACTOR:.6f}"
                    )
                    current_lr = self._reduce_learning_rate(current_lr)
                    continue

                new_psf_diameter, new_metric, new_p_value, new_simulated_data = self.run_simulation(
                    new_params, use_cache=True
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
                    f"reducing learning rate to {current_lr * self.LR_REDUCTION_FACTOR:.6f}"
                )
                current_lr = self._reduce_learning_rate(current_lr)

            except (ValueError, RuntimeError, KeyError) as e:
                logger.warning(f"Simulation failed on attempt {attempt + 1}: {e}")
                continue

        return None, None, None, None, None, False, current_lr

    def _create_step_plot(
        self,
        pdf_pages,
        current_params,
        new_psf_diameter,
        new_metric,
        new_p_value,
        new_simulated_data,
    ):
        """Create plot for an accepted gradient step."""
        if (
            pdf_pages is None
            or not self.args_dict.get("plot_all", False)
            or new_simulated_data is None
        ):
            return

        self.data_to_plot["simulated"] = new_simulated_data
        plot_psf.create_psf_parameter_plot(
            self.data_to_plot,
            current_params,
            new_psf_diameter,
            new_metric,
            False,
            pdf_pages,
            fraction=self.fraction,
            p_value=new_p_value,
            use_ks_statistic=self.use_ks_statistic,
        )
        del self.data_to_plot["simulated"]

    def _create_final_plot(self, pdf_pages, best_params, best_psf_diameter):
        """Create final plot for best parameters."""
        if pdf_pages is None or best_params is None:
            return

        logger.info("Creating final plot for best parameters with both RMSD and KS statistic...")
        _, best_ks_stat, best_p_value, best_simulated_data = self.run_simulation(
            best_params,
            pdf_pages=None,
            is_best=False,
            use_cache=False,
            use_ks_statistic=True,
        )
        best_rmsd = calculate_rmsd(
            self.data_to_plot["measured"][CUMULATIVE_PSF], best_simulated_data[CUMULATIVE_PSF]
        )

        self.data_to_plot["simulated"] = best_simulated_data
        plot_psf.create_psf_parameter_plot(
            self.data_to_plot,
            best_params,
            best_psf_diameter,
            best_rmsd,
            True,
            pdf_pages,
            fraction=self.fraction,
            p_value=best_p_value,
            use_ks_statistic=False,
            second_metric=best_ks_stat,
        )
        del self.data_to_plot["simulated"]
        pdf_pages.close()
        logger.info("Cumulative PSF plots saved")

    def run_gradient_descent(self, rmsd_threshold, learning_rate, max_iterations=200):
        """
        Run gradient descent optimization to minimize PSF fitting metric.

        Parameters
        ----------
        rmsd_threshold : float
            Convergence threshold for RMSD improvement.
        learning_rate : float
            Initial learning rate for gradient descent steps.
        max_iterations : int, optional
            Maximum number of optimization iterations.

        Returns
        -------
        tuple
            (best_params, best_psf_diameter, results)
        """
        if self.data_to_plot is None or self.radius is None:
            logger.error("No PSF measurement data provided. Cannot run optimization.")
            return None, None, []

        current_params = self.get_initial_parameters()
        pdf_pages = plot_psf.setup_pdf_plotting(
            self.args_dict, self.output_dir, self.tel_model.name
        )
        results = []

        # Evaluate initial parameters
        current_psf_diameter, current_metric, current_p_value, simulated_data = self.run_simulation(
            current_params,
            pdf_pages=pdf_pages if self.args_dict.get("plot_all", False) else None,
            is_best=False,
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

        logger.info(
            f"Initial RMSD: {current_metric:.6f}, PSF diameter: {current_psf_diameter:.6f} cm"
        )

        iteration = 0
        current_lr = learning_rate

        while iteration < max_iterations:
            if current_metric <= rmsd_threshold:
                logger.info(
                    f"Optimization converged: RMSD {current_metric:.6f} <= "
                    f"threshold {rmsd_threshold:.6f}"
                )
                break

            iteration += 1
            logger.info(f"Gradient descent iteration {iteration}")

            step_result = self.perform_gradient_step_with_retries(
                current_params,
                current_metric,
                current_lr,
            )
            (
                new_params,
                new_psf_diameter,
                new_metric,
                new_p_value,
                new_simulated_data,
                step_accepted,
                current_lr,
            ) = step_result

            if not step_accepted or new_params is None:
                current_lr = self._increase_learning_rate(current_lr)
                logger.info(f"No step accepted, increasing learning rate to {current_lr:.6f}")
                continue

            # Step was accepted - update state
            current_params, current_metric, current_psf_diameter = (
                new_params,
                new_metric,
                new_psf_diameter,
            )
            results.append(
                (
                    current_params.copy(),
                    current_metric,
                    None,
                    current_psf_diameter,
                    new_simulated_data,
                )
            )

            if current_metric < best_metric:
                best_metric, best_params, best_psf_diameter = (
                    current_metric,
                    current_params.copy(),
                    current_psf_diameter,
                )

            self._create_step_plot(
                pdf_pages,
                current_params,
                new_psf_diameter,
                new_metric,
                new_p_value,
                new_simulated_data,
            )
            logger.info(f"  Accepted step: improved to {new_metric:.6f}")

        self._create_final_plot(pdf_pages, best_params, best_psf_diameter)
        return best_params, best_psf_diameter, results

    def analyze_monte_carlo_error(self, n_simulations=500):
        """
        Analyze Monte Carlo uncertainty in PSF optimization metrics.

        Parameters
        ----------
        n_simulations : int, optional
            Number of Monte Carlo simulations to run.

        Returns
        -------
        tuple
            Monte Carlo analysis results.
        """
        if self.data_to_plot is None or self.radius is None:
            logger.error("No PSF measurement data provided. Cannot analyze Monte Carlo error.")
            return None, None, [], None, None, [], None, None, []

        initial_params = self.get_initial_parameters()
        for param_name, param_values in initial_params.items():
            logger.info(f"  {param_name}: {param_values}")

        metric_values, p_values, psf_diameter_values = [], [], []

        for i in range(n_simulations):
            try:
                psf_diameter, metric, p_value, _ = self.run_simulation(
                    initial_params,
                    use_cache=False,
                )
                metric_values.append(metric)
                psf_diameter_values.append(psf_diameter)
                p_values.append(p_value)
            except (ValueError, RuntimeError) as e:
                logger.warning(f"WARNING: Simulation {i + 1} failed: {e}")

        if not metric_values:
            logger.error("All Monte Carlo simulations failed.")
            return None, None, [], None, None, [], None, None, []

        mean_metric, std_metric = np.mean(metric_values), np.std(metric_values, ddof=1)
        mean_psf_diameter, std_psf_diameter = (
            np.mean(psf_diameter_values),
            np.std(psf_diameter_values, ddof=1),
        )

        if self.use_ks_statistic:
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


def _is_parameter_within_allowed_range(param_name, param_index, value):
    """
    Check if a parameter value is within the allowed range defined in its schema.

    Parameters
    ----------
    param_name : str
        Name of the parameter to check.
    param_index : int
        Index within the parameter array (for multi-component parameters).
    value : float
        The parameter value to check.

    Returns
    -------
    bool
        True if the parameter is within allowed range, False otherwise.
        Returns True if no range constraints are defined for the parameter.
    """
    try:
        param_schema = names.model_parameters().get(param_name)
        if param_schema is None:
            return True

        data = param_schema.get("data")
        if not isinstance(data, list) or len(data) <= param_index:
            return True

        param_data = data[param_index]
        allowed_range = param_data.get("allowed_range")
        if not allowed_range:
            return True

        min_val = allowed_range.get("min")
        max_val = allowed_range.get("max")

        if min_val is not None and value < min_val:
            return False
        return not (max_val is not None and value > max_val)

    except (KeyError, IndexError) as e:
        logger.warning(f"Error reading schema for {param_name}[{param_index}]: {e}")
        return True


def _are_all_parameters_within_allowed_range(params):
    """
    Check if all parameters in the parameter dictionary are within allowed ranges.

    Parameters
    ----------
    params : dict
        Dictionary of parameter values to validate.

    Returns
    -------
    bool
        True if all parameters are within allowed ranges, False otherwise.
    """
    for name, values in params.items():
        values = values if isinstance(values, list) else [values]
        for i, v in enumerate(values):
            if not _is_parameter_within_allowed_range(name, i, v):
                logger.debug(f"{name}[{i}]={v:.6f} out of range")
                return False
    return True


def _create_perturbed_params(current_params, param_name, param_values, param_index, value, epsilon):
    """
    Create parameter dictionary with one parameter perturbed by epsilon.

    Parameters
    ----------
    current_params : dict
        Dictionary of current parameter values.
    param_name : str
        Name of the parameter to perturb.
    param_values : float or list
        Current parameter values.
    param_index : int
        Index of the parameter to perturb (for list parameters).
    value : float
        Current value of the specific parameter component.
    epsilon : float
        Perturbation amount.

    Returns
    -------
    dict
        New parameter dictionary with the specified parameter perturbed.
    """
    perturbed_params = {
        k: v.copy() if isinstance(v, list) else v for k, v in current_params.items()
    }

    if isinstance(param_values, list):
        perturbed_params[param_name][param_index] = value + epsilon
    else:
        perturbed_params[param_name] = value + epsilon

    return perturbed_params


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

    tel_model.overwrite_parameters(pars)
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
    radius : numpy.ndarray
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
        parameter_output_path = output_dir.joinpath(telescope)
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


def cleanup_intermediate_files(output_dir):
    """
    Remove intermediate log and list files from the output directory.

    Parameters
    ----------
    output_dir : Path
        Directory containing output files to clean up.
    """
    patterns = ["*.log", "*.lis*"]
    files_removed = 0

    for pattern in patterns:
        for file_path in output_dir.glob(pattern):
            file_path.unlink()
            files_removed += 1
            logger.debug(f"Removed: {file_path.name}")

    if files_removed > 0:
        logger.info(f"Cleanup: removed {files_removed} intermediate files")


def run_psf_optimization_workflow(tel_model, site_model, args_dict, output_dir):
    """
    Run the complete PSF parameter optimization workflow using gradient descent.

    This function creates a PSFParameterOptimizer instance and orchestrates
    the optimization process.
    """
    data_to_plot, radius = load_and_process_data(args_dict)

    # Create optimizer instance to encapsulate state and methods
    optimizer = PSFParameterOptimizer(
        tel_model, site_model, args_dict, data_to_plot, radius, output_dir
    )

    # Handle Monte Carlo analysis if requested
    if args_dict.get("monte_carlo_analysis", False):
        mc_results = optimizer.analyze_monte_carlo_error()
        if mc_results[0] is not None:
            mc_file = write_monte_carlo_analysis(
                mc_results,
                output_dir,
                tel_model,
                optimizer.use_ks_statistic,
                optimizer.fraction,
            )
            logger.info(f"Monte Carlo analysis results written to {mc_file}")
            mc_plot_file = output_dir.joinpath(f"monte_carlo_uncertainty_{tel_model.name}.pdf")
            plot_psf.create_monte_carlo_uncertainty_plot(
                mc_results, mc_plot_file, optimizer.fraction, optimizer.use_ks_statistic
            )
        return

    # Run gradient descent optimization
    threshold = args_dict.get("rmsd_threshold")
    learning_rate = args_dict.get("learning_rate")

    best_pars, best_psf_diameter, gd_results = optimizer.run_gradient_descent(
        threshold, learning_rate
    )

    # Check if optimization was successful
    if not gd_results or best_pars is None:
        logger.error("Gradient descent optimization failed to produce results.")
        return

    plot_psf.create_optimization_plots(args_dict, gd_results, tel_model, data_to_plot, output_dir)

    convergence_plot_file = output_dir.joinpath(
        f"gradient_descent_convergence_{tel_model.name}.png"
    )
    plot_psf.create_gradient_descent_convergence_plot(
        gd_results,
        threshold,
        convergence_plot_file,
        optimizer.fraction,
        optimizer.use_ks_statistic,
    )

    param_file = write_gradient_descent_log(
        gd_results,
        best_pars,
        best_psf_diameter,
        output_dir,
        tel_model,
        optimizer.use_ks_statistic,
        optimizer.fraction,
    )
    logger.info(f"\nGradient descent progression written to {param_file}")

    plot_psf.create_psf_vs_offaxis_plot(tel_model, site_model, args_dict, best_pars, output_dir)

    if args_dict.get("write_psf_parameters", False):
        logger.info("Exporting best parameters as model files...")
        export_psf_parameters(
            best_pars, args_dict.get("telescope"), args_dict.get("parameter_version"), output_dir
        )

    if args_dict.get("cleanup", False):
        cleanup_intermediate_files(output_dir)
