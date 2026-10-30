"""
PSF parameter optimisation for the mirror alignment parameters.

Gradient descent minimising the RMSD (or KS statistic) of cumulative PSF curves.
Logs and plots are written by ``simtools.ray_tracing.psf_parameter_reports``.
"""

import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.table import Table
from scipy import stats

from simtools.context import RunContext
from simtools.data_model import model_data_writer as writer
from simtools.data_model import parameter_catalog as catalog
from simtools.io.output_paths import OutputPaths
from simtools.job_execution.backends.base import BackendError
from simtools.job_execution.backends.local import determine_max_workers
from simtools.model.model_utils import initialize_simulation_models
from simtools.ray_tracing.psf_evaluation import dispatch_evaluations
from simtools.ray_tracing.psf_parameter_reports import (
    create_optimization_plots,
    create_psf_vs_offaxis_plot,
    setup_pdf_plotting,
    write_gradient_descent_log,
    write_monte_carlo_analysis,
)
from simtools.ray_tracing.ray_tracing import RayTracing
from simtools.utils import general as gen
from simtools.visualization import plot_psf
from simtools.visualization.plot_psf import DEFAULT_FRACTION

logger = logging.getLogger(__name__)

RADIUS = "Radius"
CUMULATIVE_PSF = "Cumulative PSF"
LABEL = "label"
TELESCOPE = "telescope"
KS_STATISTIC = "ks_statistic"
MIRROR_ALIGNMENT = ("mirror_align_random_horizontal", "mirror_align_random_vertical")


@dataclass
class GradientStepResult:
    """Outcome of one gradient descent step; ``params`` is None when the step failed."""

    params: dict | None
    psf_diameter: float | None
    metric: float | None
    p_value: float | None
    simulated_data: np.ndarray | None
    step_accepted: bool
    learning_rate: float


def _failed_step(learning_rate):
    """Return the result of a step that was not accepted."""
    return GradientStepResult(None, None, None, None, None, False, learning_rate)


class PSFParameterOptimizer:
    """
    Gradient descent optimizer for the PSF mirror alignment parameters.

    Parameters
    ----------
    tel_model, site_model : TelescopeModel, SiteModel
        Telescope and site models.
    args_dict : dict
        Simulation configuration arguments.
    data_to_plot : dict
        Measured PSF data under the "measured" key.
    radius : numpy.ndarray
        Radius values in cm for PSF evaluation.
    output_dir : Path
        Directory for optimisation results and plots.
    context : RunContext, optional
        Run context.
    """

    LR_REDUCTION_FACTOR = 0.7
    LR_INCREASE_FACTOR = 2.0
    LR_MINIMUM_THRESHOLD = 1e-6
    LR_MAXIMUM_THRESHOLD = 0.01
    LR_RESET_VALUE = 0.0001

    def __init__(
        self, tel_model, site_model, args_dict, data_to_plot, radius, output_dir, context=None
    ):
        """Initialize the PSF parameter optimizer."""
        self.tel_model = tel_model
        self.site_model = site_model
        self.args_dict = args_dict
        self.data_to_plot = data_to_plot
        self.radius = radius
        self.output_dir = output_dir
        self.context = (context or RunContext.from_environment()).replace(args=args_dict)
        self.use_ks_statistic = args_dict.get(KS_STATISTIC, False)
        self.fraction = args_dict.get("fraction", DEFAULT_FRACTION)
        self.simulation_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.max_workers = determine_max_workers(args_dict.get("max_workers"), default_fraction=0.5)

    def _params_to_cache_key(self, params):
        """Return a hashable cache key for a parameter set."""
        return frozenset(
            (key, tuple(value) if isinstance(value, list) else value)
            for key, value in sorted(params.items())
        )

    def _reduce_learning_rate(self, current_lr):
        """Reduce the learning rate, resetting it to LR_RESET_VALUE below the threshold."""
        new_lr = current_lr * self.LR_REDUCTION_FACTOR
        return self.LR_RESET_VALUE if new_lr < self.LR_MINIMUM_THRESHOLD else new_lr

    def _increase_learning_rate(self, current_lr):
        """Increase the learning rate, capped at LR_MAXIMUM_THRESHOLD."""
        return min(current_lr * self.LR_INCREASE_FACTOR, self.LR_MAXIMUM_THRESHOLD)

    def get_initial_parameters(self):
        """Return the current PSF parameter values of the telescope model."""
        return get_previous_values(self.tel_model)

    def run_simulation(
        self, pars, pdf_pages=None, is_best=False, use_cache=True, use_ks_statistic=None
    ):
        """Run a PSF simulation (cached unless plotting); returns (diameter, metric, p, data)."""
        ks_stat = self.use_ks_statistic if use_ks_statistic is None else use_ks_statistic
        cacheable = use_cache and pdf_pages is None and not is_best
        if cacheable:
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
            context=self.context,
        )
        if cacheable:
            self.simulation_cache[self._params_to_cache_key(pars)] = result
        return result

    def calculate_gradient(self, current_params, current_metric, epsilon=0.0005):
        """Return the finite-difference gradient per parameter; None if an evaluation failed."""
        perturbations = [
            _create_perturbed_params(current_params, name, values, index, value, epsilon)
            for name, values in current_params.items()
            for index, value in enumerate(values if isinstance(values, list) else [values])
        ]
        results = self._run_parameter_batch(perturbations, use_cache=True)
        if any(result is None for result in results):
            return None
        metrics = iter((result[1] - current_metric) / epsilon for result in results)
        return {
            name: [next(metrics) for _ in values] if isinstance(values, list) else next(metrics)
            for name, values in current_params.items()
        }

    def _lookup_parameter_batch(self, parameters, use_cache):
        """Return (keys, pending, results): cache keys, unevaluated sets and cached results."""
        keys, pending, results = [], {}, {}
        for index, params in enumerate(parameters):
            key = self._params_to_cache_key(params) if use_cache else index
            keys.append(key)
            if use_cache and key in self.simulation_cache:
                self.cache_hits += 1
                results[key] = self.simulation_cache[key]
            elif key not in pending:
                pending[key] = params
                if use_cache:
                    self.cache_misses += 1
        return keys, pending, results

    def _run_parameter_batch(self, parameters, use_cache):
        """Evaluate independent parameter sets, preserving order and cached results."""
        keys, pending, results = self._lookup_parameter_batch(parameters, use_cache)
        if pending:
            evaluated = self._dispatch_parameter_evaluations(list(pending.values()))
            for key, result in zip(pending, evaluated, strict=True):
                results[key] = result
                if use_cache and result is not None:
                    self.simulation_cache[key] = result
        return [results[key] for key in keys]

    def _dispatch_parameter_evaluations(self, parameters):
        """Submit isolated evaluations to the configured execution backend."""
        return dispatch_evaluations(
            parameters,
            self.args_dict,
            self.data_to_plot,
            self.radius,
            self.output_dir,
            self.max_workers,
            _run_psf_evaluation,
            context=self.context,
        )

    def apply_gradient_step(self, current_params, gradients, learning_rate):
        """Apply a gradient step; the zenith angle (index 1) of mirror alignment is kept."""
        new_params = {}
        for name, values in current_params.items():
            if isinstance(values, list):
                new_params[name] = [
                    value
                    if name in MIRROR_ALIGNMENT and index == 1
                    else value - learning_rate * gradient
                    for index, (value, gradient) in enumerate(zip(values, gradients[name]))
                ]
            else:
                new_params[name] = values - learning_rate * gradients[name]
        return new_params

    def perform_gradient_step_with_retries(
        self, current_params, current_metric, learning_rate, max_retries=3
    ):
        """
        Attempt a gradient descent step with adaptive learning rate reduction.

        Returns
        -------
        GradientStepResult
            Updated parameters and step status.
        """
        current_lr = learning_rate
        for attempt in range(max_retries):
            try:
                gradients = self.calculate_gradient(current_params, current_metric)
                if gradients is None:
                    logger.warning(
                        f"Gradient calculation failed on attempt {attempt + 1}, skipping step"
                    )
                    return _failed_step(current_lr)

                new_params = self.apply_gradient_step(current_params, gradients, current_lr)
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
                    return GradientStepResult(
                        params=new_params,
                        psf_diameter=new_psf_diameter,
                        metric=new_metric,
                        p_value=new_p_value,
                        simulated_data=new_simulated_data,
                        step_accepted=True,
                        learning_rate=current_lr,
                    )

                logger.info(
                    f"Step rejected (RMSD {current_metric:.6f} -> {new_metric:.6f}), "
                    f"reducing learning rate to {current_lr * self.LR_REDUCTION_FACTOR:.6f}"
                )
                current_lr = self._reduce_learning_rate(current_lr)

            except BackendError:
                raise
            except (ValueError, RuntimeError, KeyError) as e:
                logger.warning(f"Simulation failed on attempt {attempt + 1}: {e}")

        return _failed_step(current_lr)

    def _create_step_plot(
        self, pdf_pages, current_params, new_psf_diameter, new_metric, new_p_value, new_simulated
    ):
        """Plot an accepted gradient step if ``plot_all`` is set."""
        if pdf_pages is None or not self.args_dict.get("plot_all", False) or new_simulated is None:
            return

        self.data_to_plot["simulated"] = new_simulated
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
        """Plot the best parameters with both the RMSD and the KS statistic."""
        if pdf_pages is None or best_params is None:
            return

        logger.info("Creating final plot for best parameters with both RMSD and KS statistic...")
        _, best_ks_stat, best_p_value, best_simulated_data = self.run_simulation(
            best_params, pdf_pages=None, is_best=False, use_cache=False, use_ks_statistic=True
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
        logger.info("Cumulative PSF plots saved")

    def run_gradient_descent(self, rmsd_threshold, learning_rate, max_iterations=200):
        """Run gradient descent; returns (best_params, best_psf_diameter, results)."""
        if self.data_to_plot is None or self.radius is None:
            logger.error("No PSF measurement data provided. Cannot run optimization.")
            return None, None, []

        with setup_pdf_plotting(self.args_dict, self.output_dir, self.tel_model.name) as report:
            return self._run_descent(rmsd_threshold, learning_rate, max_iterations, report)

    def _run_descent(self, rmsd_threshold, learning_rate, max_iterations, pdf_pages):
        """Run gradient descent while writing plots to an optional open report."""
        current_params = self.get_initial_parameters()
        current_psf_diameter, current_metric, current_p_value, simulated_data = self.run_simulation(
            current_params,
            pdf_pages=pdf_pages if self.args_dict.get("plot_all", False) else None,
            is_best=False,
        )
        results = [
            (
                current_params.copy(),
                current_metric,
                current_p_value,
                current_psf_diameter,
                simulated_data,
            )
        ]
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
            tolerance = 0.1 * rmsd_threshold
            if current_metric <= (rmsd_threshold + tolerance):
                logger.info(
                    f"Optimization converged: RMSD {current_metric:.6f} <= "
                    f"threshold {rmsd_threshold:.6f} (tolerance {tolerance:.6f})"
                )
                break

            iteration += 1
            logger.info(f"Gradient descent iteration {iteration}")
            step_result = self.perform_gradient_step_with_retries(
                current_params, current_metric, current_lr
            )
            if not step_result.step_accepted or step_result.params is None:
                current_lr = self._increase_learning_rate(current_lr)
                logger.info(f"No step accepted, increasing learning rate to {current_lr:.6f}")
                continue

            current_params = step_result.params
            current_metric = step_result.metric
            current_psf_diameter = step_result.psf_diameter
            current_lr = step_result.learning_rate
            results.append(
                (
                    current_params.copy(),
                    current_metric,
                    None,
                    current_psf_diameter,
                    step_result.simulated_data,
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
                step_result.psf_diameter,
                step_result.metric,
                step_result.p_value,
                step_result.simulated_data,
            )
            logger.info(f"  Accepted step: improved to {step_result.metric:.6f}")

        self._create_final_plot(pdf_pages, best_params, best_psf_diameter)
        return best_params, best_psf_diameter, results

    def analyze_monte_carlo_error(self, n_simulations=500):
        """Run Monte Carlo simulations of the initial parameters and summarise the spread."""
        empty = (None, None, [], None, None, [], None, None, [])
        if self.data_to_plot is None or self.radius is None:
            logger.error("No PSF measurement data provided. Cannot analyze Monte Carlo error.")
            return empty

        initial_params = self.get_initial_parameters()
        for param_name, param_values in initial_params.items():
            logger.info(f"  {param_name}: {param_values}")

        results = [
            result
            for result in self._run_parameter_batch(
                [initial_params] * n_simulations, use_cache=False
            )
            if result is not None
        ]
        if not results:
            logger.error("All Monte Carlo simulations failed.")
            return empty

        psf_diameter_values = [result[0] for result in results]
        metric_values = [result[1] for result in results]
        p_values = [result[2] for result in results]
        mean_metric, std_metric = np.mean(metric_values), _sample_standard_deviation(metric_values)
        mean_psf_diameter, std_psf_diameter = (
            np.mean(psf_diameter_values),
            _sample_standard_deviation(psf_diameter_values),
        )

        mean_p_value = std_p_value = None
        if self.use_ks_statistic:
            valid_p_values = [p for p in p_values if p is not None]
            if valid_p_values:
                mean_p_value = np.mean(valid_p_values)
                std_p_value = _sample_standard_deviation(valid_p_values)

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
    """Check a value against the schema range; True when no range is defined."""
    try:
        param_schema = catalog.model_parameters().get(param_name)
        data = param_schema.get("data") if param_schema is not None else None
        if not isinstance(data, list) or len(data) <= param_index:
            return True
        allowed_range = data[param_index].get("allowed_range")
        if not allowed_range:
            return True
        min_val, max_val = allowed_range.get("min"), allowed_range.get("max")
        if min_val is not None and value < min_val:
            return False
        return not (max_val is not None and value > max_val)

    except (KeyError, IndexError) as e:
        logger.warning(f"Error reading schema for {param_name}[{param_index}]: {e}")
        return True


def _are_all_parameters_within_allowed_range(params):
    """Check that every parameter component is within its allowed range."""
    for name, values in params.items():
        for i, v in enumerate(values if isinstance(values, list) else [values]):
            if not _is_parameter_within_allowed_range(name, i, v):
                logger.debug(f"{name}[{i}]={v:.6f} out of range")
                return False
    return True


def _create_perturbed_params(current_params, param_name, param_values, param_index, value, epsilon):
    """Copy the parameter vectors and perturb one component without changing the input."""
    perturbed_params = {
        k: v.copy() if isinstance(v, list) else v for k, v in current_params.items()
    }
    if isinstance(param_values, list):
        perturbed_params[param_name][param_index] = value + epsilon
    else:
        perturbed_params[param_name] = value + epsilon
    return perturbed_params


def calculate_rmsd(data, sim):
    """Return the RMSD between measured and simulated cumulative PSF curves."""
    return np.sqrt(np.mean((data - sim) ** 2))


def calculate_ks_statistic(data, sim):
    """Return the two-sample KS test between measured and simulated cumulative PSF curves."""
    return stats.ks_2samp(data, sim)


def get_previous_values(tel_model):
    """Return the current horizontal and vertical mirror alignment parameter values."""
    return {name: tel_model.get_parameter_value(name) for name in MIRROR_ALIGNMENT}


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
    context=None,
):
    """
    Run one ray tracing PSF simulation and calculate the optimisation metric.

    Parameters
    ----------
    tel_model, site : TelescopeModel, Site
        Telescope and site models.
    args_dict : dict
        Simulation configuration arguments.
    pars : dict
        Parameter values to test.
    data_to_plot : dict
        Measured PSF data under the "measured" key.
    radius : numpy.ndarray
        Radius values in cm for PSF evaluation.
    pdf_pages : PDF report, optional
        Open report for adding plots.
    is_best : bool, optional
        True if this is the best parameter set.
    use_ks_statistic : bool, optional
        Use the KS statistic as metric instead of RMSD.
    context : RunContext, optional
        Run context for ray tracing.

    Returns
    -------
    tuple of (float, float, float or None, array)
        PSF diameter [cm], metric, p-value (None for RMSD), simulated PSF data.
    """
    if pars is None:
        raise ValueError("No best parameters found")

    tel_model.overwrite_parameters(pars, flat_dict=True)
    fraction = args_dict.get("fraction", DEFAULT_FRACTION)
    ray = RayTracing(
        telescope_model=tel_model,
        site_model=site,
        label=args_dict.get(LABEL) or getattr(tel_model, LABEL, None),
        zenith_angle=args_dict["zenith_angle"],
        source_distance=args_dict["source_distance"],
        off_axis_angle=[0.0] * u.deg,
        context=context,
    )
    ray.simulate(test=args_dict.get("test", False), force=True)
    ray.analyze(force=True, use_rx=False)
    im = ray.images()[0]
    psf_diameter = im.get_psf(fraction=fraction)

    if radius is None:
        raise ValueError("Radius data is not available.")
    simulated_data = im.get_cumulative_data(radius * u.cm)

    measured = data_to_plot["measured"][CUMULATIVE_PSF]
    if use_ks_statistic:
        metric, p_value = calculate_ks_statistic(measured, simulated_data[CUMULATIVE_PSF])
    else:
        metric = calculate_rmsd(measured, simulated_data[CUMULATIVE_PSF])
        p_value = None

    if pdf_pages is not None and args_dict.get("plot_all", False):
        data_to_plot["simulated"] = simulated_data
        plot_psf.create_psf_parameter_plot(
            data_to_plot,
            pars,
            psf_diameter,
            metric,
            is_best,
            pdf_pages,
            fraction=fraction,
            p_value=p_value,
            use_ks_statistic=use_ks_statistic,
        )
        del data_to_plot["simulated"]

    return psf_diameter, metric, p_value, simulated_data


def load_and_process_data(args_dict):
    """
    Load the measured PSF data from an ECSV file.

    Returns
    -------
    tuple of (OrderedDict, array)
        Normalised cumulative PSF under "measured", and the radius values in cm.
    """
    if args_dict["data"] is None:
        raise FileNotFoundError("No data file specified for PSF optimization.")

    table = Table.read(
        gen.find_file(args_dict["data"], args_dict["data_search_path"]), format="ascii.ecsv"
    )
    radius_column = next((col for col in table.colnames if "radius" in col.lower()), None)
    integral_psf_column = next((col for col in table.colnames if "integral" in col.lower()), None)

    data = np.zeros(len(table), dtype={"names": (RADIUS, CUMULATIVE_PSF), "formats": ("f8", "f8")})
    data[RADIUS] = table[radius_column].to(u.cm).value
    data[CUMULATIVE_PSF] = table[integral_psf_column]
    data[CUMULATIVE_PSF] /= np.max(np.abs(data[CUMULATIVE_PSF]))  # Normalize to max = 1.0

    return OrderedDict([("measured", data)]), data[RADIUS]


def _add_units_to_psf_parameters(best_pars):
    """Add astropy units to the PSF parameters."""
    units = (u.deg, u.deg, u.dimensionless_unscaled, u.dimensionless_unscaled)
    return {
        name: [values[i] * units[i] for i in range(4)] if name in MIRROR_ALIGNMENT else values
        for name, values in best_pars.items()
    }


def export_psf_parameters(best_pars, telescope, parameter_version, output_dir, context=None):
    """
    Export the optimised PSF parameters as simulation model parameter files.

    One JSON file per parameter: {output_dir}/{telescope}/{name}-{version}.json.
    Errors are logged, not raised.
    """
    try:
        parameter_output_path = output_dir.joinpath(telescope)
        for parameter_name, parameter_value in _add_units_to_psf_parameters(best_pars).items():
            writer.ModelDataWriter.write_model_parameter(
                parameter_name=parameter_name,
                value=parameter_value,
                instrument=telescope,
                parameter_version=parameter_version,
                output_file=f"{parameter_name}-{parameter_version}.json",
                output_path=parameter_output_path,
                context=context,
            )
        logger.info(f"simulation model parameter files exported to {output_dir}")

    except (ValueError, KeyError, OSError) as e:
        logger.error(f"Error exporting simulation parameters: {e}")


def _sample_standard_deviation(values):
    """Return the sample standard deviation, NaN when fewer than two samples exist."""
    return float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")


def _initialize_job_models(args, **kwargs):
    """Initialize the telescope and site models named in the PSF job arguments."""
    return initialize_simulation_models(
        label=args.get(LABEL),
        site=args["site"],
        telescope_name=args[TELESCOPE],
        model_version=args["model_version"],
        **kwargs,
    )


def _run_psf_evaluation(job, context=None):
    """Evaluate one parameter set in its own output directory; None if it fails."""
    context = context or RunContext.from_environment()
    paths = OutputPaths(job.output_dir, model_path=context.paths.model_path)
    context = context.replace(paths=paths)
    try:
        telescope, site, _ = _initialize_job_models(
            job.args, model_directory=job.output_dir / "model", context=context
        )
        return run_psf_simulation(
            telescope,
            site,
            job.args,
            job.parameters,
            job.measured_data,
            job.radius,
            use_ks_statistic=job.args.get(KS_STATISTIC, False),
            context=context,
        )
    except (ValueError, RuntimeError) as exc:
        logger.warning("PSF evaluation %s failed: %s", job.output_dir.name, exc)
        return None


@dataclass(frozen=True)
class PSFOptimizationJob:
    """Serializable arguments and output directory for one complete PSF workflow."""

    args: dict
    output_dir: Path
    context: RunContext | None = None


def run_psf_optimization_job(job):
    """Initialize the controller models and run the PSF optimisation workflow."""
    tel_model, site_model, _ = _initialize_job_models(job.args, context=job.context)
    run_psf_optimization_workflow(tel_model, site_model, job.args, job.output_dir, job.context)


def run_psf_optimization_workflow(tel_model, site_model, args_dict, output_dir, context=None):
    """Run the PSF parameter optimisation workflow using gradient descent."""
    data_to_plot, radius = load_and_process_data(args_dict)
    optimizer = PSFParameterOptimizer(
        tel_model, site_model, args_dict, data_to_plot, radius, output_dir, context=context
    )

    if args_dict.get("monte_carlo_analysis", False):
        mc_results = optimizer.analyze_monte_carlo_error()
        if mc_results[0] is not None:
            mc_file = write_monte_carlo_analysis(
                mc_results, output_dir, tel_model, optimizer.use_ks_statistic, optimizer.fraction
            )
            logger.info(f"Monte Carlo analysis results written to {mc_file}")
            plot_psf.create_monte_carlo_uncertainty_plot(
                mc_results,
                output_dir.joinpath(f"monte_carlo_uncertainty_{tel_model.name}.pdf"),
                optimizer.fraction,
                optimizer.use_ks_statistic,
            )
        return

    threshold = args_dict.get("rmsd_threshold")
    best_pars, best_psf_diameter, gd_results = optimizer.run_gradient_descent(
        threshold,
        args_dict.get("learning_rate"),
        max_iterations=args_dict.get("max_iterations", 200),
    )
    if not gd_results or best_pars is None:
        logger.error("Gradient descent optimization failed to produce results.")
        return

    create_optimization_plots(args_dict, gd_results, tel_model, data_to_plot, output_dir)
    plot_psf.create_gradient_descent_convergence_plot(
        gd_results,
        threshold,
        output_dir.joinpath(f"gradient_descent_convergence_{tel_model.name}.png"),
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

    create_psf_vs_offaxis_plot(
        tel_model, site_model, args_dict, best_pars, output_dir, context=context
    )

    if args_dict.get("write_psf_parameters", False):
        logger.info("Exporting best parameters as model files...")
        export_psf_parameters(
            best_pars,
            args_dict.get(TELESCOPE),
            args_dict.get("parameter_version"),
            output_dir,
            context=context,
        )

    if args_dict.get("cleanup", False):
        gen.cleanup_intermediate_files(output_dir, patterns=("*.log", "*.lis*", "*.dat"))
