"""Mirror panel PSF calculation with per-mirror d80 optimization."""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.table import Table

import simtools.utils.general as gen
from simtools.data_model import model_data_writer
from simtools.job_execution.process_pool import process_pool_map_ordered
from simtools.model.model_utils import initialize_simulation_models
from simtools.ray_tracing.ray_tracing import RayTracing
from simtools.utils import names
from simtools.visualization import plot_psf


@dataclass
class GradientStepResult:
    """Result of evaluating a single RNDA candidate."""

    rnda: list
    simulated_d80_mm: float
    signed_pct_diff: float
    objective: float


@dataclass(frozen=True)
class Bounds:
    """Parameter bounds."""

    min: float
    max: float


@dataclass(frozen=True)
class RndaGradientDescentSettings:
    """Settings for RNDA gradient descent optimization."""

    threshold: float
    learning_rate: float
    grad_clip: float

    sigma1: Bounds
    sigma2: Bounds
    frac2: Bounds

    max_log_step: float
    max_frac_step: float
    max_iterations: int


def _worker_optimize_mirror_forked(args):
    """Optimize a single mirror in a forked worker process.

    Parameters
    ----------
    args : tuple
        (mirror_idx, instance)
    """
    mirror_idx, instance = args
    measured_d80_mm = float(instance.measured_data[mirror_idx])
    return instance.optimize_single_mirror(mirror_idx, measured_d80_mm)


class MirrorPanelPSF:
    """
    Mirror panel PSF and random reflection angle calculation.

    This class derives the random reflection angle (RNDA) for mirror panels
    by optimizing per-mirror d80 values using percentage difference as the metric.

    Parameters
    ----------
    label : str
        Application label.
    args_dict : dict
        Dictionary with input arguments.
    """

    # Internal guard-rail defaults for the RNDA optimizer.
    DEFAULT_RNDA_GRAD_CLIP: float = 1e4
    DEFAULT_RNDA_MAX_LOG_STEP: float = 0.25
    DEFAULT_RNDA_MAX_FRAC_STEP: float = 0.1
    DEFAULT_RNDA_MAX_ITERATIONS: int = 100

    def __init__(self, label, args_dict):
        """Initialize the mirror-panel PSF optimizer.

        Parameters
        ----------
        label : str
            Application label.
        args_dict : dict
            Dictionary with input arguments.
        """
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Initializing MirrorPanelPSF")

        self.label = label
        self.args_dict = args_dict
        self.telescope_model, self.site_model, _ = initialize_simulation_models(
            label=label,
            site=self.args_dict["site"],
            telescope_name=self.args_dict["telescope"],
            model_version=self.args_dict["model_version"],
        )

        self.measured_data = self._load_measured_data()

        # Limit mirrors in test mode
        if self.args_dict.get("test", False):
            self.args_dict["number_of_mirrors_to_test"] = 10

        self.rnda_start = self.telescope_model.get_parameter_value("mirror_reflection_random_angle")
        self.rnda_opt = None
        self.per_mirror_results = []
        self.final_percentage_diff = None

    def _load_measured_data(self):
        """
        Load measured d80 from ECSV file.

        Returns
        -------
        Table
            Astropy table with d80 (mm) columns.
        """
        data_file = gen.find_file(self.args_dict["data"], self.args_dict.get("model_path", "."))
        table = Table.read(data_file)
        if "psf_opt" in table.colnames:
            return table["psf_opt"]

        if "d80" in table.colnames:
            return table["d80"]

        raise ValueError("Data file must contain either 'psf_opt' or 'd80' column")

    def _simulate_single_mirror_d80(self, mirror_idx, rnda_values):
        """
        Simulate a single mirror and return its d80.

        Parameters
        ----------
        mirror_idx : int
            Mirror index (0-based).
        rnda_values : list
            Random reflection angle values [sigma1, fraction2, sigma2].

        Returns
        -------
        float
            Simulated d80 in mm.
        """
        # Set the RNDA parameter
        self.telescope_model.overwrite_model_parameter(
            "mirror_reflection_random_angle", rnda_values
        )

        mirror_sim_index = int(mirror_idx)
        mirror_number = mirror_sim_index + 1  # user-facing label

        # Create ray tracing for single mirror.
        # Use a per-mirror label so parallel runs do not collide on output filenames.
        # IMPORTANT: RayTracing and SimulatorRayTracing derive filenames from
        # telescope_model.label. Passing RayTracing(label=...) alone breaks this link
        # and causes FileNotFoundError when RayTracing looks for the photons file.
        rt_label = f"{self.label}_m{mirror_number}"
        old_label = getattr(self.telescope_model, "label", None)
        old_config_file_path = getattr(self.telescope_model, "_config_file_path", None)
        old_config_file_directory = getattr(self.telescope_model, "_config_file_directory", None)
        try:
            self.telescope_model.label = rt_label
            # Force regeneration of config file path/name based on the new label.
            # These attributes are cached inside ModelParameter.
            if hasattr(self.telescope_model, "_config_file_path"):
                setattr(self.telescope_model, "_config_file_path", None)
            if hasattr(self.telescope_model, "_config_file_directory"):
                setattr(self.telescope_model, "_config_file_directory", None)
            ray = RayTracing(
                telescope_model=self.telescope_model,
                site_model=self.site_model,
                single_mirror_mode=True,
                mirror_numbers=[mirror_sim_index],
            )

            # Simulate and analyze
            ray.simulate(test=self.args_dict.get("test", False), force=True)
            ray.analyze(force=True)
        finally:
            if old_label is not None:
                self.telescope_model.label = old_label
            if hasattr(self.telescope_model, "_config_file_path"):
                setattr(self.telescope_model, "_config_file_path", old_config_file_path)
            if hasattr(self.telescope_model, "_config_file_directory"):
                setattr(self.telescope_model, "_config_file_directory", old_config_file_directory)

        return float(ray.get_d80_mm())

    def _calculate_percentage_difference(self, measured_d80, simulated_d80):
        """Calculate signed percentage difference."""
        measured = float(measured_d80)
        if measured <= 0:
            raise ValueError("Measured d80 must be positive")
        return 100.0 * (float(simulated_d80) - measured) / measured

    def _evaluate_rnda_candidate(
        self,
        mirror_idx,
        measured_d80_mm,
        rnda_values,
    ) -> GradientStepResult:
        """Evaluate a candidate RNDA by simulating and computing objective."""
        simulated_d80_mm = float(self._simulate_single_mirror_d80(mirror_idx, rnda_values))
        signed_pct = float(self._calculate_percentage_difference(measured_d80_mm, simulated_d80_mm))
        obj = float(signed_pct * signed_pct)
        return GradientStepResult(
            rnda=list(rnda_values),
            simulated_d80_mm=simulated_d80_mm,
            signed_pct_diff=signed_pct,
            objective=obj,
        )

    def _finite_difference_objective_gradient(
        self,
        mirror_idx,
        measured_d80_mm,
        current_rnda,
        param_index,
        plus_value,
        minus_value,
    ):
        """Compute the derivative ``d(objective)/d(param)`` via symmetric finite differences.

        Parameters
        ----------
        mirror_idx : int
            Mirror index (0-based).
        measured_d80_mm : float
            Measured d80 in mm.
        current_rnda : list
            Current RNDA values.
        param_index : int
            Index of the parameter to perturb.
        plus_value : float
            Perturbed value for the plus step.
        minus_value : float
            Perturbed value for the minus step.

        Returns
        -------
        float
            Estimated gradient component.
        """
        denom = float(plus_value - minus_value)
        if denom <= 0:
            return 0.0
        rnda_plus = list(current_rnda)
        rnda_minus = list(current_rnda)
        rnda_plus[param_index] = plus_value
        rnda_minus[param_index] = minus_value
        plus_eval = self._evaluate_rnda_candidate(
            mirror_idx=mirror_idx,
            measured_d80_mm=measured_d80_mm,
            rnda_values=rnda_plus,
        )
        minus_eval = self._evaluate_rnda_candidate(
            mirror_idx=mirror_idx,
            measured_d80_mm=measured_d80_mm,
            rnda_values=rnda_minus,
        )
        return float((plus_eval.objective - minus_eval.objective) / denom)

    def _get_allowed_range_from_schema(self, param_name: str, index: int):
        """Return allowed range for one entry of a model parameter.

        Parameters
        ----------
        param_name : str
            Model parameter name.
        index : int
            Index into the parameter's schema ``data`` list.

        Returns
        -------
        tuple
            Tuple ``(min_value, max_value)``. Returns ``(None, None)`` if the
            schema does not define a range.
        """
        schema = names.model_parameters().get(param_name) or {}
        data = schema.get("data")
        if not isinstance(data, list) or index < 0 or index >= len(data):
            return None, None
        allowed_range = (data[index] or {}).get("allowed_range")
        if not isinstance(allowed_range, dict):
            return None, None
        return allowed_range.get("min"), allowed_range.get("max")

    def _get_rnda_parameter_bounds(self, param_name: str):
        """Get bounds for RNDA parameters.

        Parameters
        ----------
        param_name : str
            Model parameter name for RNDA (typically
            ``"mirror_reflection_random_angle"``).

        Returns
        -------
        tuple
            Tuple ``(sigma1_bounds, frac2_bounds, sigma2_bounds)`` of
            :class:`~simtools.ray_tracing.mirror_panel_psf.Bounds`.
        """
        sigma1_min, sigma1_max = self._get_allowed_range_from_schema(param_name, 0)
        frac2_min, frac2_max = self._get_allowed_range_from_schema(param_name, 1)
        sigma2_min, sigma2_max = self._get_allowed_range_from_schema(param_name, 2)

        sigma1_min = max(sigma1_min, 1e-12)
        sigma2_min = max(sigma2_min, 1e-12)

        return (
            Bounds(min=sigma1_min, max=sigma1_max),
            Bounds(min=frac2_min, max=frac2_max),
            Bounds(min=sigma2_min, max=sigma2_max),
        )

    def _build_rnda_gradient_descent_settings(
        self,
        threshold,
        learning_rate,
    ) -> RndaGradientDescentSettings:
        """Build gradient descent settings from defaults and schema-defined bounds."""
        grad_clip = float(self.DEFAULT_RNDA_GRAD_CLIP)
        max_log_step = float(self.DEFAULT_RNDA_MAX_LOG_STEP)
        max_frac_step = float(self.DEFAULT_RNDA_MAX_FRAC_STEP)
        max_iterations = int(self.DEFAULT_RNDA_MAX_ITERATIONS)

        sigma1_bounds, frac2_bounds, sigma2_bounds = self._get_rnda_parameter_bounds(
            "mirror_reflection_random_angle"
        )

        return RndaGradientDescentSettings(
            threshold=float(threshold),
            learning_rate=float(learning_rate),
            grad_clip=grad_clip,
            sigma1=sigma1_bounds,
            sigma2=sigma2_bounds,
            frac2=frac2_bounds,
            max_log_step=max_log_step,
            max_frac_step=max_frac_step,
            max_iterations=max_iterations,
        )

    def _propose_next_rnda(
        self,
        mirror_idx,
        measured_d80_mm,
        current_rnda,
        settings,
        learning_rate,
    ):
        """Propose a next RNDA point using one gradient step.

        Parameters
        ----------
        mirror_idx : int
            Mirror index (0-based).
        measured_d80_mm : float
            Measured d80 in mm.
        current_rnda : list
            Current RNDA values.
        settings : RndaGradientDescentSettings
            Optimizer settings (includes bounds).
        learning_rate : float
            Current learning rate.

        Returns
        -------
        list
            Proposed RNDA values ``[sigma1, fraction2, sigma2]``.
        """
        sigma1, frac2, sigma2 = map(float, current_rnda)
        specs = (
            {
                "idx": 0,
                "kind": "log",
                "value": sigma1,
                "min": float(settings.sigma1.min),
                "max": float(settings.sigma1.max),
                "eps": max(1e-6, 0.05 * sigma1),
            },
            {
                "idx": 1,
                "kind": "linear",
                "value": frac2,
                "min": float(settings.frac2.min),
                "max": float(settings.frac2.max),
                "eps": max(1e-6, 0.02),
            },
            {
                "idx": 2,
                "kind": "log",
                "value": sigma2,
                "min": float(settings.sigma2.min),
                "max": float(settings.sigma2.max),
                "eps": max(1e-6, 0.05 * sigma2),
            },
        )

        new_values = [sigma1, frac2, sigma2]
        for spec in specs:
            v = float(spec["value"])
            v_plus = min(float(spec["max"]), v + float(spec["eps"]))
            v_minus = max(float(spec["min"]), v - float(spec["eps"]))

            grad = self._finite_difference_objective_gradient(
                mirror_idx=mirror_idx,
                measured_d80_mm=measured_d80_mm,
                current_rnda=current_rnda,
                param_index=int(spec["idx"]),
                plus_value=float(v_plus),
                minus_value=float(v_minus),
            )

            if spec["kind"] == "log":
                grad_log = float(np.clip(grad * v, -settings.grad_clip, settings.grad_clip))
                log_step = float(
                    np.clip(
                        -float(learning_rate) * grad_log,
                        -settings.max_log_step,
                        settings.max_log_step,
                    )
                )
                new_v = float(
                    np.clip(
                        v * float(np.exp(log_step)),
                        float(spec["min"]),
                        float(spec["max"]),
                    )
                )
            else:
                grad = float(np.clip(grad, -settings.grad_clip, settings.grad_clip))
                step = float(
                    np.clip(
                        -float(learning_rate) * grad,
                        -settings.max_frac_step,
                        settings.max_frac_step,
                    )
                )
                new_v = float(np.clip(v + step, float(spec["min"]), float(spec["max"])))

            new_values[int(spec["idx"])] = new_v

        return list(map(float, new_values))

    def _run_rnda_gradient_descent(
        self,
        mirror_idx,
        measured_d80_mm,
        current_rnda,
        settings,
    ):
        """Run a 3-parameter RNDA gradient descent (sigma1, fraction2, sigma2).

        Parameters
        ----------
        mirror_idx : int
            Mirror index (0-based).
        measured_d80_mm : float
            Measured d80 in mm.
        current_rnda : list
            Starting RNDA values.
        settings : RndaGradientDescentSettings
            Optimizer settings.

        Returns
        -------
        tuple
            Tuple ``(best_rnda, best_sim_d80, best_pct_diff)``.
        """
        current_eval = self._evaluate_rnda_candidate(
            mirror_idx=mirror_idx,
            measured_d80_mm=measured_d80_mm,
            rnda_values=current_rnda,
        )

        best = {
            "rnda": list(current_eval.rnda),
            "sim_d80": float(current_eval.simulated_d80_mm),
            "obj": float(current_eval.objective),
            "pct": abs(float(current_eval.signed_pct_diff)),
        }

        self._logger.info(
            "  Initial: simulated d80 = %.3f mm, signed pct diff = %.2f%%",
            best["sim_d80"],
            current_eval.signed_pct_diff,
        )

        if best["pct"] <= settings.threshold * 100:
            self._logger.info("  Already converged!")
            return best["rnda"], best["sim_d80"], best["pct"]

        learning_rate = float(settings.learning_rate)
        for iteration in range(settings.max_iterations):
            old_rnda = list(map(float, current_rnda))
            new_rnda = self._propose_next_rnda(
                mirror_idx=mirror_idx,
                measured_d80_mm=measured_d80_mm,
                current_rnda=current_rnda,
                settings=settings,
                learning_rate=learning_rate,
            )
            new_eval = self._evaluate_rnda_candidate(
                mirror_idx=mirror_idx,
                measured_d80_mm=measured_d80_mm,
                rnda_values=new_rnda,
            )

            new_obj = float(new_eval.objective)
            new_pct_diff = abs(float(new_eval.signed_pct_diff))

            self._logger.info(
                (
                    "  Iter %d: s1=%.6f->%.6f f2=%.3f->%.3f s2=%.6f->%.6f, "
                    "|pct|=%.2f%%->%.2f%%, lr=%.3g"
                ),
                iteration + 1,
                old_rnda[0],
                new_rnda[0],
                old_rnda[1],
                new_rnda[1],
                old_rnda[2],
                new_rnda[2],
                best["pct"],
                new_pct_diff,
                learning_rate,
            )

            if new_obj < best["obj"]:
                best["obj"] = new_obj
                best["pct"] = new_pct_diff
                best["rnda"] = list(new_rnda)
                best["sim_d80"] = float(new_eval.simulated_d80_mm)
                current_rnda = list(new_rnda)
                learning_rate *= 1.1
            else:
                learning_rate *= 0.5

            if best["pct"] <= settings.threshold * 100:
                self._logger.info("  Converged at iteration %d!", iteration + 1)
                break

            if learning_rate < 1e-12:
                self._logger.info("  Learning rate too small, stopping.")
                break

        self._logger.info(
            "  Final: simulated d80 = %.3f mm, pct diff = %.2f%%",
            best["sim_d80"],
            best["pct"],
        )
        return best["rnda"], best["sim_d80"], best["pct"]

    def optimize_single_mirror(self, mirror_idx, measured_d80_mm):
        """
        Optimize RNDA for a single mirror using gradient descent.

        Parameters
        ----------
        mirror_idx : int
            Mirror index (0-based).
        measured_d80_mm : float
            Measured d80 in mm.

        Returns
        -------
        dict
            Optimization result with rnda, simulated_d80, percentage_diff.
        """
        threshold = float(self.args_dict.get("threshold", 0.05))  # 5% default
        learning_rate = float(self.args_dict.get("learning_rate", 0.001))

        settings = self._build_rnda_gradient_descent_settings(
            threshold=threshold,
            learning_rate=learning_rate,
        )
        current_rnda = list(self.rnda_start)

        self._logger.info(f"Mirror {mirror_idx + 1}: measured d80 = {measured_d80_mm:.3f} mm")

        best_rnda, best_sim_d80, best_pct_diff = self._run_rnda_gradient_descent(
            mirror_idx=mirror_idx,
            measured_d80_mm=measured_d80_mm,
            current_rnda=current_rnda,
            settings=settings,
        )
        return {
            "mirror": mirror_idx + 1,
            "measured_d80_mm": measured_d80_mm,
            "optimized_rnda": best_rnda,
            "simulated_d80_mm": best_sim_d80,
            "percentage_diff": best_pct_diff,
        }

    def _optimize_mirrors_parallel(self, n_mirrors, n_workers):
        """Optimize mirrors in parallel.

        Parameters
        ----------
        n_mirrors : int
            Number of mirrors to optimize.
        n_workers : int
            Number of worker processes.

        Returns
        -------
        list
            List of per-mirror results, ordered by mirror index.

        Notes
        -----
        Uses the ``fork`` start method so workers inherit the parent state.
        """
        # Parent instance created once
        parent_instance = MirrorPanelPSF(
            label=self.label, args_dict=dict(self.args_dict, parallel=False)
        )
        worker_inputs = [(i, parent_instance) for i in range(n_mirrors)]
        return process_pool_map_ordered(
            _worker_optimize_mirror_forked,
            worker_inputs,
            max_workers=n_workers,
            mp_start_method="fork",
        )

    def optimize_with_gradient_descent(self):
        """
        Optimize RNDA for each mirror individually using percentage difference metric.

        The final optimized RNDA is the average of per-mirror optimized values.
        """
        n_mirrors = len(self.measured_data)
        if self.args_dict.get("test", False):
            n_mirrors = min(n_mirrors, self.args_dict.get("number_of_mirrors_to_test", 10))

        n_workers = int(self.args_dict.get("n_workers", 0) or 0)
        if n_workers <= 0:
            n_workers = os.cpu_count()

        self._logger.info("Running per-mirror optimization with %d worker processes", n_workers)
        self.per_mirror_results = self._optimize_mirrors_parallel(n_mirrors, n_workers)

        # Average the optimized RNDA values
        avg_rnda = [0.0, 0.0, 0.0]
        for result in self.per_mirror_results:
            for j in range(3):
                avg_rnda[j] += result["optimized_rnda"][j]
        for j in range(3):
            avg_rnda[j] /= len(self.per_mirror_results)

        self.rnda_opt = avg_rnda

        # Calculate average percentage difference
        self.final_percentage_diff = np.mean(
            [r["percentage_diff"] for r in self.per_mirror_results]
        )

        self._logger.info(
            f"Optimization complete. Mean percentage difference: {self.final_percentage_diff:.2f}%"
        )

    def write_optimization_data(self):
        """Write optimization results.

        In addition to the per-mirror results, this also exports the averaged
        ``mirror_reflection_random_angle`` as a DB-style model parameter file
        (same format as other simulation model parameter exporters).
        """
        output_dir = Path(self.args_dict.get("output_path", "."))
        output_dir.mkdir(parents=True, exist_ok=True)

        telescope = self.args_dict.get("telescope")
        parameter_version = self.args_dict.get("parameter_version")
        parameter_name = "mirror_reflection_random_angle"
        parameter_output_path = output_dir.joinpath(str(telescope))
        parameter_output_path.mkdir(parents=True, exist_ok=True)
        output_file = parameter_output_path / "per_mirror_rnda.json"
        per_mirror_results_out = []
        for r in self.per_mirror_results:
            out = dict(r)
            for key in (
                "measured_d80_mm",
                "simulated_d80_mm",
                "percentage_diff",
                "simulated_d80_mm_plot",
                "percentage_diff_plot",
            ):
                if key in out:
                    out[key] = float(f"{float(out[key]):.4f}")
            if "optimized_rnda" in out:
                out["optimized_rnda"] = [float(f"{float(v):.4f}") for v in out["optimized_rnda"]]
            per_mirror_results_out.append(out)

        results_data = {
            "telescope": telescope,
            "model_version": self.args_dict.get("model_version"),
            "per_mirror_results": per_mirror_results_out,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2)

        self._logger.info("Results written to %s", str(output_file))

        if telescope and parameter_version and self.rnda_opt is not None:
            try:
                rnda_opt_rounded = [float(f"{float(v):.4f}") for v in self.rnda_opt]
                model_data_writer.ModelDataWriter.dump_model_parameter(
                    parameter_name=parameter_name,
                    value=rnda_opt_rounded,
                    instrument=str(telescope),
                    parameter_version=str(parameter_version),
                    output_file=f"{parameter_name}-{parameter_version}.json",
                    output_path=parameter_output_path,
                    unit=["deg", "dimensionless", "deg"],
                )
                self._logger.info(
                    "Exported model parameter %s (%s) to %s",
                    parameter_name,
                    str(parameter_version),
                    str(parameter_output_path),
                )
            except (OSError, ValueError, TypeError) as e:
                self._logger.warning(
                    "Failed to export model parameter %s: %s", parameter_name, str(e)
                )

    def write_d80_histogram(self):
        """Write histogram comparing measured vs simulated d80 distributions.

        Returns
        -------
        Path or None
            Path to the created histogram plot, if created.
        """
        measured = [r.get("measured_d80_mm") for r in (self.per_mirror_results or [])]
        simulated = [r.get("simulated_d80_mm") for r in (self.per_mirror_results or [])]
        return plot_psf.plot_d80_histogram(measured, simulated, self.args_dict)
