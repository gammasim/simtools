"""Mirror panel PSF calculation with per-mirror d80 optimization."""

import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from astropy.table import Table

import simtools.utils.general as gen
from simtools.data_model import model_data_writer
from simtools.model.model_utils import initialize_simulation_models
from simtools.ray_tracing.ray_tracing import RayTracing
from simtools.settings import config as settings_config
from simtools.utils import names

_WORKER_INSTANCE = None


@dataclass
class GradientStepResult:
    """Result of evaluating a single RNDA candidate."""

    rnda: list
    simulated_d80_mm: float
    signed_pct_diff: float
    objective: float


@dataclass(frozen=True)
class RndaGradientDescentSettings:
    """Settings for RNDA gradient descent optimization."""

    threshold: float
    learning_rate: float
    grad_clip: float
    max_log_step: float
    sigma1_min: float
    sigma1_max: float
    sigma2_min: float
    sigma2_max: float
    frac2_min: float
    frac2_max: float
    max_frac_step: float
    max_iterations: int


def _worker_init(label, args_dict, db_config):
    """Initialize per-process MirrorPanelPSF instance.

    Important: DB configuration is stored in ``simtools.settings.config`` at runtime.
    With the multiprocessing "spawn" start method, workers do not inherit that state,
    so we must call ``config.load`` again in each worker.
    """
    global _WORKER_INSTANCE  # pylint: disable=global-statement
    settings_config.load(args=args_dict, db_config=db_config)
    _WORKER_INSTANCE = MirrorPanelPSF(label=label, args_dict=args_dict)


def _worker_optimize_mirror(mirror_idx):
    """Optimize a single mirror index using the per-process instance."""
    if _WORKER_INSTANCE is None:
        raise RuntimeError("Worker not initialized")
    measured_d80_mm = float(_WORKER_INSTANCE.measured_data["d80"][mirror_idx])
    focal_length_m = float(_WORKER_INSTANCE.measured_data["focal_length"][mirror_idx])
    return _WORKER_INSTANCE.optimize_single_mirror(mirror_idx, measured_d80_mm, focal_length_m)


class MirrorPanelPSF:
    """
    Mirror panel PSF and random reflection angle calculation.

    This class derives the random reflection angle (RNDA) for mirror panels
    by optimizing per-mirror d80 values using percentage difference as the metric.

    Parameters
    ----------
    label: str
        Application label.
    args_dict: dict
        Dictionary with input arguments.
    """

    # Internal guard-rail defaults for the RNDA optimizer.
    DEFAULT_RNDA_GRAD_CLIP: float = 1e4
    DEFAULT_RNDA_MAX_LOG_STEP: float = 0.25
    DEFAULT_RNDA_MAX_FRAC_STEP: float = 0.1
    DEFAULT_RNDA_MAX_ITERATIONS: int = 100

    def __init__(self, label, args_dict):
        """Initialize the MirrorPanelPSF class."""
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Initializing MirrorPanelPSF")

        self.label = label
        self.args_dict = args_dict
        self.telescope_model, self.site_model = self._define_telescope_model(label)

        # Load measured d80 and focal_length data
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
        Load measured d80 and focal_length data from ECSV file.

        Returns
        -------
        Table
            Astropy table with d80 (mm) and focal_length (m) columns.
        """
        data_file = gen.find_file(self.args_dict["data"], self.args_dict.get("model_path", "."))
        table = Table.read(data_file, format="ascii.ecsv")

        # Ensure required columns exist
        if "d80" not in table.colnames or "focal_length" not in table.colnames:
            raise ValueError("Data file must contain 'd80' and 'focal_length' columns")

        self._logger.info("Loaded %d mirrors from %s", len(table), str(data_file))
        return table

    def _define_telescope_model(self, label):
        """
        Define telescope model.

        This includes updating the configuration with mirror list and/or random focal length given
        as input.

        Returns
        -------
        tel : TelescopeModel
            The telescope model.
        site_model : SiteModel
            The site model.
        """
        tel_model, site_model, _ = initialize_simulation_models(
            label=label,
            site=self.args_dict["site"],
            telescope_name=self.args_dict["telescope"],
            model_version=self.args_dict["model_version"],
        )
        return tel_model, site_model

    def _simulate_single_mirror_d80(self, mirror_idx, _focal_length_m, rnda_values):
        """
        Simulate a single mirror and return its d80.

        Parameters
        ----------
        mirror_idx : int
            Mirror index (0-based).
        focal_length_m : float
            Mirror focal length in meters (currently unused; mirror-panel focal length is
            derived from the telescope model in single-mirror mode).
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
                use_random_focal_length=False,
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
        return 100.0 * (float(simulated_d80) - measured) / measured

    def _evaluate_rnda_candidate(
        self,
        *,
        mirror_idx: int,
        measured_d80_mm: float,
        focal_length_m: float,
        rnda_values,
    ) -> GradientStepResult:
        """Evaluate a candidate RNDA by simulating and computing objective."""
        simulated_d80_mm = float(
            self._simulate_single_mirror_d80(mirror_idx, focal_length_m, rnda_values)
        )
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
        *,
        mirror_idx: int,
        measured_d80_mm: float,
        focal_length_m: float,
        current_rnda,
        param_index: int,
        plus_value: float,
        minus_value: float,
    ) -> float:
        """Compute the derivative d(objective)/d(param) via symmetric finite differences."""
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
            focal_length_m=focal_length_m,
            rnda_values=rnda_plus,
        )
        minus_eval = self._evaluate_rnda_candidate(
            mirror_idx=mirror_idx,
            measured_d80_mm=measured_d80_mm,
            focal_length_m=focal_length_m,
            rnda_values=rnda_minus,
        )
        return float((plus_eval.objective - minus_eval.objective) / denom)

    def _run_rnda_gradient_descent(
        self,
        mirror_idx: int,
        measured_d80_mm: float,
        focal_length_m: float,
        current_rnda,
        settings: RndaGradientDescentSettings,
    ):
        """Run a 3-parameter RNDA gradient descent (sigma1, fraction2, sigma2).

        Returns best_rnda, best_sim_d80, best_pct_diff.
        """
        # pylint: disable=too-many-locals
        current_eval = self._evaluate_rnda_candidate(
            mirror_idx=mirror_idx,
            measured_d80_mm=measured_d80_mm,
            focal_length_m=focal_length_m,
            rnda_values=current_rnda,
        )

        best_rnda = list(current_eval.rnda)
        best_sim_d80 = float(current_eval.simulated_d80_mm)
        best_obj = float(current_eval.objective)
        best_pct_diff = abs(float(current_eval.signed_pct_diff))

        self._logger.info(
            "  Initial: simulated d80 = %.3f mm, signed pct diff = %.2f%%",
            best_sim_d80,
            current_eval.signed_pct_diff,
        )

        if best_pct_diff <= settings.threshold * 100:
            self._logger.info("  Already converged!")
            return best_rnda, best_sim_d80, best_pct_diff

        learning_rate = float(settings.learning_rate)
        for iteration in range(settings.max_iterations):
            sigma1, frac2, sigma2 = map(float, current_rnda)

            specs = [
                {
                    "idx": 0,
                    "kind": "log",
                    "value": sigma1,
                    "min": settings.sigma1_min,
                    "max": settings.sigma1_max,
                    "eps": max(1e-6, 0.05 * sigma1),
                },
                {
                    "idx": 1,
                    "kind": "linear",
                    "value": frac2,
                    "min": settings.frac2_min,
                    "max": settings.frac2_max,
                    "eps": max(1e-6, 0.02),
                },
                {
                    "idx": 2,
                    "kind": "log",
                    "value": sigma2,
                    "min": settings.sigma2_min,
                    "max": settings.sigma2_max,
                    "eps": max(1e-6, 0.05 * sigma2),
                },
            ]

            new_values = [sigma1, frac2, sigma2]
            for spec in specs:
                v = float(spec["value"])
                v_plus = min(float(spec["max"]), v + float(spec["eps"]))
                v_minus = max(float(spec["min"]), v - float(spec["eps"]))

                grad = self._finite_difference_objective_gradient(
                    mirror_idx=mirror_idx,
                    measured_d80_mm=measured_d80_mm,
                    focal_length_m=focal_length_m,
                    current_rnda=current_rnda,
                    param_index=int(spec["idx"]),
                    plus_value=float(v_plus),
                    minus_value=float(v_minus),
                )

                if spec["kind"] == "log":
                    grad_log = float(np.clip(grad * v, -settings.grad_clip, settings.grad_clip))
                    log_step = float(
                        np.clip(
                            -learning_rate * grad_log,
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
                            -learning_rate * grad,
                            -settings.max_frac_step,
                            settings.max_frac_step,
                        )
                    )
                    new_v = float(np.clip(v + step, float(spec["min"]), float(spec["max"])))

                new_values[int(spec["idx"])] = new_v

            new_sigma1, new_frac2, new_sigma2 = map(float, new_values)
            new_rnda = [new_sigma1, new_frac2, new_sigma2]
            new_eval = self._evaluate_rnda_candidate(
                mirror_idx=mirror_idx,
                measured_d80_mm=measured_d80_mm,
                focal_length_m=focal_length_m,
                rnda_values=new_rnda,
            )

            new_sim_d80 = float(new_eval.simulated_d80_mm)
            new_obj = float(new_eval.objective)
            new_pct_diff = abs(float(new_eval.signed_pct_diff))

            self._logger.info(
                (
                    "  Iter %d: s1=%.6f->%.6f f2=%.3f->%.3f s2=%.6f->%.6f, "
                    "|pct|=%.2f%%->%.2f%%, lr=%.3g"
                ),
                iteration + 1,
                sigma1,
                new_sigma1,
                frac2,
                new_frac2,
                sigma2,
                new_sigma2,
                best_pct_diff,
                new_pct_diff,
                learning_rate,
            )

            if new_obj < best_obj:
                best_obj = new_obj
                best_pct_diff = new_pct_diff
                best_rnda = list(new_rnda)
                best_sim_d80 = new_sim_d80
                current_rnda = list(new_rnda)
                learning_rate *= 1.1
            else:
                learning_rate *= 0.5

            if best_pct_diff <= settings.threshold * 100:
                self._logger.info("  Converged at iteration %d!", iteration + 1)
                break

            if learning_rate < 1e-12:
                self._logger.info("  Learning rate too small, stopping.")
                break

        self._logger.info(
            "  Final: simulated d80 = %.3f mm, pct diff = %.2f%%",
            best_sim_d80,
            best_pct_diff,
        )
        return best_rnda, best_sim_d80, best_pct_diff

    def optimize_single_mirror(self, mirror_idx, measured_d80_mm, focal_length_m):
        """
        Optimize RNDA for a single mirror using gradient descent.

        Parameters
        ----------
        mirror_idx : int
            Mirror index (0-based).
        measured_d80_mm : float
            Measured d80 in mm.
        focal_length_m : float
            Focal length in meters.

        Returns
        -------
        dict
            Optimization result with rnda, simulated_d80, percentage_diff.
        """
        # pylint: disable=too-many-locals
        threshold = float(self.args_dict.get("threshold", 0.05))  # 5% default
        learning_rate = float(self.args_dict.get("learning_rate", 0.001))

        grad_clip = float(self.DEFAULT_RNDA_GRAD_CLIP)
        max_log_step = float(self.DEFAULT_RNDA_MAX_LOG_STEP)
        # Bounds are schema-defined. Defaults are only a fallback if schema metadata is missing.
        param_name = "mirror_reflection_random_angle"

        def _get_allowed_range_from_schema(p_name: str, idx: int):
            schema = names.model_parameters().get(p_name) or {}
            data = schema.get("data")
            if not isinstance(data, list) or idx < 0 or idx >= len(data):
                return None, None
            allowed_range = (data[idx] or {}).get("allowed_range")
            if not isinstance(allowed_range, dict):
                return None, None
            return allowed_range.get("min"), allowed_range.get("max")

        sigma1_min, sigma1_max = _get_allowed_range_from_schema(param_name, 0)
        frac2_min, frac2_max = _get_allowed_range_from_schema(param_name, 1)
        sigma2_min, sigma2_max = _get_allowed_range_from_schema(param_name, 2)

        sigma1_min = 1e-4 if sigma1_min is None else float(sigma1_min)
        sigma1_max = 0.1 if sigma1_max is None else float(sigma1_max)
        frac2_min = 0.0 if frac2_min is None else float(frac2_min)
        frac2_max = 1.0 if frac2_max is None else float(frac2_max)
        sigma2_min = 1e-4 if sigma2_min is None else float(sigma2_min)
        sigma2_max = 0.1 if sigma2_max is None else float(sigma2_max)

        # Sigma parameters are used in log-space updates; ensure strictly positive lower bounds.
        sigma1_min = max(sigma1_min, 1e-12)
        sigma2_min = max(sigma2_min, 1e-12)
        max_frac_step = float(self.DEFAULT_RNDA_MAX_FRAC_STEP)
        max_iterations = int(self.DEFAULT_RNDA_MAX_ITERATIONS)

        current_rnda = list(self.rnda_start)

        self._logger.info(
            f"Mirror {mirror_idx + 1}: measured d80 = {measured_d80_mm:.3f} mm, "
            f"focal_length = {focal_length_m:.3f} m"
        )

        settings = RndaGradientDescentSettings(
            threshold=threshold,
            learning_rate=learning_rate,
            grad_clip=grad_clip,
            max_log_step=max_log_step,
            sigma1_min=sigma1_min,
            sigma1_max=sigma1_max,
            sigma2_min=sigma2_min,
            sigma2_max=sigma2_max,
            frac2_min=frac2_min,
            frac2_max=frac2_max,
            max_frac_step=max_frac_step,
            max_iterations=max_iterations,
        )
        best_rnda, best_sim_d80, best_pct_diff = self._run_rnda_gradient_descent(
            mirror_idx=mirror_idx,
            measured_d80_mm=measured_d80_mm,
            focal_length_m=focal_length_m,
            current_rnda=current_rnda,
            settings=settings,
        )

        return {
            "mirror": mirror_idx + 1,
            "measured_d80_mm": measured_d80_mm,
            "focal_length_m": focal_length_m,
            "optimized_rnda": best_rnda,
            "simulated_d80_mm": best_sim_d80,
            "percentage_diff": best_pct_diff,
        }

    def _optimize_mirrors_parallel(self, n_mirrors, n_workers):
        # Ensure worker processes don't recursively attempt to parallelize.
        worker_args = dict(self.args_dict)
        worker_args["parallel"] = False

        # Snapshot DB configuration from the parent process and pass to workers.
        worker_db_config = dict(settings_config.db_config) if settings_config.db_config else {}

        mp_start_method = "fork" if os.name == "posix" else "spawn"
        ctx = get_context(mp_start_method)

        results = [None] * n_mirrors
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(self.label, worker_args, worker_db_config),
        ) as executor:
            futures = {executor.submit(_worker_optimize_mirror, i): i for i in range(n_mirrors)}
            for fut in as_completed(futures):
                mirror_idx = futures[fut]
                results[mirror_idx] = fut.result()
        return results

    def optimize_with_gradient_descent(self):
        """
        Optimize RNDA for each mirror individually using percentage difference metric.

        The final optimized RNDA is the average of per-mirror optimized values.
        """
        n_mirrors = len(self.measured_data)
        if self.args_dict.get("test", False):
            n_mirrors = min(n_mirrors, self.args_dict.get("number_of_mirrors_to_test", 10))

        self._logger.info("Optimizing RNDA for %d mirrors...", n_mirrors)

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

    def _format_results_lines(self):
        lines = []

        lines.append("")
        lines.append("=" * 70)
        lines.append("Single-Mirror d80 Optimization Results (Percentage Difference Metric)")
        lines.append("=" * 70)

        lines.append("")
        lines.append(f"Number of mirrors optimized: {len(self.per_mirror_results)}")
        lines.append(f"Mean percentage difference: {self.final_percentage_diff:.2f}%")

        have_plot_d80 = any(
            (isinstance(r, dict) and ("simulated_d80_mm_plot" in r or "percentage_diff_plot" in r))
            for r in self.per_mirror_results
        )

        lines.append("")
        lines.append("Per-mirror results:")
        lines.append("-" * 120 if have_plot_d80 else "-" * 90)
        if have_plot_d80:
            lines.append(
                f"{'Mirror':>6} {'Meas d80':>10} {'Sim d80':>10} {'Pct Diff':>10} "
                f"{'Plot Sim':>10} {'Plot %':>10} {'Optimized RNDA [sigma1, frac2, sigma2]':<40}"
            )
            lines.append(
                f"{'':>6} {'(mm)':>10} {'(mm)':>10} {'(%)':>10} "
                f"{'(mm)':>10} {'(%)':>10} {'(deg, -, deg)':<40}"
            )
        else:
            lines.append(
                f"{'Mirror':>6} {'Meas d80':>10} {'Sim d80':>10} {'Pct Diff':>10} "
                f"{'Optimized RNDA [sigma1, frac2, sigma2]':<40}"
            )
            lines.append(f"{'':>6} {'(mm)':>10} {'(mm)':>10} {'(%)':>10} {'(deg, -, deg)':<40}")
        lines.append("-" * 120 if have_plot_d80 else "-" * 90)

        for r in self.per_mirror_results:
            rnda_str = (
                f"[{r['optimized_rnda'][0]:.4f}, {r['optimized_rnda'][1]:.4f}, "
                f"{r['optimized_rnda'][2]:.4f}]"
            )
            if have_plot_d80:
                plot_sim = r.get("simulated_d80_mm_plot", float("nan"))
                plot_pct = r.get("percentage_diff_plot", float("nan"))
                lines.append(
                    f"{r['mirror']:>6} {r['measured_d80_mm']:>10.3f} "
                    f"{r['simulated_d80_mm']:>10.3f} {r['percentage_diff']:>10.2f} "
                    f"{plot_sim:>10.3f} {plot_pct:>10.2f} {rnda_str:<40}"
                )
            else:
                lines.append(
                    f"{r['mirror']:>6} {r['measured_d80_mm']:>10.3f} "
                    f"{r['simulated_d80_mm']:>10.3f} {r['percentage_diff']:>10.2f} "
                    f"{rnda_str:<40}"
                )

        lines.append("-" * 120 if have_plot_d80 else "-" * 90)
        lines.append("")
        lines.append("mirror_reflection_random_angle [sigma1, fraction2, sigma2]")
        lines.append(f"Previous values = {[f'{x:.4f}' for x in self.rnda_start]}")
        lines.append(f"Optimized values (averaged) = {[f'{x:.4f}' for x in self.rnda_opt]}")
        lines.append("")

        return lines

    def write_results_log(self):
        """Write the results table to a ``.log`` file."""
        output_dir = Path(self.args_dict.get("output_path", "."))
        output_dir.mkdir(parents=True, exist_ok=True)

        tel = str(self.args_dict.get("telescope", "")).strip()
        model_version = str(self.args_dict.get("model_version", "")).strip()
        suffix = ""
        if tel or model_version:
            suffix = f"_{tel}_{model_version}"
        filename = f"mirror_rnda_results{suffix}.log"

        out_path = Path(filename)
        if not out_path.is_absolute():
            out_path = output_dir / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        text = "\n".join(self._format_results_lines()) + "\n"
        out_path.write_text(text, encoding="utf-8")

        self._logger.info("Results written to %s", str(out_path))
        return str(out_path)

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
                "focal_length_m",
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
            except Exception as e:  # pylint: disable=broad-exception-caught
                self._logger.warning(
                    "Failed to export model parameter %s: %s", parameter_name, str(e)
                )

    def write_d80_histogram(self):
        """Write histogram comparing measured vs simulated d80 distributions.

        Returns
        -------
        str | None
            Path to the written file, or None if nothing was written.
        """
        out_name = self.args_dict.get("d80_hist")
        if not out_name:
            return None

        import matplotlib as mpl  # pylint: disable=import-outside-toplevel

        mpl.use("Agg", force=True)
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

        output_dir = Path(self.args_dict.get("output_path", "."))
        out_path = Path(out_name)
        if not out_path.is_absolute():
            out_path = output_dir / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        measured = np.asarray(
            [r.get("measured_d80_mm") for r in (self.per_mirror_results or [])], dtype=float
        )
        simulated = np.asarray(
            [r.get("simulated_d80_mm") for r in (self.per_mirror_results or [])], dtype=float
        )

        measured = measured[np.isfinite(measured)]
        simulated = simulated[np.isfinite(simulated)]

        if measured.size == 0 or simulated.size == 0:
            self._logger.warning("No valid d80 values available to plot histogram")
            return None

        bins = 25

        all_vals = np.concatenate([measured, simulated])
        x_min = float(np.nanmin(all_vals))
        x_max = float(np.nanmax(all_vals))
        if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
            self._logger.warning("Invalid d80 range for histogram")
            return None

        bin_edges = np.linspace(x_min, x_max, bins + 1)

        meas_mean = float(np.mean(measured))
        meas_rms = float(np.std(measured, ddof=0))
        sim_mean = float(np.mean(simulated))
        sim_rms = float(np.std(simulated, ddof=0))

        fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
        ax.hist(
            measured,
            bins=bin_edges,
            alpha=0.55,
            color="tab:red",
            edgecolor="white",
            label=f"Measured (mean={meas_mean:.2f} mm, rms={meas_rms:.2f} mm)",
        )
        ax.hist(
            simulated,
            bins=bin_edges,
            alpha=0.55,
            color="tab:blue",
            edgecolor="white",
            label=f"Simulated (mean={sim_mean:.2f} mm, rms={sim_rms:.2f} mm)",
        )

        ax.axvline(meas_mean, color="tab:red", linestyle="--", linewidth=1)
        ax.axvline(sim_mean, color="tab:blue", linestyle="--", linewidth=1)

        ax.set_xlabel("d80 (mm)")
        ax.set_ylabel("Count")
        tel = self.args_dict.get("telescope", "")
        model_version = self.args_dict.get("model_version", "")
        ax.set_title(f"d80 distributions ({tel} {model_version})")
        ax.legend(loc="best", fontsize=9, frameon=True)

        fig.savefig(out_path)
        plt.close(fig)

        self._logger.info("d80 histogram written to %s", str(out_path))
        return str(out_path)
