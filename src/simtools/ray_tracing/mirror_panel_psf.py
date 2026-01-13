"""Mirror panel PSF calculation with per-mirror d80 optimization."""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
import json
import logging
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.table import Table

import simtools.utils.general as gen
from simtools.model.model_utils import initialize_simulation_models
from simtools.ray_tracing.ray_tracing import RayTracing


_WORKER_INSTANCE = None


def _worker_init(label, args_dict, db_config):
    """Initializer for per-process MirrorPanelPSF instance.

    Important: DB configuration is stored in ``simtools.settings.config`` at runtime.
    With the multiprocessing "spawn" start method, workers do not inherit that state,
    so we must call ``config.load`` again in each worker.
    """
    global _WORKER_INSTANCE  # noqa: PLW0603
    from simtools.settings import config as _config

    _config.load(args=args_dict, db_config=db_config)
    _WORKER_INSTANCE = MirrorPanelPSF(label=label, args_dict=args_dict)


def _worker_optimize_mirror(mirror_idx):
    """Optimize a single mirror index using the per-process instance."""
    if _WORKER_INSTANCE is None:
        raise RuntimeError("Worker not initialized")
    measured_d80_mm = float(_WORKER_INSTANCE.measured_data["d80"][mirror_idx])
    focal_length_m = float(_WORKER_INSTANCE.measured_data["focal_length"][mirror_idx])
    return _WORKER_INSTANCE._optimize_single_mirror(mirror_idx, measured_d80_mm, focal_length_m)


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
            self.args_dict["number_of_mirrors_to_test"] = 50

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

        self._logger.info(f"Loaded {len(table)} mirrors from {data_file}")
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
        if self.args_dict.get("mirror_list"):
            mirror_list_file = gen.find_file(
                name=self.args_dict["mirror_list"], loc=self.args_dict.get("model_path", ".")
            )
            tel_model.overwrite_model_parameter("mirror_list", self.args_dict["mirror_list"])
            tel_model.overwrite_model_file("mirror_list", mirror_list_file)

        return tel_model, site_model

    def _simulate_single_mirror_d80(self, mirror_idx, focal_length_m, rnda_values):
        """
        Simulate a single mirror and return its d80.

        Parameters
        ----------
        mirror_idx : int
            Mirror index (0-based).
        focal_length_m : float
            Mirror focal length in meters.
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

        # Create ray tracing for single mirror.
        # Use a per-mirror label so parallel runs do not collide on output filenames.
        # IMPORTANT: RayTracing and SimulatorRayTracing derive filenames from
        # telescope_model.label. Passing RayTracing(label=...) alone breaks this link
        # and causes FileNotFoundError when RayTracing looks for the photons file.
        rt_label = f"{self.label}_m{mirror_idx + 1}"
        old_label = getattr(self.telescope_model, "label", None)
        old_config_file_path = getattr(self.telescope_model, "_config_file_path", None)
        old_config_file_directory = getattr(self.telescope_model, "_config_file_directory", None)
        try:
            self.telescope_model.label = rt_label
            # Force regeneration of config file path/name based on the new label.
            # These attributes are cached inside ModelParameter.
            if hasattr(self.telescope_model, "_config_file_path"):
                self.telescope_model._config_file_path = None
            if hasattr(self.telescope_model, "_config_file_directory"):
                self.telescope_model._config_file_directory = None
            ray = RayTracing(
                telescope_model=self.telescope_model,
                site_model=self.site_model,
                single_mirror_mode=True,
                mirror_numbers=[mirror_idx],
                use_random_focal_length=False,
            )

            # Simulate and analyze
            ray.simulate(test=self.args_dict.get("test", False), force=True)
            ray.analyze(force=True)
        finally:
            if old_label is not None:
                self.telescope_model.label = old_label
            if hasattr(self.telescope_model, "_config_file_path"):
                self.telescope_model._config_file_path = old_config_file_path
            if hasattr(self.telescope_model, "_config_file_directory"):
                self.telescope_model._config_file_directory = old_config_file_directory

        # Get d80 from results (in cm, convert to mm)
        results = ray._results
        d80_cm = results["d80_cm"][0].value
        d80_mm = d80_cm * 10.0

        return d80_mm

    def _calculate_percentage_difference(self, measured_d80, simulated_d80):
        """Calculate signed percentage difference."""
        measured = float(measured_d80)
        return 100.0 * (float(simulated_d80) - measured) / measured

    def _optimize_single_mirror(self, mirror_idx, measured_d80_mm, focal_length_m):
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
        threshold = float(self.args_dict.get("threshold", 0.05))  # 5% default
        learning_rate = float(self.args_dict.get("learning_rate", 0.0001))
        n_runs_per_eval = int(self.args_dict.get("n_runs_per_eval", 1))
        grad_clip = float(self.args_dict.get("grad_clip", 1e4))
        max_log_step = float(self.args_dict.get("max_log_step", 0.25))
        sigma_min = float(self.args_dict.get("sigma1_min", 1e-4))
        sigma_max = float(self.args_dict.get("sigma1_max", 0.1))
        max_iterations = 100

        # Start with current RNDA values
        current_rnda = list(self.rnda_start)
        best_rnda = list(current_rnda)

        self._logger.info(
            f"Mirror {mirror_idx + 1}: measured d80 = {measured_d80_mm:.3f} mm, "
            f"focal_length = {focal_length_m:.3f} m"
        )

        def simulate_avg(rnda_values):
            d80_values = [
                self._simulate_single_mirror_d80(mirror_idx, focal_length_m, rnda_values)
                for _ in range(max(1, n_runs_per_eval))
            ]
            return float(np.mean(d80_values))

        def objective_from_sim(simulated_d80_mm):
            signed_pct = self._calculate_percentage_difference(measured_d80_mm, simulated_d80_mm)
            return float(signed_pct), float(signed_pct * signed_pct)

        # Get initial simulation
        sim_d80 = simulate_avg(current_rnda)
        signed_pct, obj = objective_from_sim(sim_d80)
        best_pct_diff = abs(signed_pct)
        best_sim_d80 = sim_d80
        best_obj = obj

        self._logger.info(
            f"  Initial: simulated d80 = {sim_d80:.3f} mm, signed pct diff = {signed_pct:.2f}%"
        )

        if best_pct_diff <= threshold * 100:
            self._logger.info("  Already converged!")
            return {
                "mirror": mirror_idx + 1,
                "measured_d80_mm": measured_d80_mm,
                "focal_length_m": focal_length_m,
                "optimized_rnda": best_rnda,
                "simulated_d80_mm": best_sim_d80,
                "percentage_diff": best_pct_diff,
            }

        # Gradient descent on sigma1 (primary component)
        # - Use a smooth objective: squared signed percentage difference
        # - Use central differences with a relative step in sigma1
        # - Update in log-space to avoid huge absolute steps when sigma1 is small
        for iteration in range(max_iterations):
            sigma1 = float(current_rnda[0])
            # Relative finite-difference step (avoid eps too small)
            epsilon = max(1e-6, 0.05 * sigma1)

            rnda_plus = list(current_rnda)
            rnda_minus = list(current_rnda)
            rnda_plus[0] = min(sigma_max, sigma1 + epsilon)
            rnda_minus[0] = max(sigma_min, sigma1 - epsilon)

            sim_plus = simulate_avg(rnda_plus)
            _, obj_plus = objective_from_sim(sim_plus)

            sim_minus = simulate_avg(rnda_minus)
            _, obj_minus = objective_from_sim(sim_minus)

            denom = (rnda_plus[0] - rnda_minus[0])
            if denom <= 0:
                self._logger.info("  Sigma1 step collapsed; stopping.")
                break

            # d(obj)/d(sigma1)
            grad_sigma = (obj_plus - obj_minus) / denom
            # Convert to d(obj)/d(log(sigma1)) = d(obj)/d(sigma1) * sigma1
            grad_log = grad_sigma * sigma1
            grad_log = float(np.clip(grad_log, -grad_clip, grad_clip))

            # Propose log-space update with clipping
            log_step = float(np.clip(-learning_rate * grad_log, -max_log_step, max_log_step))
            new_sigma1 = float(np.clip(sigma1 * float(np.exp(log_step)), sigma_min, sigma_max))

            new_rnda = list(current_rnda)
            new_rnda[0] = new_sigma1

            new_sim_d80 = simulate_avg(new_rnda)
            new_signed_pct, new_obj = objective_from_sim(new_sim_d80)
            new_pct_diff = abs(new_signed_pct)

            # Debug: print gradient info (now for smooth objective)
            self._logger.info(
                f"  Iter {iteration + 1}: sigma1={sigma1:.6f} -> {new_sigma1:.6f}, "
                f"d80={best_sim_d80:.3f}mm, d80_new={new_sim_d80:.3f}mm, "
                f"|pct|={best_pct_diff:.2f}% -> {new_pct_diff:.2f}%, "
                f"grad_log={grad_log:.3g}, log_step={log_step:.3g}, lr={learning_rate:.3g}"
            )

            if new_obj < best_obj:
                best_obj = new_obj
                best_pct_diff = new_pct_diff
                best_rnda = list(new_rnda)
                best_sim_d80 = new_sim_d80
                current_rnda = list(new_rnda)
                learning_rate *= 1.1  # Increase LR on success
            else:
                learning_rate *= 0.5  # Decrease LR on failure

            if best_pct_diff <= threshold * 100:
                self._logger.info(f"  Converged at iteration {iteration + 1}!")
                break

            if learning_rate < 1e-12:
                self._logger.info("  Learning rate too small, stopping.")
                break

        self._logger.info(
            f"  Final: simulated d80 = {best_sim_d80:.3f} mm, pct diff = {best_pct_diff:.2f}%"
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
        from simtools.settings import config as _config

        worker_db_config = dict(_config.db_config) if _config.db_config else {}

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
                try:
                    results[mirror_idx] = fut.result()
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(
                        "Parallel optimization failed for mirror "
                        f"{mirror_idx + 1}. If you see repeated sim_telarray failures, "
                        "try reducing --n_workers."
                    ) from exc
        return results

    def optimize_with_gradient_descent(self):
        """
        Optimize RNDA for each mirror individually using percentage difference metric.

        The final optimized RNDA is the average of per-mirror optimized values.
        """
        n_mirrors = len(self.measured_data)
        if self.args_dict.get("test", False):
            n_mirrors = min(n_mirrors, self.args_dict.get("number_of_mirrors_to_test", 3))

        self._logger.info(f"Optimizing RNDA for {n_mirrors} mirrors...")

        n_workers = int(self.args_dict.get("n_workers", 0) or 0)
        if n_workers <= 0:
            n_workers = min(os.cpu_count() or 1, 8)
        n_workers = max(1, n_workers)

        self._logger.info(f"Running per-mirror optimization with {n_workers} worker processes")
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

    def print_results(self):
        """Print results to stdout."""
        print("\n" + "=" * 70)
        print("Single-Mirror d80 Optimization Results (Percentage Difference Metric)")
        print("=" * 70)

        print(f"\nNumber of mirrors optimized: {len(self.per_mirror_results)}")
        print(f"Mean percentage difference: {self.final_percentage_diff:.2f}%")

        print("\nPer-mirror results:")
        print("-" * 90)
        print(f"{'Mirror':>6} {'Meas d80':>10} {'Sim d80':>10} {'Pct Diff':>10} {'Optimized RNDA [sigma1, frac2, sigma2]':<40}")
        print(f"{'':>6} {'(mm)':>10} {'(mm)':>10} {'(%)':>10} {'(deg, -, deg)':<40}")
        print("-" * 90)

        for r in self.per_mirror_results:
            rnda_str = f"[{r['optimized_rnda'][0]:.6f}, {r['optimized_rnda'][1]:.6f}, {r['optimized_rnda'][2]:.6f}]"
            print(
                f"{r['mirror']:>6} {r['measured_d80_mm']:>10.3f} {r['simulated_d80_mm']:>10.3f} "
                f"{r['percentage_diff']:>10.2f} {rnda_str:<40}"
            )

        print("-" * 90)
        print("\nmirror_reflection_random_angle [sigma1, fraction2, sigma2]")
        print(f"Previous values = {[f'{x:.6f}' for x in self.rnda_start]}")
        print(f"Optimized values (averaged) = {[f'{x:.6f}' for x in self.rnda_opt]}\n")

    def write_optimization_data(self):
        """Write optimization results as a JSON file."""
        output_dir = Path(self.args_dict.get("output_path", "."))
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "per_mirror_optimization_results.json"

        results_data = {
            "telescope": self.args_dict.get("telescope"),
            "model_version": self.args_dict.get("model_version"),
            "optimization_metric": "percentage_difference",
            "mean_percentage_diff": self.final_percentage_diff,
            "original_rnda": self.rnda_start,
            "optimized_rnda_averaged": self.rnda_opt,
            "per_mirror_results": self.per_mirror_results,
        }

        with open(output_file, "w") as f:
            json.dump(results_data, f, indent=2)

        self._logger.info(f"Results written to {output_file}")
