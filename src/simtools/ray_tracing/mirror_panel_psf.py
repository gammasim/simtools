"""Mirror panel PSF calculation with per-mirror d80 optimization."""

import json
import logging
import os
from dataclasses import asdict, dataclass
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
class MirrorOptimizationResult:
    """Dataclass to store the result of a single mirror RNDA optimization."""

    mirror: int  # One-based mirror index
    measured_d80_mm: float  # Measured d80 value in mm
    optimized_rnda: list[float]  # Optimized RNDA values [sigma1, fraction2, sigma2]
    simulated_d80_mm: float  # Simulated d80 after optimization
    percentage_diff: float  # Absolute percentage difference


def _optimize_single_mirror_worker(args):
    """Worker wrapper for optimizing a single mirror in parallel."""
    instance, mirror_idx, measured = args
    return instance.optimize_single_mirror(mirror_idx, measured)


class MirrorPanelPSF:
    """
    Mirror panel PSF and random reflection angle (RNDA) calculation.

    This class derives the RNDA for mirror panels by optimizing
    per-mirror d80 values using a percentage difference metric.
    Optimization uses a gradient descent with finite-difference gradients.

    Parameters
    ----------
    label : str
        Application label.
    args_dict : dict
        Dictionary with input arguments, e.g. site, telescope, data path, model version.
    """

    # Hard guardrails
    GRAD_CLIP = 1e4
    MAX_LOG_STEP = 0.25
    MAX_FRAC_STEP = 0.1
    MAX_ITER = 100

    def __init__(self, label, args_dict):
        self._logger = logging.getLogger(__name__)
        self.label = label
        self.args_dict = args_dict

        self.telescope_model, self.site_model, _ = initialize_simulation_models(
            label=label,
            site=args_dict["site"],
            telescope_name=args_dict["telescope"],
            model_version=args_dict["model_version"],
        )

        self.measured_data = self._load_measured_data()
        self.rnda_start = list(
            self.telescope_model.get_parameter_value("mirror_reflection_random_angle")
        )

        self.per_mirror_results = []
        self.rnda_opt = None
        self.final_percentage_diff = None

    def _load_measured_data(self):
        """
        Load measured d80 from ECSV file.

        Returns
        -------
        astropy.table.Column
            Column containing d80 values in mm.
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
            Zero-based index of the mirror.
        rnda_values : list of float
            RNDA values [sigma1, fraction2, sigma2].

        Returns
        -------
        float
            Simulated d80 in mm.
        """
        self.telescope_model.overwrite_model_parameter(
            "mirror_reflection_random_angle", rnda_values
        )

        mirror_number = mirror_idx + 1
        rt_label = f"{self.label}_m{mirror_number}"
        old_label = getattr(self.telescope_model, "label", None)
        # Store cached config path/name (if present) and restore after simulation.
        model_dict = getattr(self.telescope_model, "__dict__", None)
        old_path = model_dict.get("_config_file_path") if model_dict is not None else None
        old_dir = model_dict.get("_config_file_directory") if model_dict is not None else None

        # Force regeneration of config file path/name based on the new label.
        # These attributes are cached inside ModelParameter.
        try:
            self.telescope_model.label = rt_label
            if model_dict is not None:
                model_dict["_config_file_path"] = None
                model_dict["_config_file_directory"] = None

            ray = RayTracing(
                telescope_model=self.telescope_model,
                site_model=self.site_model,
                single_mirror_mode=True,
                mirror_numbers=[mirror_idx],
            )
            ray.simulate(test=self.args_dict.get("test", False), force=True)
            ray.analyze(force=True)
        finally:
            self.telescope_model.label = old_label
            if model_dict is not None:
                model_dict["_config_file_path"] = old_path
                model_dict["_config_file_directory"] = old_dir

        return float(ray.get_d80_mm())

    @staticmethod
    def _signed_pct_diff(measured, simulated):
        """Compute signed percentage difference."""
        if measured <= 0:
            raise ValueError("Measured d80 must be positive")
        return 100.0 * (simulated - measured) / measured

    def _evaluate(self, mirror_idx, measured, rnda):
        """
        Evaluate the objective function for a mirror.

        Parameters
        ----------
        mirror_idx : int
            Mirror index (0-based).
        measured : float
            Measured d80 value (mm).
        rnda : list of float
            Current RNDA values.

        Returns
        -------
        tuple of (float, float, float)
            Simulated d80, signed percentage difference, squared percentage difference.
        """
        sim = self._simulate_single_mirror_d80(mirror_idx, rnda)
        pct = self._signed_pct_diff(measured, sim)
        return sim, pct, pct * pct

    @staticmethod
    def _rnda_bounds():
        """
        Get allowed bounds for RNDA parameters.

        Returns
        -------
        list of tuple of float
            [(min1, max1), (min2, max2), (min3, max3)]
        """
        schema = names.model_parameters()["mirror_reflection_random_angle"]["data"]
        return [
            (max(schema[0]["allowed_range"]["min"], 1e-12), schema[0]["allowed_range"]["max"]),
            (schema[1]["allowed_range"]["min"], schema[1]["allowed_range"]["max"]),
            (max(schema[2]["allowed_range"]["min"], 1e-12), schema[2]["allowed_range"]["max"]),
        ]

    def optimize_single_mirror(self, mirror_idx, measured_d80):
        """
        Optimize RNDA for a single mirror using gradient descent.

        Parameters
        ----------
        mirror_idx : int
            Zero-based mirror index.
        measured_d80 : float
            Measured d80 value in mm.

        Returns
        -------
        MirrorOptimizationResult
            Optimization results for the mirror.
        """
        threshold_pct = 100 * float(self.args_dict.get("threshold", 0.05))
        learning_rate = float(self.args_dict.get("learning_rate", 1e-3))
        bounds = self._rnda_bounds()

        rnda = list(self.rnda_start)
        sim, pct, obj = self._evaluate(mirror_idx, measured_d80, rnda)
        best = {"rnda": list(rnda), "sim": sim, "pct": abs(pct), "obj": obj}

        self._logger.info("Mirror %d | initial d80 %.3f mm | pct %.2f%%", mirror_idx + 1, sim, pct)

        for iteration in range(self.MAX_ITER):
            old_rnda = list(rnda)
            for param_idx, (param_value, param_bounds) in enumerate(zip(old_rnda, bounds)):
                rnda[param_idx] = self._update_single_rnda_parameter(
                    mirror_idx=mirror_idx,
                    measured_d80=measured_d80,
                    rnda=rnda,
                    param_idx=param_idx,
                    param_value=param_value,
                    param_bounds=param_bounds,
                    learning_rate=learning_rate,
                )

            sim, pct, obj = self._evaluate(mirror_idx, measured_d80, rnda)
            pct = abs(pct)

            old_rnda_str = "[" + ", ".join(f"{v:.4g}" for v in old_rnda) + "]"
            rnda_str = "[" + ", ".join(f"{v:.4g}" for v in rnda) + "]"

            self._logger.info(
                "Iter %d | rnda %s -> %s | pct %.2f -> %.2f | lr %.2g",
                iteration + 1,
                old_rnda_str,
                rnda_str,
                best["pct"],
                pct,
                learning_rate,
            )

            if obj < best["obj"]:
                best.update(rnda=list(rnda), sim=sim, pct=pct, obj=obj)
                learning_rate *= 1.1
            else:
                rnda = list(old_rnda)
                learning_rate *= 0.5

            if best["pct"] <= threshold_pct or learning_rate < 1e-12:
                break

        return MirrorOptimizationResult(
            mirror=mirror_idx + 1,
            measured_d80_mm=float(measured_d80),
            optimized_rnda=best["rnda"],
            simulated_d80_mm=best["sim"],
            percentage_diff=best["pct"],
        )

    def _update_single_rnda_parameter(
        self,
        mirror_idx,
        measured_d80,
        rnda,
        param_idx,
        param_value,
        param_bounds,
        learning_rate,
    ):
        """
        Update a single RNDA parameter using finite-difference gradient descent.

        Parameters
        ----------
        mirror_idx : int
        measured_d80 : float
        rnda : list of float
        param_idx : int
        param_value : float
        param_bounds : tuple of float
        learning_rate : float

        Returns
        -------
        float
            Updated RNDA parameter value.
        """
        param_min, param_max = param_bounds
        is_log_param = param_idx != 1
        epsilon = max(1e-6, 0.05 * param_value) if is_log_param else 0.05

        rnda[param_idx] = min(param_max, param_value + epsilon)
        _, _, f_plus = self._evaluate(mirror_idx, measured_d80, rnda)

        rnda[param_idx] = max(param_min, param_value - epsilon)
        _, _, f_minus = self._evaluate(mirror_idx, measured_d80, rnda)

        rnda[param_idx] = param_value
        gradient = (f_plus - f_minus) / (2 * epsilon)

        # Clip the gradient to avoid excessively large steps.
        if is_log_param:
            gradient = np.clip(gradient * param_value, -self.GRAD_CLIP, self.GRAD_CLIP)
            step = np.clip(-learning_rate * gradient, -self.MAX_LOG_STEP, self.MAX_LOG_STEP)
            # clip the parameters so they are within allowed range
            return np.clip(param_value * np.exp(step), param_min, param_max)

        gradient = np.clip(gradient, -self.GRAD_CLIP, self.GRAD_CLIP)
        step = np.clip(-learning_rate * gradient, -self.MAX_FRAC_STEP, self.MAX_FRAC_STEP)
        return np.clip(param_value + step, param_min, param_max)

    def optimize_with_gradient_descent(self):
        """
        Optimize all mirrors in parallel using process pool.

        Sets
        ----
        self.per_mirror_results : list of dict
        self.rnda_opt : list of float
        self.final_percentage_diff : float
        """
        n_mirrors = len(self.measured_data)
        if self.args_dict.get("test"):
            n_mirrors = min(n_mirrors, self.args_dict.get("number_of_mirrors_to_test"))

        n_workers = int(self.args_dict.get("n_workers") or os.cpu_count())
        parent = MirrorPanelPSF(self.label, dict(self.args_dict))
        worker_args = [(parent, i, parent.measured_data[i]) for i in range(n_mirrors)]

        self.per_mirror_results = process_pool_map_ordered(
            _optimize_single_mirror_worker,
            worker_args,
            max_workers=n_workers,
            mp_start_method="fork",
        )

        self.rnda_opt = np.mean(
            [r.optimized_rnda for r in self.per_mirror_results], axis=0
        ).tolist()

        self.final_percentage_diff = float(
            np.mean([r.percentage_diff for r in self.per_mirror_results])
        )

    def write_optimization_data(self):
        """Write optimization results and optionally export as a model parameter."""
        output_dir = Path(self.args_dict.get("output_path", "."))
        telescope = self.args_dict.get("telescope")
        parameter_version = self.args_dict.get("parameter_version")
        parameter_name = "mirror_reflection_random_angle"
        parameter_output_path = output_dir / str(telescope)
        parameter_output_path.mkdir(parents=True, exist_ok=True)
        output_file = parameter_output_path / "per_mirror_rnda.json"

        per_mirror_results_out = []
        for result in self.per_mirror_results:
            result_dict = asdict(result)
            for k, v in result_dict.items():
                if k == "optimized_rnda":
                    result_dict[k] = [float(f"{x:.4f}") for x in v]
                elif isinstance(v, (int, float)):
                    result_dict[k] = float(f"{v:.4f}")
            per_mirror_results_out.append(result_dict)

        output_file.write_text(
            json.dumps(
                {
                    "telescope": telescope,
                    "model_version": self.args_dict.get("model_version"),
                    "per_mirror_results": per_mirror_results_out,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self._logger.info("Results written to %s", output_file)

        # Export averaged RNDA
        if telescope and parameter_version and self.rnda_opt is not None:
            try:
                model_data_writer.ModelDataWriter.dump_model_parameter(
                    parameter_name=parameter_name,
                    value=[float(f"{v:.4f}") for v in self.rnda_opt],
                    instrument=str(telescope),
                    parameter_version=str(parameter_version),
                    output_file=f"{parameter_name}-{parameter_version}.json",
                    output_path=parameter_output_path,
                    unit=["deg", "dimensionless", "deg"],
                )
                self._logger.info(
                    "Exported model parameter %s (%s) to %s",
                    parameter_name,
                    parameter_version,
                    parameter_output_path,
                )
            except (OSError, ValueError, TypeError) as e:
                self._logger.warning("Failed to export model parameter %s: %s", parameter_name, e)

    def write_d80_histogram(self):
        """Plot histogram of measured vs simulated d80 values."""
        measured = [r.measured_d80_mm for r in self.per_mirror_results]
        simulated = [r.simulated_d80_mm for r in self.per_mirror_results]
        return plot_psf.plot_d80_histogram(measured, simulated, self.args_dict)
