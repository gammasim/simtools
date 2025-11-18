"""Mirror panel PSF calculation."""

import logging
from pathlib import Path

import simtools.utils.general as gen
from simtools.model.model_utils import initialize_simulation_models
from simtools.ray_tracing import psf_parameter_optimisation as psf_opt
from simtools.ray_tracing.ray_tracing import RayTracing
from simtools.visualization import plot_psf


class MirrorPanelPSF:
    """
    Mirror panel PSF and random reflection angle calculation.

    This class is used to derive the random reflection angle for the mirror panels in the telescope.

    Known limitations: single Gaussian PSF model, no support for multiple PSF components (as allowed
    in the model parameters).

    Parameters
    ----------
    label: str
        Application label.
    args_dict: dict
        Dictionary with input arguments.
    db_config:
        Dictionary with database configuration.
    """

    def __init__(self, label, args_dict, db_config):
        """Initialize the MirrorPanelPSF class."""
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Initializing MirrorPanelPSF")

        self.args_dict = args_dict
        self.telescope_model, self.site_model = self._define_telescope_model(label, db_config)

        if self.args_dict["test"]:
            self.args_dict["number_of_mirrors_to_test"] = 10

        self.rnda_start = self.telescope_model.get_parameter_value("mirror_reflection_random_angle")
        self.rnda_opt = None
        self.gd_optimizer = None
        self.final_rmsd = None

    def _define_telescope_model(self, label, db_config):
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
            db_config=db_config,
            site=self.args_dict["site"],
            telescope_name=self.args_dict["telescope"],
            model_version=self.args_dict["model_version"],
        )
        if self.args_dict["mirror_list"]:
            mirror_list_file = gen.find_file(
                name=self.args_dict["mirror_list"], loc=self.args_dict["model_path"]
            )
            tel_model.overwrite_model_parameter("mirror_list", self.args_dict["mirror_list"])
            tel_model.overwrite_model_file("mirror_list", mirror_list_file)
        if self.args_dict["random_focal_length"]:
            tel_model.overwrite_model_parameter(
                "random_focal_length", str(self.args_dict["random_focal_length"])
            )

        return tel_model, site_model

    def optimize_with_gradient_descent(self):
        """Use the generalized PSF parameter optimizer for gradient descent optimization."""
        # Prepare args_dict compatible with PSFParameterOptimizer
        optimizer_args = self.args_dict.copy()
        optimizer_args["single_mirror_mode"] = True
        optimizer_args["mirror_numbers"] = (
            list(range(1, self.args_dict["number_of_mirrors_to_test"] + 1))
            if self.args_dict["test"]
            else "all"
        )
        optimizer_args["use_random_focal_length"] = self.args_dict.get(
            "use_random_focal_length", False
        )
        if self.args_dict.get("random_focal_length_seed") is not None:
            optimizer_args["random_focal_length_seed"] = self.args_dict["random_focal_length_seed"]

        data_to_plot, radius = psf_opt.load_and_process_data(optimizer_args)
        output_dir = Path(self.args_dict.get("output_path"))

        # Create optimizer for mirror_reflection_random_angle
        self.gd_optimizer = psf_opt.PSFParameterOptimizer(
            tel_model=self.telescope_model,
            site_model=self.site_model,
            args_dict=optimizer_args,
            data_to_plot=data_to_plot,
            radius=radius,
            output_dir=output_dir,
            optimize_only=["mirror_reflection_random_angle"],
        )

        threshold = self.args_dict.get("threshold")
        learning_rate = self.args_dict.get("learning_rate")

        best_params, _, gd_results = self.gd_optimizer.run_gradient_descent(
            rmsd_threshold=threshold, learning_rate=learning_rate
        )

        if best_params is None:
            raise ValueError("Gradient descent optimization failed")

        self.rnda_opt = best_params["mirror_reflection_random_angle"]
        self.final_rmsd = gd_results[-1][1] if gd_results else None

        self._logger.info(
            f"Optimization complete. RMSD: {self.final_rmsd:.6f}, Optimized values: {self.rnda_opt}"
        )

        final_simulated_data = gd_results[-1][4] if gd_results else None

        if final_simulated_data is not None:
            plot_psf.create_final_psf_comparison_plot(
                tel_model=self.telescope_model,
                optimized_params=best_params,
                data_to_plot=data_to_plot,
                output_dir=output_dir,
                final_rmsd=self.final_rmsd,
                simulated_data=final_simulated_data,
            )

    def run_simulations_and_analysis(self, rnda):
        """
        Run ray tracing simulations and analysis for one given value of rnda.

        Parameters
        ----------
        rnda: float
            Random reflection angle in degrees.
        """
        self.telescope_model.overwrite_model_parameter("mirror_reflection_random_angle", rnda)
        ray = RayTracing(
            telescope_model=self.telescope_model,
            site_model=self.site_model,
            simtel_path=self.args_dict.get("simtel_path", None),
            single_mirror_mode=True,
            mirror_numbers=(
                list(range(1, self.args_dict["number_of_mirrors_to_test"] + 1))
                if self.args_dict["test"]
                else "all"
            ),
            use_random_focal_length=self.args_dict["use_random_focal_length"],
            random_focal_length_seed=self.args_dict.get("random_focal_length_seed"),
        )
        ray.simulate(test=self.args_dict["test"], force=True)
        ray.analyze(force=True)

    def print_results(self):
        """Print results to stdout."""
        print("\nOptimization Results (RMSD-based):")
        if hasattr(self, "final_rmsd") and self.final_rmsd is not None:
            print(f"RMSD (full PSF curve): {self.final_rmsd:.6f}")

        print("\nmirror_reflection_random_angle [sigma1, fraction2, sigma2]")
        print(f"Previous values = {[f'{x:.6f}' for x in self.rnda_start]}")
        print(f"Optimized values = {[f'{x:.6f}' for x in self.rnda_opt]}\n")

    def write_optimization_data(self):
        """
        Write optimization results as a JSON model parameter file.

        Writes the optimized mirror_reflection_random_angle parameter.
        """
        output_dir = Path(self.args_dict.get("output_path"))
        best_params = {"mirror_reflection_random_angle": self.rnda_opt}

        # Use "0.0.0" as default if parameter_version is not provided
        parameter_version = self.args_dict.get("parameter_version") or "0.0.0"

        psf_opt.export_psf_parameters(
            best_pars=best_params,
            telescope=self.args_dict.get("telescope"),
            parameter_version=parameter_version,
            output_dir=output_dir,
        )
