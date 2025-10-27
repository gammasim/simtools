"""Mirror panel PSF calculation."""

import logging

import astropy.units as u
import numpy as np
from astropy.table import QTable, Table

import simtools.data_model.model_data_writer as writer
import simtools.utils.general as gen
from simtools.data_model.metadata_collector import MetadataCollector
from simtools.model.model_utils import initialize_simulation_models
from simtools.ray_tracing.ray_tracing import RayTracing


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
            self.args_dict["number_of_mirrors_to_test"] = 2

        if self.args_dict["psf_measurement"]:
            self._get_psf_containment()
        if not self.args_dict["psf_measurement_containment_mean"]:
            raise ValueError("Missing PSF measurement")

        self.mean_d80 = None
        self.sig_d80 = None
        self.rnda_start = self._get_starting_value()
        self.rnda_opt = None
        self.results_rnda = []
        self.results_mean = []
        self.results_sig = []

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
        if self.args_dict["mirror_list"] is not None:
            mirror_list_file = gen.find_file(
                name=self.args_dict["mirror_list"], loc=self.args_dict["model_path"]
            )
            tel_model.overwrite_model_parameter("mirror_list", self.args_dict["mirror_list"])
            tel_model.overwrite_model_file("mirror_list", mirror_list_file)
        if self.args_dict["random_focal_length"] is not None:
            tel_model.overwrite_model_parameter(
                "random_focal_length", str(self.args_dict["random_focal_length"])
            )

        return tel_model, site_model

    def _get_psf_containment(self):
        """Read measured single-mirror point-spread function from file and return mean and sigma."""
        # If this is a test, read just the first few lines since we only simulate those mirrors
        data_end = (
            self.args_dict["number_of_mirrors_to_test"] + 1 if self.args_dict["test"] else None
        )
        _psf_list = Table.read(
            self.args_dict["psf_measurement"], format="ascii.ecsv", data_end=data_end
        )
        try:
            self.args_dict["psf_measurement_containment_mean"] = np.nanmean(
                np.array(_psf_list["psf_opt"].to("cm").value)
            )
            self.args_dict["psf_measurement_containment_sigma"] = np.nanstd(
                np.array(_psf_list["psf_opt"].to("cm").value)
            )
        except KeyError as exc:
            raise KeyError(
                f"Missing column psf measurement (psf_opt) in {self.args_dict['psf_measurement']}"
            ) from exc

        self._logger.info(
            f"Determined PSF containment to {self.args_dict['psf_measurement_containment_mean']:.4}"
            f" +- {self.args_dict['psf_measurement_containment_sigma']:.4} cm"
        )

    def derive_random_reflection_angle(self, save_figures=False):
        """
        Minimize the difference between measured and simulated PSF for reflection angle.

        Main loop of the optimization process. The method iterates over different values of the
        random reflection angle until the difference in the mean value of the D80 containment
        is minimal.
        """
        if self.args_dict["no_tuning"]:
            self.rnda_opt = self.rnda_start
        else:
            self._optimize_reflection_angle()

        self.mean_d80, self.sig_d80 = self.run_simulations_and_analysis(
            self.rnda_opt, save_figures=save_figures
        )

    def _optimize_reflection_angle(self, step_size=0.1, max_iteration=100):
        """
        Optimize the random reflection angle to minimize the difference in D80 containment.

        Parameters
        ----------
        step_size: float
            Initial step size for optimization.
        max_iteration: int
            Maximum number of iterations.

        Raises
        ------
        ValueError
            If the optimization reaches the maximum number of iterations without converging.

        """
        relative_tolerance_d80 = self.args_dict["rtol_psf_containment"]
        self._logger.info(
            "Optimizing random reflection angle "
            f"(relative tolerance = {relative_tolerance_d80}, "
            f"step size = {step_size}, max iteration = {max_iteration})"
        )

        def collect_results(rnda, mean, sig):
            self.results_rnda.append(rnda)
            self.results_mean.append(mean)
            self.results_sig.append(sig)

        reference_d80 = self.args_dict["psf_measurement_containment_mean"]
        rnda = self.rnda_start
        prev_error_d80 = float("inf")
        iteration = 0

        while True:
            mean_d80, sig_d80 = self.run_simulations_and_analysis(rnda)
            error_d80 = abs(1 - mean_d80 / reference_d80)
            collect_results(rnda, mean_d80, sig_d80)

            if error_d80 < relative_tolerance_d80:
                break

            if mean_d80 < reference_d80:
                rnda += step_size * self.rnda_start
            else:
                rnda -= step_size * self.rnda_start

            if error_d80 >= prev_error_d80:
                step_size = step_size / 2
            prev_error_d80 = error_d80
            iteration += 1
            if iteration > max_iteration:
                raise ValueError(
                    f"Maximum iterations ({max_iteration}) reached without convergence."
                )

        self.rnda_opt = rnda

    def _get_starting_value(self):
        """Get optimization starting value from command line or previous model."""
        if self.args_dict["rnda"] != 0:
            rnda_start = self.args_dict["rnda"]
        else:
            rnda_start = self.telescope_model.get_parameter_value("mirror_reflection_random_angle")[
                0
            ]

        self._logger.info(f"Start value for mirror_reflection_random_angle: {rnda_start} deg")
        return rnda_start

    def run_simulations_and_analysis(self, rnda, save_figures=False):
        """
        Run ray tracing simulations and analysis for one given value of rnda.

        Parameters
        ----------
        rnda: float
            Random reflection angle in degrees.
        save_figures: bool
            Save figures.

        Returns
        -------
        mean_d80: float
            Mean value of D80 in cm.
        sig_d80: float
            Standard deviation of D80 in cm.
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
        ray.simulate(test=self.args_dict["test"], force=True)  # force has to be True, always
        ray.analyze(force=True)
        if save_figures:
            ray.plot("d80_cm", save=True, d80=self.args_dict["psf_measurement_containment_mean"])

        return (
            ray.get_mean("d80_cm").to(u.cm).value,
            ray.get_std_dev("d80_cm").to(u.cm).value,
        )

    def print_results(self):
        """Print results to stdout."""
        containment_fraction_percent = int(self.args_dict["containment_fraction"] * 100)

        print(f"\nMeasured D{containment_fraction_percent}:")
        if self.args_dict["psf_measurement_containment_sigma"] is not None:
            print(
                f"Mean = {self.args_dict['psf_measurement_containment_mean']:.3f} cm, "
                f"StdDev = {self.args_dict['psf_measurement_containment_sigma']:.3f} cm"
            )
        else:
            print(f"Mean = {self.args_dict['psf_measurement_containment_mean']:.3f} cm")
        print(f"\nSimulated D{containment_fraction_percent}:")
        print(f"Mean = {self.mean_d80:.3f} cm, StdDev = {self.sig_d80:.3f} cm")
        print("\nmirror_random_reflection_angle")
        print(f"Previous value = {self.rnda_start:.6f}")
        print(f"New value = {self.rnda_opt:.6f}\n")

    def write_optimization_data(self):
        """
        Write optimization results to an astropy table (ecsv file).

        Used mostly for debugging of the optimization process.
        The first entry of the table is the best fit result.
        """
        containment_fraction_percent = int(self.args_dict["containment_fraction"] * 100)

        result_table = QTable(
            [
                [True] + [False] * len(self.results_rnda),
                [self.rnda_opt, *self.results_rnda] * u.deg,
                ([0.0] * (len(self.results_rnda) + 1)),
                ([0.0] * (len(self.results_rnda) + 1)) * u.deg,
                [self.mean_d80, *self.results_mean] * u.cm,
                [self.sig_d80, *self.results_sig] * u.cm,
            ],
            names=(
                "best_fit",
                "mirror_reflection_random_angle_sigma1",
                "mirror_reflection_random_angle_fraction2",
                "mirror_reflection_random_angle_sigma2",
                f"containment_radius_D{containment_fraction_percent}",
                f"containment_radius_sigma_D{containment_fraction_percent}",
            ),
        )
        writer.ModelDataWriter.dump(
            args_dict=self.args_dict,
            metadata=MetadataCollector(args_dict=self.args_dict),
            product_data=result_table,
        )
