"""
Provides functionality to evaluate statistical errors from FITS files.

Classes
-------
StatisticalErrorEvaluator: Handles error calculation for given FITS files and metrics.
"""

import logging
import os

import numpy as np
from astropy.io import fits


class StatisticalErrorEvaluator:
    """
    Evaluates statistical errors from a FITS file.

    Parameters
    ----------
    file_path : str
        Path to the FITS file.
    file_type : str
        Type of the file, either 'On-source' or 'Offset'.
    metrics : dict, optional
        Dictionary of metrics to evaluate. Default is None.
    """

    def __init__(self, file_path: str, file_type: str, metrics: dict[str, float] | None = None):
        """
        Initialize the evaluator with a specific FITS file, its type, and metrics to calculate.

        Parameters
        ----------
        file_path : str
            The path to the FITS file.
        file_type : str
            The type of the file ('On-source' or 'Offset').
        metrics : dict, optional
            Dictionary specifying which metrics to compute and their reference values.
        """
        self.file_path = file_path
        self.file_type = file_type
        self.metrics = metrics or {}
        self.data = self.load_data_from_file()
        self.error_eff_area = None
        self.error_sig_eff_gh = None
        self.error_energy_estimate_bdt_reg_tree = None
        self.sigma_energy = None
        self.delta_energy = None
        self.error_gamma_ray_psf = None
        self.error_image_template_methods = None
        self.metric_results = None

    def load_data_from_file(self):
        """
        Load data from the FITS file and return relevant arrays.

        Returns
        -------
        dict
            Dictionary containing arrays from the FITS file.
        """
        data = {}
        try:
            with fits.open(self.file_path) as hdul:
                events_data = hdul["EVENTS"].data  # pylint: disable=E1101
                event_energies = events_data["ENERGY"]
                mc_energies = events_data["MC_ENERGY"]

                sim_events_data = hdul["SIMULATED EVENTS"].data  # pylint: disable=E1101
                bin_edges_low = sim_events_data["MC_ENERG_LO"]
                bin_edges_high = sim_events_data["MC_ENERG_HI"]
                simulated_event_histogram = sim_events_data["EVENTS"]

                data = {
                    "event_energies": event_energies,
                    "mc_energies": mc_energies,
                    "bin_edges_low": bin_edges_low,
                    "bin_edges_high": bin_edges_high,
                    "simulated_event_histogram": simulated_event_histogram,
                }
        except (FileNotFoundError, KeyError) as e:
            logging.error(f"Error loading file {self.file_path}: {e}")
        return data

    def create_bin_edges(self):
        """
        Create unique energy bin edges.

        Parameters
        ----------
        bin_edges_low : array
            Array of lower bin edges.
        bin_edges_high : array
            Array of upper bin edges.

        Returns
        -------
        bin_edges : array
            Array of unique energy bin edges.
        """
        bin_edges_low = self.data["bin_edges_low"]
        bin_edges_high = self.data["bin_edges_high"]
        bin_edges = np.concatenate([bin_edges_low, [bin_edges_high[-1]]])
        return np.unique(bin_edges)

    def compute_histogram(self, event_energies, bin_edges):
        """
        Compute histogram for triggered events.

        Parameters
        ----------
        event_energies : array
            Array of energies of the observed events.
        bin_edges : array
            Array of energy bin edges.

        Returns
        -------
        triggered_event_histogram : array
            Histogram of triggered events.
        """
        triggered_event_histogram, _ = np.histogram(event_energies, bins=bin_edges)
        return triggered_event_histogram

    def compute_efficiency_and_errors(self, triggered_event_counts, simulated_event_counts):
        """
        Compute energy efficiency and its statistical error using the binomial distribution.

        Parameters
        ----------
        triggered_event_counts : array
            Histogram counts of the triggered events.
        simulated_event_counts : array
            Histogram counts of the simulated events.

        Returns
        -------
        efficiencies : array
            Array of calculated efficiencies.
        uncertainties : array
            Array of calculated uncertainties.
        relative_errors : array
            Array of relative uncertainties.
        """
        efficiencies = np.divide(
            triggered_event_counts,
            simulated_event_counts,
            out=np.zeros_like(triggered_event_counts, dtype=float),
            where=simulated_event_counts > 0,
        )

        # Calculate uncertainties with binomial distribution
        valid = (simulated_event_counts > 0) & (triggered_event_counts <= simulated_event_counts)
        uncertainties = np.zeros_like(triggered_event_counts, dtype=float)

        if np.any(valid):
            uncertainties[valid] = np.sqrt(
                np.maximum(
                    triggered_event_counts[valid]
                    * (1 - triggered_event_counts[valid] / simulated_event_counts[valid]),
                    0,
                )
            )

        # Compute relative errors
        relative_errors = np.divide(
            uncertainties,
            simulated_event_counts,
            out=np.zeros_like(uncertainties, dtype=float),
            where=uncertainties > 0,
        )

        return efficiencies, uncertainties, relative_errors

    def calculate_error_eff_area(self):
        """
        Calculate the uncertainties on the effective collection area.

        Returns
        -------
        errors : dict
            Dictionary with uncertainties for the file.
        """
        bin_edges = self.create_bin_edges()
        triggered_event_histogram = self.compute_histogram(self.data["event_energies"], bin_edges)
        simulated_event_histogram = self.data["simulated_event_histogram"]
        _, _, relative_errors = self.compute_efficiency_and_errors(
            triggered_event_histogram, simulated_event_histogram
        )
        return {"relative_errors": relative_errors}

    def calculate_error_sig_eff_gh(self):
        """
        Calculate the uncertainties on signal efficiency in gamma-hadron separation.

        Returns
        -------
        float
            The calculated uncertainty for signal efficiency.
        """
        return 0.02  # placeholder

    def calculate_error_energy_estimate_bdt_reg_tree(self):
        """
        Calculate the uncertainties in energy estimation.

        Returns
        -------
        float
            The calculated uncertainty for energy estimation.
        """
        logging.info("Calculating Energy Resolution Error")

        event_energies = np.array(self.data["event_energies"])
        mc_energies = np.array(self.data["mc_energies"])

        if len(event_energies) != len(mc_energies):
            raise ValueError(f"Mismatch in the number of energies for file {self.file_path}")

        # Calculate energy deviations
        energy_deviation = (event_energies - mc_energies) / mc_energies

        # Bin the energy deviations
        bin_edges = self.create_bin_edges()
        bin_indices = np.digitize(event_energies, bin_edges) - 1

        # Group deviations by bin
        energy_deviation_by_bin = [
            energy_deviation[bin_indices == i] for i in range(len(bin_edges) - 1)
        ]

        # Calculate sigma (standard deviation) for each bin
        sigma_energy = [np.std(d) if len(d) > 0 else np.nan for d in energy_deviation_by_bin]

        # Calculate delta_energy as the mean deviation for each bin
        delta_energy = [np.mean(d) if len(d) > 0 else np.nan for d in energy_deviation_by_bin]

        # Combine sigma into a single measure if needed
        overall_uncertainty = np.nanmean(sigma_energy)

        return overall_uncertainty, sigma_energy, delta_energy

    def calculate_error_gamma_ray_psf(self):
        """
        Calculate the uncertainties on gamma-ray PSF (68, 80, and 95% containment radius).

        Returns
        -------
        float
            The calculated uncertainty for gamma-ray PSF.
        """
        return 0.01  # Placeholder for the actual calculation

    def calculate_error_image_template_methods(self):
        """
        Calculate the uncertainties relevant for image template methods.

        Returns
        -------
        float
            The calculated uncertainty for image template methods.
        """
        return 0.05  # Placeholder for the actual calculation

    def calculate_metrics(self):
        """Calculate all defined metrics as specified in self.metrics and store results."""
        if "error_eff_area" in self.metrics:
            self.error_eff_area = self.calculate_error_eff_area()
            ref_value = self.metrics.get("error_eff_area")
            if self.error_eff_area:
                avg_error = np.mean(self.error_eff_area["relative_errors"])
                print(f"Effective Area Error (avg): {avg_error:.3f}, Reference: {ref_value:.3f}")

        if "error_sig_eff_gh" in self.metrics:
            self.error_sig_eff_gh = self.calculate_error_sig_eff_gh()
            ref_value = self.metrics.get("error_sig_eff_gh")
            print(
                f"Signal Efficiency Error: {self.error_sig_eff_gh:.3f}, Reference: {ref_value:.3f}"
            )

        if "error_energy_estimate_bdt_reg_tree" in self.metrics:
            self.error_energy_estimate_bdt_reg_tree, self.sigma_energy, self.delta_energy = (
                self.calculate_error_energy_estimate_bdt_reg_tree()
            )
            ref_value = self.metrics.get("error_energy_estimate_bdt_reg_tree")
            print(
                f"Energy Estimate Error: {self.error_energy_estimate_bdt_reg_tree:.3f},"
                f"Reference: {ref_value:.3f}"
            )

        if "error_gamma_ray_psf" in self.metrics:
            self.error_gamma_ray_psf = self.calculate_error_gamma_ray_psf()
            ref_value = self.metrics.get("error_gamma_ray_psf")
            print(
                f"Gamma-Ray PSF Error: {self.error_gamma_ray_psf:.3f}, Reference: {ref_value:.3f}"
            )

        if "error_image_template_methods" in self.metrics:
            self.error_image_template_methods = self.calculate_error_image_template_methods()
            ref_value = self.metrics.get("error_image_template_methods")
            print(
                f"Image Template Methods Error: {self.error_image_template_methods:.3f},"
                f"Reference: {ref_value:.3f}"
            )

        self.metric_results = {
            "error_eff_area": self.error_eff_area,
            "error_sig_eff_gh": self.error_sig_eff_gh,
            "error_energy_estimate_bdt_reg_tree": self.error_energy_estimate_bdt_reg_tree,
            "error_gamma_ray_psf": self.error_gamma_ray_psf,
            "error_image_template_methods": self.error_image_template_methods,
        }

    def calculate_max_error_for_effective_area(self):
        """
        Calculate the maximum relative error for effective area.

        Returns
        -------
        max_error : float
            Maximum relative error.
        """
        if self.error_eff_area:
            return np.max(self.error_eff_area["relative_errors"])
        return None

    def calculate_overall_metric(self, metric="average"):
        """
        Calculate an overall metric for the systematic errors.

        Parameters
        ----------
        metric : str
            The metric to calculate ('average', 'maximum').

        Returns
        -------
        dict
            Dictionary with overall maximum errors for each metric.
        """
        if self.metric_results is None:
            raise ValueError("Metrics have not been computed yet.")

        overall_max_errors = {}

        for metric_name, result in self.metric_results.items():
            if metric_name == "error_eff_area":
                max_errors = self.calculate_max_error_for_effective_area()
                overall_max_errors[metric_name] = max(max_errors.values()) if max_errors else 0
            elif metric_name in [
                "error_sig_eff_gh",
                "error_energy_estimate_bdt_reg_tree",
                "error_gamma_ray_psf",
                "error_image_template_methods",
            ]:
                overall_max_errors[metric_name] = result
            else:
                raise ValueError(f"Unsupported result type for {metric_name}: {type(result)}")

        # Compute the overall maximum error
        all_max_errors = list(overall_max_errors.values())

        if metric == "average":
            overall_metric = np.mean(all_max_errors)
        elif metric == "maximum":
            overall_metric = np.max(all_max_errors)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        overall_max_errors["overall_max"] = overall_metric

        return overall_max_errors


def main():
    """Calculate specific uncertainties for fits files."""
    base_path = "/Users/znb68/PD/CTA/"

    # Instantiate the SystematicErrorEvaluator class for the On-source file
    on_source_file = os.path.join(
        base_path, "gamma_onSource.N.BL-4LSTs15MSTs-MSTN_ID0.eff-0-CUT0.fits"
    )
    on_source_evaluator = StatisticalErrorEvaluator(
        on_source_file,
        "On-source",
        metrics={
            "error_eff_area": 0.1,
            "error_sig_eff_gh": 0.02,
            "error_energy_estimate_bdt_reg_tree": 0.05,
            "error_gamma_ray_psf": 0.01,
            "error_image_template_methods": 0.03,
        },
    )

    # Calculate metrics for On-source file
    on_source_evaluator.calculate_metrics()

    # Output the metrics for On-source file
    print("Metrics for On-source file:")
    for metric_name, value in on_source_evaluator.metric_results.items():
        print(f"{metric_name}: {value}")

    # Calculate and print overall metric for On-source file
    overall_metric_on_source = on_source_evaluator.calculate_overall_metric(metric="maximum")
    print("Overall Metric for On-source (Maximum):", overall_metric_on_source)

    # Instantiate StatisticalErrorEvaluator class for Offset files
    offset_files = [
        os.path.join(base_path, f"gamma_cone.N.BL-4LSTs15MSTs-MSTN_ID0.eff-{i}-CUT0.fits")
        for i in range(6)
    ]
    offset_evaluators = [
        StatisticalErrorEvaluator(
            file,
            "Offset",
            metrics={
                "error_eff_area": 0.1,
                "error_sig_eff_gh": 0.02,
                "error_energy_estimate_bdt_reg_tree": 0.05,
                "error_gamma_ray_psf": 0.01,
                "error_image_template_methods": 0.03,
            },
        )
        for file in offset_files
    ]

    # Calculate and print metrics for each Offset file
    for i, evaluator in enumerate(offset_evaluators):
        evaluator.calculate_metrics()
        print(f"\nMetrics for Offset file {i}:")
        for metric_name, value in evaluator.metric_results.items():
            print(f"{metric_name}: {value}")

        # Calculate and print overall metric for each Offset file
        overall_metric_offset = evaluator.calculate_overall_metric(metric="maximum")
        print(f"Overall Metric for Offset file {i} (Maximum):", overall_metric_offset)


if __name__ == "__main__":
    main()
