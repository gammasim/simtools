"""
Provides functionality to evaluate statistical errors from FITS files.

Classes
-------
StatisticalErrorEvaluator: Handles error calculation for given FITS files and metrics.
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.interpolate import RegularGridInterpolator


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
    grid_point : tuple, optional
        Tuple specifying the grid point (energy, azimuth, zenith, NSB, offset).
    """

    def __init__(
        self,
        file_path: str,
        file_type: str,
        metrics: dict[str, float] | None = None,
        grid_point: tuple[float, float, float, float, float] | None = None,
    ):
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
        grid_point : tuple, optional
            Tuple specifying the grid point (energy, azimuth, zenith, NSB, offset).
        """
        self.file_path = file_path
        self.file_type = file_type
        self.metrics = metrics or {}
        self.grid_point = grid_point

        self.data = self.load_data_from_file()

        self.error_eff_area = None
        self.error_sig_eff_gh = None
        self.error_energy_estimate_bdt_reg_tree = None
        self.sigma_energy = None
        self.delta_energy = None
        self.error_gamma_ray_psf = None
        self.error_image_template_methods = None
        self.metric_results = None
        self.scaled_events = None
        self.scaled_events_gridpoint = None

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
                if self.grid_point is None:
                    unique_azimuths = np.unique(events_data["PNT_AZ"])
                    unique_zeniths = np.unique(events_data["PNT_ALT"])
                    if len(unique_azimuths) == 1 and len(unique_zeniths) == 1:
                        self.grid_point = (
                            0,
                            unique_azimuths[0],
                            unique_zeniths[0],
                            0,
                            0,
                        )  # Init gridpoint with az and zenith here
                        # grid point(energy, azimuth, zenith, NSB, offset)

        except (FileNotFoundError, KeyError) as e:
            logging.error(f"Error loading file {self.file_path}: {e}")
        return data

    def create_bin_edges(self):
        """
        Create unique energy bin edges.

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
        # TODO: implement
        return 0.02  # Placeholder value

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
        # TODO: implement
        return 0.01  # Placeholder value

    def calculate_error_image_template_methods(self):
        """
        Calculate the uncertainties relevant for image template methods.

        Returns
        -------
        float
            The calculated uncertainty for image template methods.
        """
        # TODO: implement
        return 0.05  # Placeholder value

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
                f"Energy Estimate Error: {self.error_energy_estimate_bdt_reg_tree:.3f}, "
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
                f"Image Template Methods Error: {self.error_image_template_methods:.3f}, "
                f"Reference: {ref_value:.3f}"
            )

        # Collect all results in a dictionary
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

    def calculate_scaled_events(
        self, grid_point: tuple[float, float, float, float, float] | None = None
    ) -> float:
        """
        Calculate the scaled number of events for a specific grid point.

        Parameters
        ----------
        grid_point : tuple, optional
            Tuple specifying the grid point (energy, azimuth, zenith, NSB, offset).

        Returns
        -------
        float
            Scaled number of events for the specified grid point.
        """
        if not self.grid_point:
            raise ValueError("Grid point data is not available for this evaluator.")

        bin_edges = self.create_bin_edges()
        simulated_event_histogram = self.data["simulated_event_histogram"]
        if not simulated_event_histogram.size:
            raise ValueError("Simulated event histogram is empty.")

        # TODO: Add here the implementation that uses a combination of the required metrics
        # Currently we only use the rel error on the eff area for scaling
        energy = grid_point[0]
        bin_idx = np.digitize(energy, bin_edges) - 1
        if bin_idx < 0 or bin_idx >= len(simulated_event_histogram):
            raise ValueError("Grid point is outside the range of the current file's data.")
        self.scaled_events = (
            self.data["simulated_event_histogram"]
            * self.error_eff_area["relative_errors"]
            / self.metrics["error_eff_area"]
        )
        self.scaled_events_gridpoint = self.scaled_events[bin_idx]
        return self.scaled_events_gridpoint

    def calculate_overall_metric(self, metric="average"):
        """
        Calculate an overall metric for the statistical errors.

        Parameters
        ----------
        metric : str
            The metric to calculate ('average', 'maximum').

        Returns
        -------
        dict
            Dictionary with overall maximum errors for each metric.
        """
        # TODO: Decide how to combine the metrics
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

        all_max_errors = list(overall_max_errors.values())
        if metric == "average":
            overall_metric = np.mean(all_max_errors)
        elif metric == "maximum":
            overall_metric = np.max(all_max_errors)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        overall_max_errors["overall_max"] = overall_metric

        return overall_max_errors


class InterpolationHandler:
    """
    Handle interpolation between multiple StatisticalErrorEvaluator instances.

    Parameters
    ----------
    evaluators : list
        List of StatisticalErrorEvaluator instances.
    """

    def __init__(self, evaluators: list[StatisticalErrorEvaluator]):
        self.evaluators = evaluators

        # Gather all grid points and bin edges from the evaluators, filtering out None values
        self.azimuths = np.unique(
            [e.grid_point[1] for e in self.evaluators if e.grid_point[1] is not None]
        )
        self.zeniths = np.unique(
            [e.grid_point[2] for e in self.evaluators if e.grid_point[2] is not None]
        )
        self.nsbs = np.unique(
            [e.grid_point[3] for e in self.evaluators if e.grid_point[3] is not None]
        )
        self.offsets = np.unique(
            [e.grid_point[4] for e in self.evaluators if e.grid_point[4] is not None]
        )

        self.energies, self.data = self._handle_energy_bin_edges()

        # Init the grid
        self.grid = (self.energies, self.azimuths, self.zeniths, self.nsbs, self.offsets)

        # Create the interpolator function
        self.interpolator = RegularGridInterpolator(
            self.grid, self.data, bounds_error=False, fill_value=None
        )

    def _handle_energy_bin_edges(self):
        """
        Handle energy bin edges.

        Returns
        -------
        energies : np.array
            The final energy grid.
        data : np.array
            The filled data array.
        """
        # Collect bin edges from each evaluator
        bin_edges_high = [e.data["bin_edges_high"] for e in self.evaluators]
        bin_edges_low = [e.data["bin_edges_low"] for e in self.evaluators]

        # Calculate midpoints for each evaluator
        midpoints = [0.5 * (high + low) for high, low in zip(bin_edges_high, bin_edges_low)]

        # Check if midpoints are the same across all evaluators
        if all(np.array_equal(midpoints[0], m) for m in midpoints):
            # If midpoints are the same, use the first evaluator's energy grid
            energies = np.unique(np.concatenate(midpoints))
            data = self._fill_data_array(midpoints[0], energies)
        else:
            # If midpoints differ, perform complex interpolation
            energies, data = self.complex_interpolation(midpoints, bin_edges_high)

        return energies, data

    def _fill_data_array(self, midpoints, energies):
        """
        Fill the data array based on the provided midpoints and energies.

        Returns
        -------
        np.array
            The filled data array.
        """
        data = np.zeros(
            (
                len(energies),
                len(self.azimuths),
                len(self.zeniths),
                len(self.nsbs),
                len(self.offsets),
            )
        )

        for e in self.evaluators:
            energy_indices = np.digitize(midpoints, energies) - 1
            azimuth_index = np.searchsorted(self.azimuths, e.grid_point[1])
            zenith_index = np.searchsorted(self.zeniths, e.grid_point[2])
            nsb_index = np.searchsorted(self.nsbs, e.grid_point[3])
            offset_index = np.searchsorted(self.offsets, e.grid_point[4])

            for i, energy_index in enumerate(energy_indices[:-1]):
                data[energy_index, azimuth_index, zenith_index, nsb_index, offset_index] = (
                    e.scaled_events[i]
                )

        return data

    def complex_interpolation(self, midpoints, bin_edges_high):
        """
        Perform complex interpolation between different energy grids.

        Returns
        -------
        np.array
            The combined energy grid.
        np.array
            The filled data array with interpolated values.
        """
        # Combine all bin edges and sort them to create a unified energy grid
        # Probably not the best solution.
        combined_edges = np.unique(np.concatenate(bin_edges_high))
        energies = np.unique(combined_edges)

        data = np.zeros(
            (
                len(energies),
                len(self.azimuths),
                len(self.zeniths),
                len(self.nsbs),
                len(self.offsets),
            )
        )

        for e, midpoint, _ in zip(self.evaluators, midpoints, bin_edges_high):
            energy_indices = np.digitize(midpoint, energies) - 1
            azimuth_index = np.searchsorted(self.azimuths, e.grid_point[1])
            zenith_index = np.searchsorted(self.zeniths, e.grid_point[2])
            nsb_index = np.searchsorted(self.nsbs, e.grid_point[3])
            offset_index = np.searchsorted(self.offsets, e.grid_point[4])

            for i, energy_index in enumerate(energy_indices[:-1]):
                data[energy_index, azimuth_index, zenith_index, nsb_index, offset_index] = (
                    e.scaled_events[i]
                )

        return energies, data

    def interpolate_simulated_events(
        self, grid_point: tuple[float | list[float], float, float, float, float]
    ) -> list[float]:
        """
        Interpolate the number of simulated events. Handles lists for grid point elements.

        Parameters
        ----------
        grid_point : tuple
            Tuple specifying the grid point (energy, azimuth, zenith, NSB, offset).

        Returns
        -------
        list[float]
            Scaled number of events for each energy in the specified grid point.
        """
        # Expand grid points to handle lists
        energies, azimuth, zenith, nsb, offset = grid_point

        if isinstance(energies, list):
            results = []
            for energy in energies:
                single_point = (energy, azimuth, zenith, nsb, offset)
                result = self.interpolator(single_point)
                results.append(result)
            return results

        return [self.interpolator(grid_point)]

    def plot_comparison(self, evaluator: StatisticalErrorEvaluator):
        """
        Plot a comparison.

        Plot the simulated events, scaled events, and triggered events.

        Parameters
        ----------
        evaluator : StatisticalErrorEvaluator
            The evaluator for which to plot the comparison.
        """
        midpoints = 0.5 * (evaluator.data["bin_edges_high"] + evaluator.data["bin_edges_low"])
        midpoints = midpoints[:-1]
        print(len(midpoints))
        # interpolate for all bin centers
        grid_point = evaluator.grid_point
        scaled_events = self.interpolate_simulated_events(
            (midpoints, grid_point[1], grid_point[2], grid_point[3], grid_point[4])
        )

        # Plot the simulated events
        plt.plot(midpoints, evaluator.data["simulated_event_histogram"][:-1], label="Simulated")

        # Plot the scaled events
        plt.plot(midpoints, scaled_events, label="Scaled")

        # Plot the triggered events
        triggered_event_histogram, _ = np.histogram(
            evaluator.data["event_energies"], bins=evaluator.data["bin_edges_low"]
        )
        plt.plot(midpoints, triggered_event_histogram, label="Triggered")

        plt.legend()
        plt.yscale("log")
        plt.xscale("log")
        plt.xlabel("Energy (Midpoint of Bin Edges)")
        plt.ylabel("Event Count")
        plt.title("Comparison of Simulated, Scaled, and Triggered Events")
        plt.show()


def main():
    """Calculate specific uncertainties for fits files."""
    # Instantiate StatisticalErrorEvaluator for each file
    base_path = "/Users/znb68/PD/CTA/"
    files = [
        os.path.join(
            base_path,
            f"prod5b-LaPalma-{zenith}deg-lin51-LL/"
            f"gamma_cone.N.BL-4LSTs15MSTs-MSTN_ID0.eff-0-CUT0.fits",
        )
        for zenith in [20, 40, 60]
    ]
    evaluator_instances = [
        StatisticalErrorEvaluator(
            file_path,
            file_type="On-source",
            metrics={
                "error_eff_area": 0.01,
                "error_sig_eff_gh": 0.02,
                "error_energy_estimate_bdt_reg_tree": 0.05,
                "error_gamma_ray_psf": 0.01,
                "error_image_template_methods": 0.03,
            },
            grid_point=(1, 180, zenith, 1, 0),  # Example grid point
        )
        for file_path, zenith in zip(files, [20, 40, 60])
    ]

    # Calculate metrics for each evaluator
    for evaluator in evaluator_instances:
        evaluator.calculate_metrics()
        evaluator.calculate_scaled_events()

    # Create InterpolationHandler and interpolate
    interpolation_handler = InterpolationHandler(evaluator_instances)
    grid_point = ([1, 2, 10], 180, 34, 0, 0.0)  # Example grid point
    # (energy, azimuth, zenith, NSB, offset)
    scaled_events = interpolation_handler.interpolate_simulated_events(grid_point)
    print(f"Scaled events for grid point {grid_point}: {scaled_events}")


if __name__ == "__main__":
    main()
