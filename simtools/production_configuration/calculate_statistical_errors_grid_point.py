"""
Provides functionality to evaluate statistical errors from FITS files.

Classes
-------
StatisticalErrorEvaluator
    Handles error calculation for given FITS files and specified metrics.

Example Usage
-------------
1. Instantiate Evaluators:
    Create instances of `StatisticalErrorEvaluator` for each combination of
     zenith angles and offsets.

   ```python
   import os
   import numpy as np
   from simtools.production_configuration.calculate_statistical_errors_grid_point import (
       StatisticalErrorEvaluator, InterpolationHandler
   )

   base_path = "/path/"
   zeniths = [20, 40, 60]
   offsets = [0, 1, 2, 3, 4, 5]  # Offsets in degrees

   evaluator_instances = []
   for zenith in zeniths:
       for offset in offsets:
           file_name = (
               f"prod5b-LaPalma-{zenith}deg-lin51-LL/"
               f"gamma_onSource.N.BL-4LSTs15MSTs-MSTN_ID0.eff-{offset}-CUT0.fits"
               if offset == 0
               else f"prod5b-LaPalma-{zenith}deg-lin51-LL/"
               f"gamma_cone.N.BL-4LSTs15MSTs-MSTN_ID0.eff-{offset}-CUT0.fits"
           )
           file_path = os.path.join(base_path, file_name)
           offset_value = 2 * offset
           evaluator_instances.append(
               StatisticalErrorEvaluator(
                   file_path,
                   file_type="On-source" if offset == 0 else "Gamma-cone",
                   metrics={
                       "error_eff_area": 0.02,
                       "error_sig_eff_gh": 0.02,
                       "error_energy_estimate_bdt_reg_tree": 0.05,
                       "error_gamma_ray_psf": 0.01,
                       "error_image_template_methods": 0.03,
                   },
                   grid_point=(1, 180, zenith, 0, offset_value),
               )
           )

2. Calculate Metrics:
    For each evaluator instance, calculate metrics and scaled events.

    ```python
    for evaluator in evaluator_instances:
        evaluator.calculate_metrics()
        evaluator.calculate_scaled_events()

3. Interpolate Results:
    Create an InterpolationHandler and interpolate results for a specified grid point.

    ```python
    interpolation_handler = InterpolationHandler(evaluator_instances)

    grid_point = (1, 180, 30, 0, 0)  # Example grid point (energy, azimuth, zenith, NSB, offset)
    query_points = np.array([grid_point])
    scaled_events = interpolation_handler.interpolate(query_points)
    print(f"Scaled events for grid point {grid_point}: {scaled_events}")
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.interpolate import griddata


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
        self.energy_threshold = None

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

        # Calculate uncertainties with binomial distribution,
        # apply threshold of 5 for triggered event counts data
        valid = (
            (simulated_event_counts > 0)
            & (triggered_event_counts <= simulated_event_counts)
            & (triggered_event_counts > 5)
        )
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

    def calculate_energy_threshold(self):
        """
        Calculate the energy threshold where the effective area exceeds 50% of its maximum value.

        Returns
        -------
        float
            Energy threshold value.
        """
        bin_edges = self.create_bin_edges()
        triggered_event_histogram = self.compute_histogram(self.data["event_energies"], bin_edges)
        simulated_event_histogram = self.data["simulated_event_histogram"]

        # Calculate efficiencies (effective area as a function of energy)
        efficiencies, _, _ = self.compute_efficiency_and_errors(
            triggered_event_histogram, simulated_event_histogram
        )

        # Determine the effective area threshold (50% of max effective area)
        max_efficiency = np.max(efficiencies)
        threshold_efficiency = 0.1 * max_efficiency

        # Find the first bin where efficiency exceeds the threshold
        threshold_index = np.argmax(efficiencies >= threshold_efficiency)
        if threshold_index == 0 and efficiencies[0] < threshold_efficiency:
            return  # No valid threshold found

        # Calculate the corresponding energy threshold
        self.energy_threshold = bin_edges[threshold_index]

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

    def calculate_scaled_events(self):
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
            raise Warning("Grid point data is not available for this evaluator.")

        bin_edges = self.create_bin_edges()
        simulated_event_histogram = self.data["simulated_event_histogram"]
        if not simulated_event_histogram.size:
            raise ValueError("Simulated event histogram is empty.")

        # TODO: Add here the implementation that uses a combination of the required metrics
        # Currently we only use the rel error on the eff area for scaling
        energy = self.grid_point[0]
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
    """Handle interpolation between multiple StatisticalErrorEvaluator instances."""

    def __init__(self, evaluators: list["StatisticalErrorEvaluator"]):
        self.evaluators = evaluators

        # Collect all grid points for each dimension (no uniqueness applied)
        self.azimuths = np.array([e.grid_point[1] for e in self.evaluators])
        self.zeniths = np.array([e.grid_point[2] for e in self.evaluators])
        self.nsbs = np.array([e.grid_point[3] for e in self.evaluators])
        self.offsets = np.array([e.grid_point[4] for e in self.evaluators])

        # Collect energy grids and corresponding scaled events
        self.energy_grids = [
            (e.data["bin_edges_low"][:-1] + e.data["bin_edges_high"][:-1]) / 2
            for e in self.evaluators
        ]
        self.scaled_events = [e.scaled_events for e in self.evaluators]
        self.energy_thresholds = np.array([e.energy_threshold for e in self.evaluators])

        # Prepare data for interpolation
        self.data, self.grid_points = self._build_data_array()

    def _build_data_array(self):
        """
        Build a data array with interpolated values across all dimensions including energy.

        Returns
        -------
        np.ndarray
            The data array with interpolated values.
        np.ndarray
            The corresponding grid points.
        """
        # Flatten the energy grid and other dimensions into a combined array
        flat_data_list = []
        flat_grid_points = []

        for e, energy_grid, scaled_events in zip(
            self.evaluators, self.energy_grids, self.scaled_events
        ):
            az = np.full(len(energy_grid), e.grid_point[1])
            zen = np.full(len(energy_grid), e.grid_point[2])
            nsb = np.full(len(energy_grid), e.grid_point[3])
            offset = np.full(len(energy_grid), e.grid_point[4])

            # Combine grid points and data
            grid_points = np.column_stack([energy_grid, az, zen, nsb, offset])
            flat_grid_points.append(grid_points)
            flat_data_list.append(scaled_events)

        # Flatten the list and convert to numpy arrays
        flat_grid_points = np.vstack(flat_grid_points)
        flat_data = np.hstack(flat_data_list)

        # Sort the grid points and corresponding data by energy
        sorted_indices = np.argsort(flat_grid_points[:, 0])
        sorted_grid_points = flat_grid_points[sorted_indices]
        sorted_data = flat_data[sorted_indices]

        return sorted_data, sorted_grid_points

    def _remove_flat_dimensions(self, grid_points):
        """Identify and remove flat dimensions (dimensions with no variance)."""
        variance = np.var(grid_points, axis=0)
        non_flat_mask = variance > 1e-6  # Threshold for determining flatness
        reduced_grid_points = grid_points[:, non_flat_mask]
        return reduced_grid_points, non_flat_mask

    def interpolate(self, query_points: np.ndarray) -> np.ndarray:
        """
        Interpolate the number of simulated events given query points.

        Parameters
        ----------
        query_points : np.ndarray
            Array of query points with shape (n, 5), where n is the number of points,
            and 5 represents (energy, azimuth, zenith, nsb, offset).

        Returns
        -------
        np.ndarray
            Interpolated values at the query points.
        """
        # Remove flat dimensions from the normalized grid and query points
        reduced_grid_points, non_flat_mask = self._remove_flat_dimensions(self.grid_points)
        reduced_query_points = query_points[:, non_flat_mask]

        # Interpolate using the reduced dimensions
        return griddata(
            reduced_grid_points,
            self.data,
            reduced_query_points,
            method="linear",
            fill_value=np.nan,
            rescale=True,
        )

    def interpolate_energy_threshold(self, query_point: np.ndarray) -> float:
        """
        Interpolate the energy threshold for a given grid point.

        Parameters
        ----------
        query_point : np.ndarray
            Array specifying the grid point (energy, azimuth, zenith, NSB, offset).

        Returns
        -------
        float
            Interpolated energy threshold.
        """
        flat_grid_points = []
        flat_energy_thresholds = []

        for e in self.evaluators:
            az = e.grid_point[1]
            zen = e.grid_point[2]
            nsb = e.grid_point[3]
            offset = e.grid_point[4]
            grid_point = np.array([az, zen, nsb, offset])
            flat_grid_points.append(grid_point)
            flat_energy_thresholds.append(e.energy_threshold)

        flat_grid_points = np.array(flat_grid_points)
        flat_energy_thresholds = np.array(flat_energy_thresholds)

        reduced_grid_points, non_flat_mask = self._remove_flat_dimensions(flat_grid_points)
        full_non_flat_mask = np.concatenate(([False], non_flat_mask))
        reduced_query_point = query_point[0][full_non_flat_mask]

        interpolated_threshold = griddata(
            reduced_grid_points,
            flat_energy_thresholds,
            reduced_query_point,
            method="linear",
            fill_value=np.nan,
            rescale=False,
        )

        return interpolated_threshold.item()

    def plot_comparison(self, evaluator: "StatisticalErrorEvaluator"):
        """
        Plot a comparison between the simulated, scaled, and triggered events.

        Parameters
        ----------
        evaluator : StatisticalErrorEvaluator
            The evaluator for which to plot the comparison.
        """
        midpoints = 0.5 * (evaluator.data["bin_edges_high"] + evaluator.data["bin_edges_low"])

        self.grid_points = np.column_stack(
            [
                midpoints,
                np.full_like(midpoints, evaluator.grid_point[1]),
                np.full_like(midpoints, evaluator.grid_point[2]),
                np.full_like(midpoints, evaluator.grid_point[3]),
                np.full_like(midpoints, evaluator.grid_point[4]),
            ]
        )

        self.interpolate(self.grid_points)

        plt.plot(midpoints, evaluator.scaled_events, label="Scaled")

        triggered_event_histogram, _ = np.histogram(
            evaluator.data["event_energies"], bins=evaluator.data["bin_edges_low"]
        )
        plt.plot(midpoints[:-1], triggered_event_histogram, label="Triggered")

        plt.legend()
        plt.xscale("log")
        plt.xlabel("Energy (Midpoint of Bin Edges)")
        plt.ylabel("Event Count")
        plt.title("Comparison of Simulated, Scaled, and Triggered Events")
        plt.show()
