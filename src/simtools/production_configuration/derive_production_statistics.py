"""
Calculate the event production statistics based on metrics.

Module for calculating the production event statistics based on statistical error metrics.
Contains the `ProductionStatisticsDerivator` class, which derives the number of events for
both the entire dataset and specific grid points. Event statistic is calculated using error
metrics and the evaluator's results.
"""

import astropy.units as u
import numpy as np


class ProductionStatisticsDerivator:
    """
    Derives the production statistics based on statistical error metrics.

    Supports deriving statistics for both the entire dataset and
    specific grid points like energy values.
    """

    def __init__(self, evaluator, metrics: dict):
        """
        Initialize the ProductionStatisticsDerivator with the evaluator and metrics.

        Parameters
        ----------
        evaluator : StatisticalUncertaintyEvaluator
            The evaluator responsible for calculating metrics and handling event data.
        metrics : dict
            Dictionary containing metrics, including target error for effective area.
        """
        self.evaluator = evaluator
        self.metrics = metrics

    def _compute_scaling_factor(self) -> np.ndarray:
        """
        Compute bin-wise scaling factors based on error metrics.

        Takes into account the energy range specified in the metrics and
        calculates a separate scaling factor for each energy bin.

        Returns
        -------
        np.ndarray
            Array of scaling factors for each energy bin.
        """
        uncertainty_effective_area = self.evaluator.metric_results.get("uncertainty_effective_area")
        relative_uncertainties = uncertainty_effective_area.get("relative_uncertainties")
        energy_range = (
            self.metrics.get("uncertainty_effective_area").get("energy_range").get("value")
        )
        energy_unit = u.Unit(
            self.metrics.get("uncertainty_effective_area").get("energy_range").get("unit")
        )

        energy_range_converted = np.array(energy_range) * energy_unit

        bin_edges = self.evaluator.energy_bin_edges
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Mask for bins within the metric specified energy range
        mask = (bin_centers >= energy_range_converted[0]) & (
            bin_centers <= energy_range_converted[1]
        )

        scaling_factors = np.zeros_like(relative_uncertainties)

        target_uncertainty = (
            self.metrics.get("uncertainty_effective_area").get("target_uncertainty").get("value")
        )

        # Calculate scaling factor only for bins within the energy range
        # For bins with zero events/uncertainty, use a scaling factor of 0
        valid_bins = mask & (relative_uncertainties > 0)
        scaling_factors[valid_bins] = (relative_uncertainties[valid_bins] / target_uncertainty) ** 2

        return scaling_factors

    def derive_statistics(self, return_sum: bool = True) -> u.Quantity:
        """
        Derive the production statistics based on statistical error metrics.

        Parameters
        ----------
        return_sum : bool, optional
            If True, returns the sum of production statistics for the entire set of MC events.
            If False, returns the production statistics for each grid point along the energy axis.
            Default is True.

        Returns
        -------
        u.Quantity
            If 'return_sum' is True, returns the total
            derived production statistics as a u.Quantity.
            If 'return_sum' is False, returns an array of production statistics along the energy
            axis as a u.Quantity.
        """
        scaling_factors = self._compute_scaling_factor()
        base_events = self._number_of_simulated_events()
        # currently we use the maximum of the scaling factors to scale the events. This is a soft
        # requirement if we want to keep the power law shape of the production statistics.
        scaled_events = base_events * np.max(scaling_factors)

        if return_sum:
            return np.sum(scaled_events)
        return scaled_events

    def _number_of_simulated_events(self) -> u.Quantity:
        """
        Fetch the number of simulated events from the evaluator's data.

        Returns
        -------
        u.Quantity
            The number of simulated events.
        """
        return self.evaluator.data.get("simulated_event_histogram")

    def calculate_production_statistics_at_grid_point(
        self,
        grid_point: tuple,
    ) -> u.Quantity:
        """
        Derive the production statistics for a specific energy grid point.

        Parameters
        ----------
        grid_point : tuple
            The grid point specifying energy, azimuth, zenith, NSB, and offset.

        Returns
        -------
        float
            The derived production statistics at the specified grid point (energy).
        """
        energy = grid_point[0]
        bin_edges = self.evaluator.create_bin_edges()
        bin_idx = np.digitize(energy, bin_edges) - 1

        # Get scaling factors for all bins
        scaling_factors = self._compute_scaling_factor()

        simulated_event_histogram = self.evaluator.data.get("simulated_event_histogram", [])

        if bin_idx < 0 or bin_idx >= len(simulated_event_histogram):
            raise ValueError(f"Energy {energy} is outside the range of the simulated events data.")

        base_events = self._number_of_simulated_events()
        base_event_at_energy = base_events[bin_idx]
        scaling_factor_at_energy = scaling_factors[bin_idx]

        return base_event_at_energy * scaling_factor_at_energy
