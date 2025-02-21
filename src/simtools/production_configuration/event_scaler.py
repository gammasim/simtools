"""
Calculate the scaled number of events based on metrics.

Module for scaling events based on statistical error metrics. Contains the `EventScaler` class,
which scales the number of events for both the entire dataset and specific grid points.
Scaling factors are calculated using error metrics and the evaluator's results.
"""

import astropy.units as u
import numpy as np

__all__ = ["EventScaler"]


class EventScaler:
    """
    Scales the number of events based on statistical error metrics.

    Supports scaling both the entire dataset and specific grid points like energy values.
    """

    def __init__(self, evaluator, metrics: dict):
        """
        Initialize the EventScaler with the evaluator and metrics.

        Parameters
        ----------
        evaluator : StatisticalErrorEvaluator
            The evaluator responsible for calculating metrics and handling event data.
        metrics : dict
            Dictionary containing metrics, including target error for effective area.
        """
        self.evaluator = evaluator
        self.metrics = metrics

    def scale_events(self, return_sum: bool = True) -> u.Quantity:
        """
        Calculate the scaled number of events based on statistical error metrics.

        Parameters
        ----------
        return_sum : bool, optional
            If True, returns the sum of scaled events for the entire set of MC events. If False,
            returns the scaled events for each grid point along the energy axis. Default is True.

        Returns
        -------
        u.Quantity
            If 'return_sum' is True, returns the total scaled number of events as a u.Quantity.
            If 'return_sum' is False, returns an array of scaled events along the energy axis as
            a u.Quantity.
        """
        scaling_factor = self._compute_scaling_factor()

        base_events = self._number_of_simulated_events()

        if return_sum:
            return np.sum(base_events * scaling_factor)
        return base_events * scaling_factor

    def _compute_scaling_factor(self) -> float:
        """
        Compute the scaling factor based on the error metrics.

        Returns
        -------
        float
            The scaling factor.
        """
        metric_results = self.evaluator.calculate_metrics()
        uncertainty_effective_area = metric_results.get("uncertainty_effective_area", {})
        current_max_error = uncertainty_effective_area.get("max_error")
        target_max_error = self.metrics.get("uncertainty_effective_area", {}).get("target_error")[
            "value"
        ]

        return (current_max_error / target_max_error) ** 2

    def _number_of_simulated_events(self) -> u.Quantity:
        """
        Fetch the number of simulated events from the evaluator's data.

        Returns
        -------
        u.Quantity
            The number of simulated events.
        """
        return self.evaluator.data.get("simulated_event_histogram")

    def calculate_scaled_events_at_grid_point(
        self,
        grid_point: tuple,
    ) -> u.Quantity:
        """
        Calculate the scaled number of events for a specific energy grid point.

        Parameters
        ----------
        grid_point : tuple
            The grid point specifying energy, azimuth, zenith, NSB, and offset.

        Returns
        -------
        float
            The scaled number of events at the specified grid point (energy).
        """
        energy = grid_point[0]
        bin_edges = self.evaluator.create_bin_edges()
        bin_idx = np.digitize(energy, bin_edges) - 1

        scaling_factor = self._compute_scaling_factor()

        simulated_event_histogram = self.evaluator.data.get("simulated_event_histogram", [])

        if bin_idx < 0 or bin_idx >= len(simulated_event_histogram):
            raise ValueError(f"Energy {energy} is outside the range of the simulated events data.")

        base_events = self._number_of_simulated_events()
        base_event_at_energy = base_events[bin_idx]
        return base_event_at_energy * scaling_factor
