"""
Module defines the EventScaler class.

Module for scaling events based on statistical error metrics. Contains the `EventScaler` class,
which scales the number of events for both the entire dataset and specific grid points.
Scaling factors are calculated using error metrics and the evaluator's results.
"""

import logging

import numpy as np

_logger = logging.getLogger(__name__)


class EventScaler:
    """
    Scales the number of events based on statistical error metrics.

    Supports scaling both the entire dataset and specific grid points like energy values.
    """

    def __init__(self, evaluator, science_case, metrics):
        """
        Initialize the EventScaler with the evaluator, science case, and metrics.

        Parameters
        ----------
        evaluator : object
            The evaluator responsible for calculating metrics and handling event data.
        science_case : str
            The science case used to adjust the uncertainty factor.
        metrics : dict
            Dictionary containing metrics, including target error for effective area.
        """
        self.evaluator = evaluator
        self.science_case = science_case
        self.metrics = metrics

    def scale_events(self) -> float:
        """
        Calculate the scaled number of events based on statistical error metrics.

        Returns
        -------
        float
            The scaled number of events for the entire dataset.
        """
        scaling_factor = self._compute_scaling_factor()
        base_events = self._number_of_simulated_events()
        return np.sum(base_events * scaling_factor)

    def scale_events_at_grid_point(self, grid_point: tuple) -> float:
        """
        Calculate the scaled number of events for a specific grid point (energy).

        Parameters
        ----------
        grid_point : tuple
            The grid point specifying energy, azimuth, zenith, NSB, and offset.

        Returns
        -------
        float
            The scaled number of events for the specified grid point (energy).
        """
        scaling_factor = self._compute_scaling_factor()
        return self._calculate_scaled_events_at_grid_point(grid_point, scaling_factor)

    def _compute_scaling_factor(self) -> float:
        """
        Compute the scaling factor based on the error metrics.

        Returns
        -------
        float
            The scaling factor.
        """
        metric_results = self.evaluator.calculate_metrics()
        error_eff_area = metric_results.get("error_eff_area", {})
        current_max_error = error_eff_area.get("max_error")
        target_max_error = self.metrics.get("error_eff_area", {}).get("target_error")["value"]

        return (
            current_max_error / target_max_error
        ) ** 2 * self._apply_science_case_scaling_factor()

    def _apply_science_case_scaling_factor(self) -> float:
        """
        Apply the uncertainty factor based on the science case.

        Returns
        -------
        float
            The final scaling factor after applying uncertainty.
        """
        return 1.5 if self.science_case == "science case 1" else 1.0

    def _number_of_simulated_events(self) -> int:
        """
        Fetch the number of simulated events from the evaluator's data.

        Returns
        -------
        int
            The number of simulated events.
        """
        return self.evaluator.data.get("simulated_event_histogram", [0])

    def _calculate_scaled_events_at_grid_point(
        self, grid_point: tuple, scaling_factor: float
    ) -> float:
        """
        Calculate the scaled number of events for a specific energy grid point.

        Parameters
        ----------
        grid_point : tuple
            The grid point specifying energy, azimuth, zenith, NSB, and offset.
        scaling_factor : float
            The scaling factor to apply to the base events.

        Returns
        -------
        float
            The scaled number of events at the specified grid point (energy).
        """
        energy = grid_point[0]
        bin_edges = self.evaluator.create_bin_edges()
        bin_idx = np.digitize(energy, bin_edges) - 1

        if bin_idx < 0 or bin_idx >= len(self.evaluator.data["simulated_event_histogram"]):
            raise ValueError(f"Energy {energy} is outside the range of the simulated events data.")

        base_events = self.evaluator.data["simulated_event_histogram"][bin_idx]
        return base_events * scaling_factor
