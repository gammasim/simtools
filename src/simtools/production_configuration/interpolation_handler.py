"""Interpolates between instances of StatisticalErrorEvaluator using EventScaler."""

import astropy.units as u
import numpy as np
from scipy.interpolate import griddata

from simtools.production_configuration.event_scaler import EventScaler

__all__ = ["InterpolationHandler"]


class InterpolationHandler:
    """Handle interpolation between multiple StatisticalErrorEvaluator instances."""

    def __init__(self, evaluators, metrics: dict):
        self.evaluators = evaluators
        self.metrics = metrics
        self.event_scalers = [EventScaler(e, self.metrics) for e in self.evaluators]

        self.azimuths = [e.grid_point[1].to(u.deg).value for e in self.evaluators]
        self.zeniths = [e.grid_point[2].to(u.deg).value for e in self.evaluators]
        self.nsbs = [e.grid_point[3] for e in self.evaluators]
        self.offsets = [e.grid_point[4].to(u.deg).value for e in self.evaluators]

        self.energy_grids = [
            (e.data["bin_edges_low"][:-1] + e.data["bin_edges_high"][:-1]) / 2
            for e in self.evaluators
        ]
        self.scaled_events = [
            scaler.scale_events(return_sum=False) for scaler in self.event_scalers
        ]
        self.energy_thresholds = np.array([e.energy_threshold for e in self.evaluators])

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
            az = np.full(len(energy_grid), e.grid_point[1].to(u.deg).value)
            zen = np.full(len(energy_grid), e.grid_point[2].to(u.deg).value)
            nsb = np.full(len(energy_grid), e.grid_point[3])
            offset = np.full(len(energy_grid), e.grid_point[4].to(u.deg).value)

            # Combine grid points and data
            grid_points = np.column_stack([energy_grid.to(u.TeV).value, az, zen, nsb, offset])
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
            az = e.grid_point[1].to(u.deg).value
            zen = e.grid_point[2].to(u.deg).value
            nsb = e.grid_point[3]
            offset = e.grid_point[4].to(u.deg).value
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

    def plot_comparison(self, evaluator):
        """
        Plot a comparison between the simulated, scaled, and reconstructed events.

        Parameters
        ----------
        evaluator : StatisticalErrorEvaluator
            The evaluator for which to plot the comparison.
        """
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

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

        reconstructed_event_histogram, _ = np.histogram(
            evaluator.data["event_energies_reco"], bins=evaluator.data["bin_edges_low"]
        )
        plt.plot(midpoints[:-1], reconstructed_event_histogram, label="Reconstructed")

        plt.legend()
        plt.xscale("log")
        plt.xlabel("Energy (Midpoint of Bin Edges)")
        plt.ylabel("Event Count")
        plt.title("Comparison of Simulated, scaled, and reconstructed events")
        plt.show()
