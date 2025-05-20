"""Handle interpolation between multiple StatisticalUncertaintyEvaluator instances."""

import logging

import astropy.units as u
import numpy as np
from scipy.interpolate import griddata

from simtools.production_configuration.derive_production_statistics import (
    ProductionStatisticsDerivator,
)

__all__ = ["InterpolationHandler"]


class InterpolationHandler:
    """Handle interpolation between multiple StatisticalUncertaintyEvaluator instances."""

    def __init__(self, evaluators, metrics: dict, grid_points_production: dict):
        self._logger = logging.getLogger(__name__)

        self.evaluators = evaluators
        self.metrics = metrics
        self.grid_points_production = grid_points_production

        self.derive_production_statistics = [
            ProductionStatisticsDerivator(e, self.metrics) for e in self.evaluators
        ]

        self.azimuths = [e.grid_point[1].to(u.deg).value for e in self.evaluators]
        self.zeniths = [e.grid_point[2].to(u.deg).value for e in self.evaluators]
        self.nsbs = [e.grid_point[3] for e in self.evaluators]
        self.offsets = [e.grid_point[4].to(u.deg).value for e in self.evaluators]
        self.energy_grids = [
            (e.data["bin_edges_low"][:-1] + e.data["bin_edges_high"][:-1]) / 2
            for e in self.evaluators
        ]
        self.production_statistics = [
            derivator.derive_statistics(return_sum=False)
            for derivator in self.derive_production_statistics
        ]
        self.production_statistics_sum = [
            derivator.derive_statistics(return_sum=True)
            for derivator in self.derive_production_statistics
        ]

        self.energy_thresholds = np.array([e.energy_threshold for e in self.evaluators])

        self.data, self.grid_points = self._build_data_array()
        self.interpolated_production_statistics = None
        self.interpolated_production_statistics_with_energy = None

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
        if not self.evaluators:
            return np.array([]), np.array([])

        # Flatten the energy grid and other dimensions into a combined array
        flat_data_list = []
        flat_grid_points = []

        for e, energy_grid, production_statistics in zip(
            self.evaluators, self.energy_grids, self.production_statistics
        ):
            az = np.full(len(energy_grid), e.grid_point[1].to(u.deg).value)
            zen = np.full(len(energy_grid), e.grid_point[2].to(u.deg).value)
            nsb = np.full(len(energy_grid), e.grid_point[3])
            offset = np.full(len(energy_grid), e.grid_point[4].to(u.deg).value)

            # Combine grid points and data
            grid_points = np.column_stack([energy_grid.to(u.TeV).value, az, zen, nsb, offset])
            flat_grid_points.append(grid_points)
            flat_data_list.append(production_statistics)

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

    def build_grid_points_no_energy(self):
        """
        Build grid points without energy dimension.

        Returns
        -------
        np.ndarray
            The grid points without the energy dimension.
        """
        flat_data_list = []
        flat_grid_points = []

        for e, production_statistics_sum in zip(self.evaluators, self.production_statistics_sum):
            az = e.grid_point[1].to(u.deg).value
            zen = e.grid_point[2].to(u.deg).value
            nsb = e.grid_point[3]
            offset = e.grid_point[4].to(u.deg).value
            flat_data_list.append(production_statistics_sum.value)

            grid_points = np.column_stack([az, zen, nsb, offset])
            flat_grid_points.append(grid_points)

        flat_grid_points = np.vstack(flat_grid_points)
        return flat_data_list, flat_grid_points

    def interpolate(self) -> np.ndarray:
        """
        Interpolate the number of simulated events given query points.

        Returns
        -------
        np.ndarray
            Interpolated values at the query points.
        """
        # Points defining the grid from DL2 and the production statistics
        production_statistic, reduced_grid_points = self.build_grid_points_no_energy()

        # Convert production_statistic to a proper numpy array
        production_statistic = np.array(production_statistic, dtype=float)

        # Remove flat dimensions for interpolation
        reduced_grid_points, non_flat_mask = self._remove_flat_dimensions(reduced_grid_points)

        # Convert grid_points_production to a numerical array
        production_grid_points = []
        for point in self.grid_points_production:
            production_grid_points.append(
                [
                    point["azimuth"]["value"],
                    point["zenith_angle"]["value"],
                    point["nsb"]["value"],
                    point["offset"]["value"],
                ]
            )
        production_grid_points = np.array(production_grid_points)

        # Apply the non-flat mask to the production grid points
        reduced_production_grid_points = production_grid_points[:, non_flat_mask]

        # Debugging output
        print("reduced_grid_points", reduced_grid_points)
        print("production_statistic", production_statistic)
        print("reduced_production_grid_points", reduced_production_grid_points)

        # Perform interpolation
        self.interpolated_production_statistics = griddata(
            reduced_grid_points,
            production_statistic,
            reduced_production_grid_points,
            method="linear",
            fill_value=np.nan,
            rescale=True,
        )

        # Handle energy-dependent statistics
        reduced_grid_points_energy_dependent, non_flat_mask = self._remove_flat_dimensions(
            self.grid_points
        )
        energy_dependent_statistics = []
        if not all(np.array_equal(self.energy_grids[0], grid) for grid in self.energy_grids):
            self._logger.warning(
                "Energy grids are not identical across evaluators "
                "(only relevant for comparison plots)."
            )

        energy_query_grid = []
        for energy in self.energy_grids[0]:
            for grid_point in reduced_production_grid_points:
                energy_query_grid.append(np.hstack([energy.to(u.TeV).value, grid_point]))
        energy_query_grid = np.array(energy_query_grid)

        interpolated_value = griddata(
            reduced_grid_points_energy_dependent,
            self.data,
            energy_query_grid,
            method="linear",
            fill_value=np.nan,
            rescale=True,
        )
        energy_dependent_statistics.append(
            interpolated_value.reshape(len(reduced_production_grid_points), -1)
        )

        self.interpolated_production_statistics_with_energy = np.array(energy_dependent_statistics)

        return self.interpolated_production_statistics

    def interpolate_energy_threshold(self, grid_point: np.ndarray) -> float:
        """
        Interpolate the energy threshold for a given grid point.

        Parameters
        ----------
        grid_point : np.ndarray
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
        reduced_grid_point = grid_point[0][full_non_flat_mask]

        interpolated_threshold = griddata(
            reduced_grid_points,
            flat_energy_thresholds,
            reduced_grid_point,
            method="linear",
            fill_value=np.nan,
            rescale=False,
        )

        return interpolated_threshold.item()

    def plot_comparison(self):
        """
        Plot a comparison between the interpolated production statistics and reconstructed events.

        Parameters
        ----------
        evaluator : StatisticalUncertaintyEvaluator
            The evaluator for which to plot the comparison.

        Returns
        -------
        matplotlib.axes.Axes
            The Axes object containing the plot.
        """
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

        # use first of the evaluators to get the bin edges and the events to compare to
        bin_edges_low = self.evaluators[0].data["bin_edges_low"][:-1]
        bin_edges_high = self.evaluators[0].data["bin_edges_high"][:-1]
        midpoints = (bin_edges_low + bin_edges_high) / 2

        _, ax = plt.subplots()

        ax.plot(
            midpoints,
            self.interpolated_production_statistics_with_energy[0][10],
            label="Interpolated Production Statistics",
        )

        reconstructed_event_histogram, _ = np.histogram(
            self.evaluators[0].data["event_energies_reco"],
            bins=self.evaluators[0].data["bin_edges_low"],
        )
        ax.plot(midpoints, reconstructed_event_histogram, label="Reconstructed Events")

        ax.legend()
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Energy (TeV)")
        ax.set_ylabel("Event Count")
        ax.set_title("Comparison of Interpolated and Reconstructed Events")

        return ax
