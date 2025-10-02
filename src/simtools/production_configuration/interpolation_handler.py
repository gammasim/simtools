"""Handle interpolation between multiple StatisticalUncertaintyEvaluator instances."""

import logging

import astropy.units as u
import numpy as np
from scipy.interpolate import griddata

from simtools.production_configuration.derive_production_statistics import (
    ProductionStatisticsDerivator,
)


class InterpolationHandler:
    """
    Calculate the required events for production via interpolation from a grid.

    This class provides methods to interpolate production statistics across a grid of
    parameter values (azimuth, zenith, NSB, offset) and energy.
    """

    def __init__(self, evaluators, metrics: dict, grid_points_production: list):
        """
        Initialize the InterpolationHandler.

        Parameters
        ----------
        evaluators : list
            List of StatisticalUncertaintyEvaluator instances.
        metrics : dict
            Dictionary of metrics to use for production statistics.
        grid_points_production : list
            List of grid points for interpolation, each being a dictionary with keys
            'azimuth', 'zenith_angle', 'nsb', 'offset' etc.
        """
        self._logger = logging.getLogger(__name__)
        self.evaluators = evaluators
        self.metrics = metrics
        self.grid_points_production = grid_points_production

        self._initialize_derivators()
        self._extract_grid_properties()

        self.data, self.grid_points = self._build_data_array()
        self.interpolated_production_statistics = None
        self.interpolated_production_statistics_with_energy = None
        self._non_flat_mask = None

    def _initialize_derivators(self):
        """Initialize production statistics derivators for all evaluators."""
        self.derive_production_statistics = [
            ProductionStatisticsDerivator(e, self.metrics) for e in self.evaluators
        ]

        self.production_statistics = [
            derivator.derive_statistics(return_sum=False)
            for derivator in self.derive_production_statistics
        ]
        self.production_statistics_sum = [
            derivator.derive_statistics(return_sum=True)
            for derivator in self.derive_production_statistics
        ]

    def _extract_grid_properties(self):
        """Extract grid properties from evaluators."""
        self.azimuths = [e.grid_point[1].to(u.deg).value for e in self.evaluators]
        self.zeniths = [e.grid_point[2].to(u.deg).value for e in self.evaluators]
        self.nsbs = [e.grid_point[3] for e in self.evaluators]
        self.offsets = [e.grid_point[4].to(u.deg).value for e in self.evaluators]
        self.energy_grids = [
            (e.data["bin_edges_low"][:-1] + e.data["bin_edges_high"][:-1]) / 2
            for e in self.evaluators
        ]
        self.energy_thresholds = np.array([e.energy_threshold for e in self.evaluators])

        # Check if energy grids are consistent
        if self.energy_grids and not all(
            np.array_equal(self.energy_grids[0], grid) for grid in self.energy_grids
        ):
            self._logger.warning(
                "Energy grids are not identical across evaluators. "
                "Using the first evaluator's energy grid for interpolation."
            )

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

        flat_data_list = []
        flat_grid_points = []

        for i, (energy_grid, production_statistics) in enumerate(
            zip(self.energy_grids, self.production_statistics)
        ):
            az = self.azimuths[i]
            zen = self.zeniths[i]
            nsb = self.nsbs[i]
            offset = self.offsets[i]

            az_array = np.full(len(energy_grid), az)
            zen_array = np.full(len(energy_grid), zen)
            nsb_array = np.full(len(energy_grid), nsb)
            offset_array = np.full(len(energy_grid), offset)

            grid_points = np.column_stack(
                [energy_grid.to(u.TeV).value, az_array, zen_array, nsb_array, offset_array]
            )

            flat_grid_points.append(grid_points)
            flat_data_list.append(production_statistics)

        flat_grid_points = np.vstack(flat_grid_points)
        flat_data = np.hstack(flat_data_list)

        sorted_indices = np.argsort(flat_grid_points[:, 0])
        sorted_grid_points = flat_grid_points[sorted_indices]
        sorted_data = flat_data[sorted_indices]

        return sorted_data, sorted_grid_points

    def _remove_flat_dimensions(self, grid_points, threshold=1e-6):
        """
        Identify and remove flat dimensions (dimensions with no variance).

        Parameters
        ----------
        grid_points : np.ndarray
            Grid points to analyze.
        threshold : float, optional
            Threshold for determining flatness, by default 1e-6

        Returns
        -------
        tuple
            (reduced_grid_points, non_flat_mask)
        """
        if grid_points.size == 0:
            return grid_points, np.array([], dtype=bool)

        variance = np.var(grid_points, axis=0)
        non_flat_mask = variance > threshold

        if not np.any(non_flat_mask):
            self._logger.warning(
                "All dimensions are flat. Keeping all dimensions for interpolation."
            )
            return grid_points, np.ones_like(variance, dtype=bool)

        reduced_grid_points = grid_points[:, non_flat_mask]
        return reduced_grid_points, non_flat_mask

    def build_grid_points_no_energy(self):
        """
        Build grid points without energy dimension.

        Returns
        -------
        tuple
            (production_statistics, grid_points_no_energy)
        """
        if not self.evaluators:
            self._logger.error("No evaluators available for grid point building.")
            return np.array([]), np.array([])

        flat_data_list = []
        flat_grid_points = []

        for i, production_statistics_sum in enumerate(self.production_statistics_sum):
            az = self.azimuths[i]
            zen = self.zeniths[i]
            nsb = self.nsbs[i]
            offset = self.offsets[i]

            flat_data_list.append(float(production_statistics_sum.value))

            grid_point = np.array([[az, zen, nsb, offset]])
            flat_grid_points.append(grid_point)

        flat_grid_points = np.vstack(flat_grid_points)
        return flat_data_list, flat_grid_points

    def _prepare_energy_independent_data(self):
        """
        Prepare data for energy-independent interpolation.

        Returns
        -------
        tuple
            (production_statistic, grid_points_no_energy)
        """
        production_statistic, grid_points_no_energy = self.build_grid_points_no_energy()
        production_statistic = np.array(production_statistic, dtype=float)
        grid_points_no_energy, non_flat_mask = self._remove_flat_dimensions(grid_points_no_energy)

        self._non_flat_mask = non_flat_mask  # Store for later use
        return production_statistic, grid_points_no_energy

    def _prepare_production_grid_points(self):
        """
        Convert grid_points_production to a format suitable for interpolation.

        Returns
        -------
        np.ndarray
            Reduced production grid points.
        """
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

        return production_grid_points[:, self._non_flat_mask]

    def _perform_interpolation(self, grid_points, values, query_points, method="linear"):
        """
        Perform interpolation using griddata.

        Parameters
        ----------
        grid_points : np.ndarray
            Grid points for interpolation.
        values : np.ndarray
            Values at the grid points.
        query_points : np.ndarray
            Query points for interpolation.
        method : str, optional
            Interpolation method, by default "linear".

        Returns
        -------
        np.ndarray
            Interpolated values.
        """
        self._logger.debug(f"Grid points shape: {grid_points.shape}")
        self._logger.debug(f"Values shape: {values.shape}")
        self._logger.debug(f"Query points shape: {query_points.shape}")

        return griddata(
            grid_points,
            values,
            query_points,
            method=method,
            fill_value=np.nan,
            rescale=True,
        )

    def _perform_interpolation_with_energy(self):
        """
        Perform energy-dependent interpolation.

        Returns
        -------
        np.ndarray
            Energy-dependent interpolated values.
        """
        # Get grid points with energy dimension
        grid_points_energy = self.grid_points
        grid_points_energy, _ = self._remove_flat_dimensions(grid_points_energy)

        # Build energy query grid
        reduced_production_grid_points = self._prepare_production_grid_points()
        energy_grid = self.energy_grids[0] if self.energy_grids else []

        energy_query_grid = []
        for energy in energy_grid:
            for grid_point in reduced_production_grid_points:
                energy_query_grid.append(np.hstack([energy.to(u.TeV).value, grid_point]))

        energy_query_grid = np.array(energy_query_grid)

        self._logger.debug(f"Grid points with energy shape: {grid_points_energy.shape}")
        self._logger.debug(f"Data shape: {self.data.shape}")
        self._logger.debug(f"Energy query grid shape: {energy_query_grid.shape}")

        interpolated_values = self._perform_interpolation(
            grid_points_energy, self.data, energy_query_grid
        )

        reshaped = interpolated_values.reshape(
            len(reduced_production_grid_points), len(energy_grid)
        )
        return np.array([reshaped])

    def interpolate(self) -> np.ndarray:
        """
        Interpolate production statistics at the grid points specified in grid_points_production.

        This method performs two types of interpolation:
        1. Energy-independent interpolation using the sum of production statistics
        2. Energy-dependent interpolation for each energy bin

        Returns
        -------
        np.ndarray
            Interpolated values at the query points.
        """
        if not self.evaluators:
            self._logger.error("No evaluators available for interpolation.")
            return np.array([])

        # Energy-independent interpolation
        production_statistic, grid_points_no_energy = self._prepare_energy_independent_data()
        reduced_production_grid_points = self._prepare_production_grid_points()

        self.interpolated_production_statistics = self._perform_interpolation(
            grid_points_no_energy, production_statistic, reduced_production_grid_points
        )

        # Energy-dependent interpolation
        self.interpolated_production_statistics_with_energy = (
            self._perform_interpolation_with_energy()
        )

        return self.interpolated_production_statistics

    def plot_comparison(self, grid_point_index=0):
        """
        Plot a comparison between interpolated production statistics and reconstructed events.

        Parameters
        ----------
        grid_point_index : int, optional
            Index of the grid point to plot, by default 0

        Returns
        -------
        matplotlib.axes.Axes
            The Axes object containing the plot.
        """
        import matplotlib.pyplot as plt  # pylint: disable=C0415

        if not self.evaluators:
            self._logger.error("No evaluators available for plotting.")
            _, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return ax

        # Use first evaluator for energy bins
        bin_edges_low = self.evaluators[0].data["bin_edges_low"][:-1]
        bin_edges_high = self.evaluators[0].data["bin_edges_high"][:-1]
        midpoints = (bin_edges_low + bin_edges_high) / 2

        if (
            self.interpolated_production_statistics_with_energy is None
            or len(self.interpolated_production_statistics_with_energy) == 0
            or len(self.interpolated_production_statistics_with_energy[0]) <= grid_point_index
        ):
            self._logger.warning(
                f"Invalid grid point index {grid_point_index}. Using index 0 instead."
            )
            grid_point_index = 0

        _, ax = plt.subplots()

        if (
            self.interpolated_production_statistics_with_energy is not None
            and len(self.interpolated_production_statistics_with_energy) > 0
        ):
            interpolated_stats = self.interpolated_production_statistics_with_energy[0][
                grid_point_index
            ]
            ax.plot(midpoints, interpolated_stats, label="Interpolated Production Statistics")

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
