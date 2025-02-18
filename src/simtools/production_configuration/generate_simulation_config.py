"""Derives simulation configuration parameters for a grid point based on several metrics."""

from simtools.production_configuration.calculate_statistical_errors_grid_point import (
    StatisticalErrorEvaluator,
)
from simtools.production_configuration.event_scaler import EventScaler

__all__ = ["SimulationConfig"]


class SimulationConfig:
    """
    Configures simulation parameters for a specific grid point.

    Parameters
    ----------
    grid_point : dict
        Dictionary representing a grid point with azimuth, elevation, and night sky background.
    file_path : str
        Path to the DL2 MC event file for statistical uncertainty evaluation.
    file_type : str
        Type of the DL2 MC event file ('point-like' or 'cone').
    metrics : dict, optional
        Dictionary of metrics to evaluate.
    """

    def __init__(
        self,
        grid_point: dict[str, float],
        file_path: str,
        file_type: str,
        metrics: dict[str, float] | None = None,
    ):
        """Initialize the simulation configuration for a grid point."""
        self.grid_point = grid_point
        self.file_path = file_path
        self.file_type = file_type
        self.metrics = metrics or {}
        self.evaluator = StatisticalErrorEvaluator(file_path, file_type, metrics)
        self.event_scaler = EventScaler(self.evaluator, self.metrics)
        self.simulation_params = {}

    def configure_simulation(self) -> dict[str, float]:
        """
        Configure the simulation parameters for the grid point.

        Returns
        -------
        dict
            A dictionary with simulation parameters such as core scatter area,
              viewcone, and number of simulated events.
        """
        self.simulation_params = {
            "core_scatter_area": self._calculate_core_scatter_area(),
            "viewcone": self._calculate_viewcone(),
            "number_of_events": self.calculate_required_events(),
        }
        return self.simulation_params

    def calculate_required_events(self) -> int:
        """
        Calculate the required number of simulated events based on statistical error metrics.

        Uses the EventScaler to scale the events.

        Returns
        -------
        int
            The number of simulated events required.
        """
        return self.event_scaler.scale_events()

    def _calculate_core_scatter_area(self) -> float:
        """
        Calculate the core scatter area based on the grid point.

        Returns
        -------
        float
            The core scatter area.
        """
        base_area = self._fetch_simulated_core_scatter_area()
        area_factor = self._get_area_factor_from_grid_point()
        return base_area * area_factor

    def _calculate_viewcone(self) -> float:
        """
        Calculate the viewcone based on the grid point conditions.

        Returns
        -------
        float
            The viewcone value.
        """
        base_viewcone = self._fetch_simulated_viewcone()
        viewcone_factor = self._get_viewcone_factor_from_grid_point()

        return base_viewcone * viewcone_factor

    def _get_area_factor_from_grid_point(self) -> float:
        """
        Determine the area factor.

        The area factor is based on the grid point's azimuth,
          elevation, and night sky background.

        Returns
        -------
        float
            The area factor.
        """
        azimuth = self.grid_point.get("azimuth", 0)
        elevation = self.grid_point.get("elevation", 0)
        night_sky_background = self.grid_point.get("night_sky_background", 0)

        # Implement reading of factor from LUT
        return azimuth + elevation + night_sky_background

    def _get_viewcone_factor_from_grid_point(self) -> float:
        """
        Determine the viewcone factor.

        The factor is based on the grid point's azimuth,
          elevation, and night sky background.

        Returns
        -------
        float
            The viewcone factor.
        """
        azimuth = self.grid_point.get("azimuth", 0)
        elevation = self.grid_point.get("elevation", 0)
        night_sky_background = self.grid_point.get("night_sky_background", 0)

        # Implement reading of factor from LUT
        return azimuth + elevation + night_sky_background

    def _fetch_simulated_core_scatter_area(self) -> float:
        """
        Fetch the core scatter area from existing simulated files based on grid point conditions.

        Returns
        -------
        float
            The fetched core scatter outer bound.
        """
        return self.evaluator.data["core_range"]

    def _fetch_simulated_viewcone(self) -> float:
        """
        Fetch the viewcone from existing simulated files based on grid point conditions.

        Returns
        -------
        float
            The fetched viewcone outer bound.
        """
        return self.evaluator.data["viewcone"]
