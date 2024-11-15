"""
Module defines the `SimulationConfig` class.

Used to configure and generate simulation parameters for a specific grid point
based on statistical uncertainties.
The class considers parameters, such as azimuth, elevation, and night sky background,
to compute core scatter area, viewcone, and the required number of simulated events.

Key Components:
---------------
- `SimulationConfig`: Main class to handle simulation configuration for a grid point.
  - Attributes:
    - `grid_point` (dict): Contains azimuth, elevation, and night sky background.
    - `ctao_data_level` (str): The data level for the simulation (e.g., 'A', 'B', 'C').
    - `science_case` (str): The science case for the simulation.
    - `file_path` (str): Path to the dl2_mc_events_file FITS file
       used for statistical error evaluation.
    - `file_type` (str): Type of the dl2_mc_events_file FITS file ('On-source' or 'Offset').
    - `metrics` (dict, optional): Dictionary of metrics to evaluate.

"""

import logging

from simtools.production_configuration.calculate_statistical_errors_grid_point import (
    StatisticalErrorEvaluator,
)
from simtools.production_configuration.event_scaler import EventScaler

_logger = logging.getLogger(__name__)


class SimulationConfig:
    """
    Configures simulation parameters for a specific grid point.

    Parameters
    ----------
    grid_point : dict
        Dictionary representing a grid point with azimuth, elevation, and night sky background.
    ctao_data_level : str
        The data level (e.g., 'A', 'B', 'C') for the simulation configuration.
    science_case : str
        The science case for the simulation configuration.
    file_path : str
        Path to the dl2_mc_events_file FITS file for statistical uncertainty evaluation.
    file_type : str
        Type of the dl2_mc_events_file FITS file ('On-source' or 'Offset').
    metrics : dict, optional
        Dictionary of metrics to evaluate.
    """

    def __init__(
        self,
        grid_point: dict[str, float],
        ctao_data_level: str,
        science_case: str,
        file_path: str,
        file_type: str,
        metrics: dict[str, float] | None = None,
    ):
        """Initialize the simulation configuration for a grid point."""
        self.grid_point = grid_point
        self.ctao_data_level = ctao_data_level
        self.science_case = science_case
        self.file_path = file_path
        self.file_type = file_type
        self.metrics = metrics or {}
        self.evaluator = StatisticalErrorEvaluator(file_path, file_type, metrics)
        self.event_scaler = EventScaler(self.evaluator, science_case, self.metrics)
        self.simulation_params = {}

    def configure_simulation(self) -> dict[str, float]:
        """
        Configure the simulation parameters for the grid point, data level, and science case.

        Returns
        -------
        dict
            A dictionary with simulation parameters such as core scatter area,
              viewcone, and number of simulated events.
        """
        core_scatter_area = self._calculate_core_scatter_area()
        viewcone = self._calculate_viewcone()
        number_of_events = self.calculate_required_events()

        self.simulation_params = {
            "core_scatter_area": core_scatter_area,
            "viewcone": viewcone,
            "number_of_events": number_of_events,
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
        Calculate the core scatter area based on the grid point and data level.

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
        Calculate the viewcone based on the grid point conditions and data level.

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