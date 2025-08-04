"""
Derives the required statistics for a requested set of production parameters through interpolation.

This module provides the `ProductionStatisticsHandler` class, which manages the workflow for
derivation of required number of events for a simulation production using pre-defined metrics.

The module includes functionality to:
- Initialize evaluators for statistical uncertainty calculations based on input parameters.
- Perform interpolation using the initialized evaluators to estimate production statistics at a
query point.
- Write the results of the interpolation to an output file.
"""

import itertools
import json
import logging
from pathlib import Path

import astropy.units as u

from simtools.io.ascii_handler import collect_data_from_file
from simtools.production_configuration.calculate_statistical_uncertainties_grid_point import (
    StatisticalUncertaintyEvaluator,
)
from simtools.production_configuration.interpolation_handler import InterpolationHandler


class ProductionStatisticsHandler:
    """
    Handles the workflow for deriving production statistics.

    This class manages the evaluation of statistical uncertainties from DL2 MC event files
    and performs interpolation to estimate the required number of events for a simulation
    production at a specified query point.
    """

    def __init__(self, args_dict, output_path):
        """
        Initialize the manager with the provided arguments.

        Parameters
        ----------
        args_dict : dict
            Dictionary of command-line arguments.
        output_path : Path
            Path to the directory where the event statistics output file will be saved.
        """
        self.args = args_dict
        self.logger = logging.getLogger(__name__)
        self.output_path = output_path
        self.metrics = collect_data_from_file(self.args["metrics_file"])
        self.evaluator_instances = []
        self.interpolation_handler = None
        self.grid_points_production = self._load_grid_points_production()

    def _load_grid_points_production(self):
        """Load grid points from the JSON file."""
        grid_points_production_file = self.args["grid_points_production_file"]
        return collect_data_from_file(grid_points_production_file)

    def initialize_evaluators(self):
        """Initialize StatisticalUncertaintyEvaluator instances for the given grid point."""
        if not (
            self.args["base_path"]
            and self.args["zeniths"]
            and self.args["azimuths"]
            and self.args["nsb"]
            and self.args["offsets"]
        ):
            self.logger.warning("No files read")
            self.logger.warning(f"Base Path: {self.args['base_path']}")
            self.logger.warning(f"Zeniths: {self.args['zeniths']}")
            self.logger.warning(f"Camera offsets: {self.args['offsets']}")
            return

        for zenith, azimuth, nsb, offset in itertools.product(
            self.args["zeniths"], self.args["azimuths"], self.args["nsb"], self.args["offsets"]
        ):
            file_name = self.args["file_name_template"].format(
                zenith=int(zenith),
                azimuth=azimuth,
                nsb=nsb,
                offset=offset,
            )
            file_path = Path(self.args["base_path"]).joinpath(file_name)

            if not file_path.exists():
                self.logger.warning(f"File not found: {file_path}. Skipping.")
                continue

            evaluator = StatisticalUncertaintyEvaluator(
                file_path,
                metrics=self.metrics,
                grid_point=(None, azimuth, zenith, nsb, offset * u.deg),
            )
            evaluator.calculate_metrics()
            self.evaluator_instances.append(evaluator)

    def perform_interpolation(self):
        """Perform interpolation for the query point."""
        if not self.evaluator_instances:
            self.logger.error("No evaluators initialized. Cannot perform interpolation.")
            return None

        self.interpolation_handler = InterpolationHandler(
            self.evaluator_instances,
            metrics=self.metrics,
            grid_points_production=self.grid_points_production,
        )
        qrid_points_with_statistics = []

        interpolated_production_statistics = self.interpolation_handler.interpolate()
        for grid_point, statistics in zip(
            self.grid_points_production, interpolated_production_statistics
        ):
            qrid_points_with_statistics.append(
                {
                    "grid_point": grid_point,
                    "interpolated_production_statistics": float(statistics),
                }
            )
        return qrid_points_with_statistics

    def write_output(self, production_statistics):
        """Write the derived event statistics to a file."""
        output_data = (production_statistics,)
        output_filename = self.args["output_file"]
        self.output_path.mkdir(parents=True, exist_ok=True)
        output_file_path = self.output_path.joinpath(output_filename)
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4)
        self.logger.info(f"Output saved to {self.output_path}")

    def plot_production_statistics_comparison(self):
        """Plot the derived event statistics."""
        ax = self.interpolation_handler.plot_comparison()
        plot_path = self.output_path.joinpath("production_statistics_comparison.png")
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(plot_path)
        self.logger.info(f"Plot saved to {plot_path}")

    def run(self):
        """Run the scaling and interpolation workflow."""
        self.logger.info(f"Grid Points File: {self.args['grid_points_production_file']}")
        self.logger.info(f"Metrics File: {self.args['metrics_file']}")

        self.initialize_evaluators()
        production_statistics = self.perform_interpolation()
        if self.args.get("plot_production_statistics"):
            self.plot_production_statistics_comparison()

        self.write_output(production_statistics)
