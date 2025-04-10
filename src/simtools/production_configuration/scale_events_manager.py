r"""
Module to run the StatisticalErrorEvaluator and interpolate results.

This module provides the `ScaleEventsManager` class, which manages the workflow for scaling
events and performing interpolation. It is designed to handle the evaluation of statistical
uncertainties from DL2 Monte Carlo (MC) event files and perform interpolation for a specified
query point.

The module includes functionality to:
- Initialize evaluators for statistical error calculations based on input parameters.
- Perform interpolation using the initialized evaluators to estimate scaled events at a query point.
- Write the results of the interpolation to an output file.

"""

import itertools
import json
import logging
from pathlib import Path

import astropy.units as u
import numpy as np

from simtools.io_operations import io_handler
from simtools.production_configuration.calculate_statistical_errors_grid_point import (
    StatisticalErrorEvaluator,
)
from simtools.production_configuration.interpolation_handler import InterpolationHandler
from simtools.utils.general import collect_data_from_file


class ScaleEventsManager:
    """
    Manages the workflow for scaling events and performing interpolation.

    This class handles the evaluation of statistical uncertainties from DL2 MC event files
    and performs interpolation for a specified query point.
    """

    def __init__(self, args_dict):
        """
        Initialize the manager with the provided arguments.

        Parameters
        ----------
        args_dict : dict
            Dictionary of command-line arguments.
        """
        self.args = args_dict
        self.logger = logging.getLogger(__name__)
        self.output_path = io_handler.IOHandler().get_output_directory(Path(__file__).stem)
        self.output_filepath = Path(self.output_path).joinpath(f"{self.args['output_file']}")
        self.metrics = collect_data_from_file(self.args["metrics_file"])
        self.evaluator_instances = []

    def initialize_evaluators(self):
        """Initialize StatisticalErrorEvaluator instances for the given zeniths and offsets."""
        if not (self.args["base_path"] and self.args["zeniths"] and self.args["offsets"]):
            self.logger.warning("No files read")
            self.logger.warning(f"Base Path: {self.args['base_path']}")
            self.logger.warning(f"Zeniths: {self.args['zeniths']}")
            self.logger.warning(f"Offsets: {self.args['offsets']}")
            return

        for zenith, offset in itertools.product(self.args["zeniths"], self.args["offsets"]):
            file_name = self.args["file_name_template"].format(zenith=int(zenith))
            file_path = Path(self.args["base_path"]).joinpath(file_name)

            if not file_path.exists():
                self.logger.warning(f"File not found: {file_path}. Skipping.")
                continue

            evaluator = StatisticalErrorEvaluator(
                file_path,
                file_type="Gamma-cone",
                metrics=self.metrics,
                grid_point=(1 * u.TeV, 180 * u.deg, zenith, 0, offset * u.deg),
            )
            evaluator.calculate_metrics()
            self.evaluator_instances.append(evaluator)

    def perform_interpolation(self):
        """Perform interpolation for the query point."""
        if not self.evaluator_instances:
            self.logger.error("No evaluators initialized. Cannot perform interpolation.")
            return None

        interpolation_handler = InterpolationHandler(self.evaluator_instances, metrics=self.metrics)
        query_point = self.args.get("query_point")
        if not query_point or len(query_point) != 5:
            raise ValueError(
                "Invalid query point format. "
                f"Expected 5 values, got {len(query_point) if query_point else 'None'}."
            )
        query_points = np.array([self.args["query_point"]])
        return interpolation_handler.interpolate(query_points)

    def write_output(self, scaled_events):
        """Write the interpolation results to a file."""
        output_data = {
            "query_point": self.args["query_point"],
            "scaled_events": scaled_events.tolist(),
        }
        with open(self.output_filepath, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4)
        self.logger.info(f"Output saved to {self.output_filepath}")
        self.logger.info(
            f"Scaled events for grid point {self.args['query_point']}: {scaled_events}"
        )

    def run(self):
        """Run the scaling and interpolation workflow."""
        self.logger.info(f"args dict: {self.args}")
        self.initialize_evaluators()
        scaled_events = self.perform_interpolation()
        self.write_output(scaled_events)
