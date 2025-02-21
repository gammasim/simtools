#!/usr/bin/python3

r"""
Application to run the StatisticalErrorEvaluator and interpolate results.

This application evaluates statistical uncertainties from DL2 MC event files
based on input parameters like zenith angles and offsets, and can perform interpolation
for a specified grid point.

Command line arguments
----------------------
base_path (str, required)
    Path to the directory containing the DL2 MC event file for interpolation.
zeniths (list of int, required)
    List of zenith angles to consider.
offsets (list of int, required)
    List of offsets in degrees.
interpolate (bool, optional)
    If set, performs interpolation for a specific grid point.
query_point (list of int, optional)
    Grid point for interpolation (energy, azimuth, zenith, NSB, offset).

Example
-------
To evaluate statistical uncertainties and perform interpolation, run the command line script:

.. code-block:: console

    simtools-production-scale-events --base_path tests/resources/production_dl2_fits/ \
        --zeniths 20 52 40 60 --offsets 0 --interpolate --query_point 1 180 30 0 0 \
        --metrics_file "path/to/metrics.yaml"


The output will display the scaled events for the specified grid point.
"""

import json
import logging
from pathlib import Path

import astropy.units as u
import numpy as np

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.configuration.commandline_parser import CommandLineParser
from simtools.io_operations import io_handler
from simtools.production_configuration.calculate_statistical_errors_grid_point import (
    StatisticalErrorEvaluator,
)
from simtools.production_configuration.interpolation_handler import InterpolationHandler


def _parse(label, description):
    """
    Parse command line arguments for the statistical error evaluator application.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments.
    """
    config = configurator.Configurator(label=label, description=description)

    config.parser.add_argument(
        "--base_path",
        type=str,
        required=True,
        help="Path to the DL2 MC event files for interpolation.",
    )
    config.parser.add_argument(
        "--zeniths", nargs="+", type=CommandLineParser.zenith_angle, help="List of zenith angles."
    )
    config.parser.add_argument(
        "--offsets", nargs="+", type=float, help="List of offsets in degrees."
    )

    config.parser.add_argument(
        "--interpolate", action="store_true", help="Interpolate results for a specific grid point."
    )
    config.parser.add_argument(
        "--query_point",
        nargs=5,
        type=float,
        help="Grid point for interpolation (energy, azimuth, zenith, NSB, offset).",
    )
    config.parser.add_argument(
        "--output_file",
        type=str,
        default="interpolated_scaled_events.json",
        help="Output file to store the results. (default: 'interpolated_scaled_events.json').",
    )
    config.parser.add_argument(
        "--metrics_file",
        type=str,
        default="production_simulation_config_metrics.yml",
        help="Metrics definition file. (default: production_simulation_config_metrics.yml)",
    )
    return config.initialize(db_config=False)


def main():
    """Run the evaluator and interpolate."""
    label = Path(__file__).stem
    args_dict, _ = _parse(
        label,
        "Evaluate statistical uncertainties from DL2 MC event files and interpolate results.",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info(f"args dict: {args_dict}")

    output_path = io_handler.IOHandler().get_output_directory(label)
    output_filepath = Path(output_path).joinpath(f"{args_dict['output_file']}")

    evaluator_instances = []

    metrics = (
        gen.collect_data_from_file(args_dict["metrics_file"]) if "metrics_file" in args_dict else {}
    )

    if args_dict["base_path"] and args_dict["zeniths"] and args_dict["offsets"]:
        for zenith in args_dict["zeniths"]:
            for offset in args_dict["offsets"]:
                # Build file path based on base_path, zenith, and offset
                file_name = f"prod6_LaPalma-{int(zenith.value)}deg_"
                file_name += "gamma_cone.N.Am-4LSTs09MSTs_ID0_reduced.fits"
                file_path = Path(args_dict["base_path"]).joinpath(file_name)

                evaluator = StatisticalErrorEvaluator(
                    file_path,
                    file_type="Gamma-cone",
                    metrics=metrics,
                    grid_point=(1 * u.TeV, 180 * u.deg, zenith, 0, offset * u.deg),
                )

                evaluator.calculate_metrics()
                evaluator_instances.append(evaluator)

    else:
        logger.warning("No files read")
        logger.warning(f"Base Path: {args_dict['base_path']}")
        logger.warning(f"Zeniths: {args_dict['zeniths']}")
        logger.warning(f"Offsets: {args_dict['offsets']}")

    # Perform interpolation for the given query point
    interpolation_handler = InterpolationHandler(evaluator_instances, metrics=metrics)
    query_points = np.array([args_dict["query_point"]])
    scaled_events = interpolation_handler.interpolate(query_points)

    output_data = {
        "query_point": args_dict["query_point"],
        "scaled_events": scaled_events.tolist(),
    }
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)
    logger.info(f"Output saved to {output_filepath}")
    logger.info(f"Scaled events for grid point {args_dict['query_point']}: {scaled_events}")


if __name__ == "__main__":
    main()
