#!/usr/bin/python3

r"""
Configure a simulation based on command-line arguments.

This application configures and
generates simulation parameters for a specific grid point in a statistical uncertainty
evaluation setup.

Command line arguments
----------------------
azimuth (float, required)
    Azimuth angle in degrees.
elevation (float, required)
    Elevation angle in degrees.
nsb (float, required)
    Night sky background value.
ctao_data_level (str, required)
    The data level for the simulation (e.g., 'A', 'B', 'C').
science_case (str, required)
    The science case for the simulation.
file_path (str, required)
    Path to file with MC events at CTAO DL2 data level. Used for statistical uncertainty evaluation.
file_type (str, required)
    Type of the dl2_mc_events_file file ('point-like' or 'cone').
metrics (str, optional)
    Path to a YAML file containing metrics for evaluation.
site (str, required)
    The observatory site (North or South).

Example
-------
To run the simulation configuration, execute the script as follows:

.. code-block:: console

    simtools-production-generate-simulation-config --azimuth 60.0 --elevation 45.0 \
        --nsb 0.3 --ctao_data_level "A" --science_case "high_precision" \
        --file_path tests/resources/production_dl2_fits/dl2_mc_events_file.fits \
        --file_type "point-like"    \
        --metrics_file tests/resources/production_simulation_config_metrics.yaml --site North

The output will show the configured simulation parameters.
"""

import json
import logging
from pathlib import Path

import astropy.units as u

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.io_operations import io_handler
from simtools.production_configuration.generate_simulation_config import (
    SimulationConfig,
)
from simtools.production_configuration.production_configuration_helper_functions import load_metrics


def _parse(label):
    """Parse command-line arguments."""
    config = configurator.Configurator(
        label=label, description="Configure and run a simulation based on input parameters."
    )
    config.parser.add_argument(
        "--azimuth", type=float, required=True, help="Azimuth angle in degrees."
    )
    config.parser.add_argument(
        "--elevation", type=float, required=True, help="Elevation angle in degrees."
    )
    config.parser.add_argument(
        "--nsb", type=float, required=True, help="Night sky background in units of 1/(sr*ns*cm**2)."
    )
    config.parser.add_argument(
        "--ctao_data_level", type=str, required=True, help="Data level (e.g., 'A', 'B', 'C')."
    )
    config.parser.add_argument(
        "--science_case", type=str, required=True, help="Science case for the simulation."
    )
    config.parser.add_argument(
        "--file_path", type=str, required=True, help="Path to MC event file in DL2 format."
    )
    config.parser.add_argument(
        "--file_type",
        type=str,
        required=True,
        help="Type of the dl2_mc_events_file file ('point-like' or 'cone').",
    )
    config.parser.add_argument(
        "--metrics_file",
        required=True,
        type=str,
        help="Path to YAML file containing metrics and required precision as values.",
    )
    config.parser.add_argument(
        "--site",
        type=str,
        required=True,
        help="Site location for the simulation (e.g., 'North', 'South').",
    )
    config.parser.add_argument(
        "--output_file",
        help="Name of the file to save the configured simulation parameters.",
        type=str,
        required=False,
        default="configured_simulation_params.json",
    )

    return config.initialize(db_config=False)


def main():
    """Run the Simulation Config application."""
    label = Path(__file__).stem
    args_dict, _ = _parse(label)

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))
    output_path = io_handler.IOHandler().get_output_directory(label)
    output_filepath = Path(output_path).joinpath(f"{args_dict['output_file']}")

    grid_point_config = {
        "azimuth": args_dict["azimuth"],
        "elevation": args_dict["elevation"],
        "night_sky_background": args_dict["nsb"],
    }

    metrics = load_metrics(args_dict["metrics_file"]) if "metrics_file" in args_dict else {}

    simulation_config = SimulationConfig(
        grid_point=grid_point_config,
        ctao_data_level=args_dict["ctao_data_level"],
        science_case=args_dict["science_case"],
        file_path=args_dict["file_path"],
        file_type=args_dict["file_type"],
        metrics=metrics,
    )

    simulation_params = simulation_config.configure_simulation()

    serializable_config = {}

    for key, value in simulation_params.items():
        if isinstance(value, u.Quantity):
            serializable_config[key] = f"{value.value} {value.unit}"
        else:
            serializable_config[key] = value

    logger.info(f"Simulation configuration: {serializable_config}")
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(serializable_config, f, indent=4)
    logger.info(f"Simulation configuration saved to: {output_filepath}")


if __name__ == "__main__":
    main()
