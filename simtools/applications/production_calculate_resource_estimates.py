#!/usr/bin/python3

r"""
Estimate the required computing resources for a simulation based on command-line arguments.

This application calculates compute and storage resources based on input parameters
such as azimuth, elevation, night sky background, and the number of events.
The user can also provide a file containing historical data for more accurate estimation.

Command line arguments
----------------------
azimuth (float, required)
    Azimuth angle in degrees.
elevation (float, required)
    Elevation angle in degrees.
nsb (float, required)
    Night sky background value.
site (str, required)
    Site name (e.g., 'South' or 'North').
number_of_events (float, required)
    Number of events for the simulation.
existing_data (str, optional)
    Path to a YAML file containing existing data for resource estimation.
lookup_file (str, optional)
    Path to the resource estimates YAML file (default: production_resource_estimates.yaml).

Example
-------
To estimate resources, run the script from the command line as follows:

.. code-block:: console

    simtools-production-calculate-resource-estimates --azimuth 60.0 --elevation 45.0 \
      --nsb 0.3 --site North --number_of_events 1e9 \
      --lookup_file tests/resources/production_resource_estimates.yaml

The output will show the estimated resources required for the simulation.
"""

import logging
import os

import yaml

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.production_configuration.derive_computing_resources import ResourceEstimator


def _parse():
    """Parse command-line arguments."""
    config = configurator.Configurator(
        description="Estimate compute and storage resources for simulations."
    )
    config.parser.add_argument(
        "--azimuth", type=float, required=True, help="Azimuth angle in degrees."
    )
    config.parser.add_argument(
        "--elevation", type=float, required=True, help="Elevation angle in degrees."
    )
    config.parser.add_argument(
        "--nsb", type=float, required=True, help="Night sky background value."
    )
    config.parser.add_argument("--site", type=str, required=True, help="South or North.")
    config.parser.add_argument(
        "--number_of_events", type=float, required=True, help="Number of events for the simulation."
    )
    config.parser.add_argument(
        "--existing_data", type=str, help="Path to YAML file containing existing data (optional)."
    )
    config.parser.add_argument(
        "--lookup_file",
        type=str,
        default="tests/resources/production_resource_estimates.yaml",
        help="Resource estimates YAML file (default: production_resource_estimates.yaml).",
    )

    return config.initialize(db_config=False)


def load_existing_data(file_path: str) -> list:
    """
    Load existing data from a YAML file if provided.

    Parameters
    ----------
    file_path : str
        Path to the existing data YAML file.

    Returns
    -------
    list
        List of dictionaries with existing data.
    """
    if file_path and os.path.exists(file_path):
        with open(file_path, encoding="utf-8") as file:
            return yaml.safe_load(file)
    return []


def main():
    """Run the Resource Estimator application."""
    args, _ = _parse()

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args["log_level"]))

    existing_data = load_existing_data(args.existing_data) if args.existing_data else []

    grid_point_config = {
        "azimuth": args.azimuth,
        "elevation": args.elevation,
        "night_sky_background": args.nsb,
    }

    simulation_params = {"number_of_events": args.number_of_events, "site": args.site}

    estimator = ResourceEstimator(
        grid_point=grid_point_config,
        simulation_params=simulation_params,
        existing_data=existing_data,
        lookup_file=args.lookup_file,
    )

    resources = estimator.estimate_resources()
    print("Estimated Resources:", resources)


if __name__ == "__main__":
    main()
