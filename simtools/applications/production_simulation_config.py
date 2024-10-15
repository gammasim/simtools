#!/usr/bin/python3

r"""
Configure and run a simulation based on command-line arguments.

This application configures and
generates simulation parameters for a specific grid point in a statistical error
evaluation setup.

Command line arguments
----------------------
azimuth (float, required)
    Azimuth angle in degrees.
elevation (float, required)
    Elevation angle in degrees.
nsb (float, required)
    Night sky background value.
data_level (str, required)
    The data level for the simulation (e.g., 'A', 'B', 'C').
science_case (str, required)
    The science case for the simulation.
file_path (str, required)
    Path to the FITS file used for statistical error evaluation.
file_type (str, required)
    Type of the FITS file ('On-source' or 'Offset').
metrics (str, optional)
    Path to a YAML file containing metrics for evaluation.

Example
-------
To run the simulation configuration, execute the script as follows:

.. code-block:: console

    simtools-production-simulation-config --azimuth 60.0 --elevation 45.0 \
      --nsb 0.3 --data_level "A" --science_case "high_precision" \
      --file_path "path/to/fits_file.fits" --file_type "On-source" \
      --metrics "path/to/metrics.yaml"

The output will show the configured simulation parameters.
"""

import argparse
import logging
import os

import yaml

from simtools.production_configuration.generate_simulation_config import (
    SimulationConfig,
)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Configure and run a simulation based on input parameters."
    )
    parser.add_argument("--azimuth", type=float, required=True, help="Azimuth angle in degrees.")
    parser.add_argument(
        "--elevation", type=float, required=True, help="Elevation angle in degrees."
    )
    parser.add_argument("--nsb", type=float, required=True, help="Night sky background value.")
    parser.add_argument(
        "--data_level", type=str, required=True, help="Data level (e.g., 'A', 'B', 'C')."
    )
    parser.add_argument(
        "--science_case", type=str, required=True, help="Science case for the simulation."
    )
    parser.add_argument("--file_path", type=str, required=True, help="Path to the FITS file.")
    parser.add_argument(
        "--file_type",
        type=str,
        required=True,
        help="Type of the FITS file ('On-source' or 'Offset').",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        help="Path to YAML file containing metrics and precision as values (optional).",
    )

    return parser.parse_args()


def load_metrics(file_path: str) -> dict:
    """
    Load metrics from a YAML file if provided.

    Parameters
    ----------
    file_path : str
        Path to the metrics YAML file.

    Returns
    -------
    dict
        Dictionary of metrics.

        Example:
        metrics={
            "error_eff_area": 0.02,
            "error_sig_eff_gh": 0.02,
            "error_energy_estimate_bdt_reg_tree": 0.05,
            "error_gamma_ray_psf": 0.01,
            "error_image_template_methods": 0.03,
        }
    """
    if file_path and os.path.exists(file_path):
        with open(file_path, encoding="utf-8") as file:
            return yaml.safe_load(file)
    return {}


def main():
    """Run the Simulation Config application."""
    args = parse_arguments()

    logging.basicConfig(level=logging.INFO)

    grid_point_config = {
        "azimuth": args.azimuth,
        "elevation": args.elevation,
        "night_sky_background": args.nsb,
    }

    # Load metrics if provided
    metrics = load_metrics(args.metrics) if args.metrics else {}

    config = SimulationConfig(
        grid_point=grid_point_config,
        data_level=args.data_level,
        science_case=args.science_case,
        file_path=args.file_path,
        file_type=args.file_type,
        metrics=metrics,
    )

    simulation_params = config.configure_simulation()
    print("Configured Simulation Parameters:", simulation_params)


if __name__ == "__main__":
    main()
