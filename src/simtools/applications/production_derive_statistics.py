#!/usr/bin/python3

r"""
Application to run the StatisticalErrorEvaluator and interpolate results.

This application evaluates statistical uncertainties from DL2 MC event files
based on input parameters like zenith angles and camera offsets, and performs interpolation
for a specified query point.

Command line arguments
----------------------
base_path (str, required)
    Path to the directory containing the DL2 MC event file for interpolation.
zeniths (list of int, required)
    List of zenith angles to consider.
camera_offsets (list of int, required)
    List of offsets in degrees.
query_point (list of float, required)
    Query point for interpolation. The query point must contain exactly 5 values:
        - Energy (TeV)
        - Azimuth (degrees)
        - Zenith (degrees)
        - NSB (MHz)
        - Offset (degrees)
output_file (str, optional)
    Output file to store the results. Default: 'interpolated_production_statistics.json'.
metrics_file (str, optional)
    Path to the metrics definition file. Default: 'production_simulation_config_metrics.yml'.
file_name_template (str, optional)
    Template for the file name. Default:
    'prod6_LaPalma-{zenith}deg_gamma_cone.N.Am-4LSTs09MSTs_ID0_reduced.fits'.

Example
-------
To evaluate statistical uncertainties and perform interpolation, run the command line script:

.. code-block:: console

    simtools-production-derive-statistics --base_path tests/resources/production_dl2_fits/ \\
        --zeniths 20 40 52 60 --camera_offsets 0 --query_point 1 180 30 0 0 \\
        --metrics_file "path/to/metrics.yaml" \\
        --output_path simtools-output/derived_events \\
        --output_file derived_events.json

The output will display the production statistics for the specified query point and save
 the results to the specified output file.
"""

from pathlib import Path

from simtools.configuration import configurator
from simtools.production_configuration.derive_production_statistics_handler import (
    ProductionStatisticsHandler,
)


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
        "--zeniths",
        required=True,
        nargs="+",
        type=float,
        help="List of zenith angles.",
    )
    config.parser.add_argument(
        "--camera_offsets",
        required=True,
        nargs="+",
        type=float,
        help="List of camera offsets in degrees.",
    )
    config.parser.add_argument(
        "--query_point",
        required=True,
        nargs=5,
        type=float,
        help="Grid point for interpolation (energy, azimuth, zenith, NSB, offset).",
    )
    config.parser.add_argument(
        "--metrics_file",
        required=False,
        type=str,
        default="production_simulation_config_metrics.yml",
        help="Metrics definition file. (default: production_simulation_config_metrics.yml)",
    )
    config.parser.add_argument(
        "--file_name_template",
        required=False,
        type=str,
        default=("prod6_LaPalma-{zenith}deg_gamma_cone.N.Am-4LSTs09MSTs_ID0_reduced.fits"),
        help=("Template for the DL2 MC event file name."),
    )
    return config.initialize(db_config=False, output=True)


def main():
    """Run the ProductionStatisticsHandler."""
    label = Path(__file__).stem
    args_dict, _ = _parse(
        label,
        "Evaluate statistical uncertainties from DL2 MC event files and interpolate results.",
    )

    manager = ProductionStatisticsHandler(args_dict)
    manager.run()


if __name__ == "__main__":
    main()
