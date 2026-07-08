#!/usr/bin/python3

r"""
Estimate requited Monte Carlo statistics (thrown events) from a histograms of triggered events.

This application loads a trigger-histogram file, evaluates a toy thrown-event
distribution for a configurable power-law spectrum, and computes the total Monte Carlo event
statistics required to meet a target relative statistical uncertainty.
"""

from simtools.application_control import build_application
from simtools.production_configuration.monte_carlo_statistics_estimator import (
    estimate_monte_carlo_statistics,
)


def _add_arguments(parser):
    """Register application-specific command line arguments."""
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Path to the trigger-histogram file.",
    )
    parser.add_argument(
        "--array_names",
        nargs="+",
        default=None,
        type=str,
        help="Optional list of array names to estimate.",
    )
    parser.add_argument(
        "--spectral_index",
        required=False,
        type=float,
        default=-2.0,
        help="Power-law spectral index assumed.",
    )
    parser.add_argument(
        "--target_relative_uncertainty",
        required=True,
        type=float,
        help="Target relative statistical uncertainty per relevant bin.",
    )
    parser.add_argument(
        "--thrown_energy_min",
        required=False,
        type=parser.positive_quantity("TeV"),
        default=None,
        help="Optional lower bound of the toy thrown spectrum.",
    )
    parser.add_argument(
        "--thrown_energy_max",
        required=False,
        type=parser.positive_quantity("TeV"),
        default=None,
        help="Optional upper bound of the toy thrown spectrum.",
    )
    parser.add_argument(
        "--optimization_energy_min",
        required=False,
        type=parser.positive_quantity("TeV"),
        default=None,
        help="Optional lower bound of the optimization range.",
    )
    parser.add_argument(
        "--optimization_energy_max",
        required=False,
        type=parser.positive_quantity("TeV"),
        default=None,
        help="Optional upper bound of the optimization range.",
    )
    parser.add_argument(
        "--reduced_core_radius",
        required=False,
        type=parser.positive_quantity("m"),
        default=None,
        help="Optional reduced core scatter radius used for effective-area reporting.",
    )


def main():
    """Run the Monte Carlo statistics estimator CLI application."""
    app_context = build_application(
        initialization_kwargs={"db_config": False, "output": True},
    )
    app_context.args["output_file"] = str(
        app_context.io_handler.get_output_file(app_context.args["output_file"])
    )
    estimate_monte_carlo_statistics(app_context.args)


if __name__ == "__main__":
    main()
