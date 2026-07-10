#!/usr/bin/python3

r"""
Estimate required Monte Carlo statistics (thrown events) from histograms of triggered events.

This application loads a trigger-histogram file, evaluates a toy MC-event
distribution for a configurable power-law spectrum, and computes the total Monte Carlo event
statistics required to meet a target relative statistical uncertainty.

Statistical uncertainties are estimated from the expected number of triggered events per bin
in the histogram, using the axes: energy vs. angular distance.
The derived Monte Carlo statistics is reported when all bins in the energy range
``--optimization_energy_min`` to ``--optimization_energy_max`` have a relative uncertainty
below the target value.

Example
-------
Estimate Monte Carlo statistics from a trigger-histogram file:

.. code-block:: console

    simtools-production-derive-monte-carlo-statistics \
        --trigger_histogram_file trigger_histograms.hdf5 \
        --spectral_index -2.0 \
        --target_relative_uncertainty 0.1 \
        --plot_diagnostics

"""

from simtools.application_control import build_application
from simtools.production_configuration.monte_carlo_statistics_estimator import (
    estimate_monte_carlo_statistics,
)


def _add_arguments(parser):
    """Register application-specific command line arguments."""
    parser.add_argument(
        "--trigger_histogram_file",
        required=True,
        type=str,
        dest="trigger_histogram_file",
        help="Path to the trigger-histogram file.",
    )
    parser.add_argument(
        "--array_layout_name",
        help=(
            "Optional array layout name(s) to select from a precomputed trigger-histogram "
            "file. If omitted, derive limits for all layouts available in the file."
        ),
        nargs="+",
        type=str,
        required=False,
        default=None,
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
        help=(
            "Optional reduced core scatter radius used for effective-area reporting "
            "(e.g., as derived from simtools-production-derive-corsika-limits)."
        ),
    )
    parser.add_argument(
        "--plot_diagnostics",
        help="Write diagnostic plots for expected events and relative uncertainty.",
        action="store_true",
        default=False,
    )


def main():
    """Run the Monte Carlo statistics estimator CLI application."""
    build_application(initialization_kwargs={"db_config": False, "output": True})
    estimate_monte_carlo_statistics()


if __name__ == "__main__":
    main()
