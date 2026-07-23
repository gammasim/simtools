#!/usr/bin/python3

r"""
Estimate required Monte Carlo statistics (thrown events) from histograms of triggered events.

This application loads a trigger-histogram file, evaluates a toy MC-event
distribution for a configurable power-law spectrum, and computes the total Monte Carlo event
statistics required to meet either a target relative statistical uncertainty or a target number
of triggered events.

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

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.configuration.argument_helpers import (
    positive_quantity,
    scientific_int,
)
from simtools.production_configuration.monte_carlo_statistics_estimator import (
    estimate_monte_carlo_statistics,
)

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "trigger_histogram_file",
        required=True,
        type=str,
        dest="trigger_histogram_file",
        help="Path to the trigger-histogram file.",
    ),
    cli.ArgumentDefinition(
        "array_layout_name",
        help=(
            "Optional array layout name(s) to select from a precomputed trigger-histogram "
            "file. If omitted, derive limits for all layouts available in the file."
        ),
        nargs="+",
        type=str,
        required=False,
        default=None,
    ),
    cli.ArgumentDefinition(
        "spectral_index",
        required=False,
        type=float,
        default=None,
        help=(
            "Target power-law spectral index. Reweight the simulated energy distribution "
            "from the source spectrum to this target spectrum."
        ),
    ),
    cli.ArgumentDefinition(
        "target_relative_uncertainty",
        exclusive_group="target group",
        exclusive_group_required=True,
        type=float,
        help="Target relative statistical uncertainty for each relevant bin individually.",
    ),
    cli.ArgumentDefinition(
        "target_triggered_events",
        exclusive_group="target group",
        exclusive_group_required=True,
        type=scientific_int,
        help="Target total number of triggered events across the selected optimization range.",
    ),
    cli.ArgumentDefinition(
        "optimization_energy_min",
        required=False,
        type=positive_quantity("TeV"),
        default=None,
        help="Optional lower bound of the optimization range.",
    ),
    cli.ArgumentDefinition(
        "optimization_energy_max",
        required=False,
        type=positive_quantity("TeV"),
        default=None,
        help="Optional upper bound of the optimization range.",
    ),
    cli.ArgumentDefinition(
        "reduced_core_radius",
        required=False,
        type=positive_quantity("m"),
        default=None,
        help=(
            "Optional reduced core scatter radius used for effective-area reporting "
            "(e.g., as derived from simtools-production-derive-corsika-limits)."
        ),
    ),
    cli.ArgumentDefinition(
        "reduced_view_cone_radius",
        required=False,
        type=positive_quantity("deg"),
        default=None,
        help=(
            "Optional reduced view-cone radius used for reporting required Monte Carlo "
            "statistics within a smaller angular cone."
        ),
    ),
    cli.ArgumentDefinition(
        "plot_diagnostics",
        help="Write diagnostic plots for expected events and relative uncertainty.",
        action="store_true",
        default=False,
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        *cli.PATH_ARGUMENTS,
        *cli.OUTPUT_ARGUMENTS,
    ),
    initialize_output=True,
)


def main():
    """Run the Monte Carlo statistics estimator CLI application."""
    APPLICATION.start()
    estimate_monte_carlo_statistics()


if __name__ == "__main__":
    main()
