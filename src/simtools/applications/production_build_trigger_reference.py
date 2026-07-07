#!/usr/bin/python3

r"""
Build trigger-statistics reference products from broad-range simulations.

This application reads reduced event-data files produced from broad-range simulations,
accumulates triggered and simulated histograms in angular-distance vs energy, and writes
an HDF5 reference product for later statistics estimation.
"""

from simtools.application_control import build_application
from simtools.production_configuration.trigger_statistics_reference import (
    build_trigger_statistics_reference,
)


def _add_arguments(parser):
    """Register application-specific command line arguments."""
    parser.add_argument(
        "--event_data_file",
        help=(
            "Event data file or glob pattern. Provide one or more patterns to build "
            "references for multiple productions."
        ),
        nargs="+",
        action="extend",
        required=True,
    )
    parser.add_argument(
        "--energy_bins_per_decade",
        help="Number of logarithmic energy bins per decade.",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--angular_distance_bin_count",
        help="Number of angular-distance bins.",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--plot_histograms",
        help="Write diagnostic triggered-event histograms for the built references.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--skip_invalid_event_data_files",
        help=(
            "Skip malformed or incomplete reduced event-data files inside each input pattern. "
            "By default, the application stops at the first invalid file."
        ),
        action="store_true",
        default=False,
    )


def main():
    """
    Run the trigger-reference builder CLI application.

    Returns
    -------
    None
    """
    app_context = build_application(
        initialization_kwargs={
            "db_config": True,
            "output": True,
            "simulation_model": [
                "site",
                "model_version",
                "layout",
            ],
        },
    )
    build_trigger_statistics_reference(app_context.args)


if __name__ == "__main__":
    main()
