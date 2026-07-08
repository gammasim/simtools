#!/usr/bin/python3

r"""
Build trigger-histogram products from broad-range simulations.

This application reads reduced event-data files from broad-range simulations,
accumulates triggered and simulated histograms in angular-distance vs energy, and writes
an HDF5 product for later statistics estimation.
"""

import astropy.units as u

from simtools.application_control import build_application
from simtools.production_configuration.trigger_histograms import build_trigger_histograms


def _add_arguments(parser):
    """Application-specific command line arguments."""
    parser.add_argument(
        "--event_data_file",
        help=(
            "Event data file or glob pattern. Provide one or more patterns to build "
            "histograms for multiple productions."
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
        "--angular_distance_bin_width",
        help="Angular-distance bin width. The range is taken from broad-range viewcone limits.",
        type=parser.positive_quantity("deg"),
        default=0.1 * u.deg,
    )
    parser.add_argument(
        "--plot_histograms",
        help="Write diagnostic triggered-event histograms for the built histograms.",
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
    """Run the trigger-histogram builder CLI application."""
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
    build_trigger_histograms(app_context.args)


if __name__ == "__main__":
    main()
