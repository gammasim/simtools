#!/usr/bin/python3

r"""
Write trigger-histogram products from reduced event lists.

This application reads reduced event-data files, accumulates the common simulated and
triggered-event histogram set, and writes a HDF5 histogram file for e.g.,
plotting, CORSIKA-limit derivation, and Monte Carlo statistics estimation.

Example
-------
Fill trigger histograms from reduced event-data files:

.. code-block:: console

    simtools-write-trigger-histograms \
        --event_data_file simtools-output/reduced_event_data_*.hdf5 \
        --energy_bins_per_decade 10 \
        --angular_distance_bin_width 0.5 deg \
        --plot_histograms

"""

import astropy.units as u

from simtools.application_control import build_application
from simtools.production_configuration.trigger_histograms import write_trigger_histograms


def _add_arguments(parser):
    """Application-specific command line arguments."""
    parser.add_argument(
        "--event_data_file",
        help=(
            "Reduced event-data file or glob pattern. Provide one or more patterns to build "
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
        default=0.5 * u.deg,
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
    """Run the trigger-histogram writer CLI application."""
    app_context = build_application(
        initialization_kwargs={
            "db_config": True,
            "output": True,
            "simulation_model": ["site", "model_version", "layout"],
        },
    )
    write_trigger_histograms(app_context.args)


if __name__ == "__main__":
    main()
