#!/usr/bin/python3

r"""
Derive the required number of events for MC productions for a grid of observational conditions.

This application evaluates statistical uncertainties from the analysis of events after the
application of loose gamma/hadron separation cuts, then interpolates the derived number of required
events for the specified grid points provided in a file. The resulting grid points will have the
derived number of required events added.

The metric for the required uncertainty is pre-defined and must be configured via the metrics file.

Command line arguments
----------------------
grid_points_production_file (str, required)
    Path to the file containing grid points. Each grid point should include azimuth, zenith, NSB,
    offset.
metrics_file (str, optional)
    Path to the metrics definition file. Default: 'production_simulation_config_metrics.yml'.
base_path (str, required)
    Path to the directory containing the event files for interpolation (after loose gamma/hadron
    cuts).
file_name_template (str, optional)
    Template for the event file name. Default:
    'prod6_LaPalma-{zenith}deg_gamma_cone.N.Am-4LSTs09MSTs_ID0_reduced.fits'.
plot_production_statistics (flag, optional)
    If provided, plots the production statistics. Default: False.

Example
-------
To evaluate statistical uncertainties and perform interpolation, run the command line script:

.. code-block:: console

    simtools-production-derive-statistics \\
        --grid_points_production_file path/to/grid_points_production.json \\
        --metrics_file "path/to/metrics.yaml" \\
        --base_path path/to/production_event_files/ \\
        --file_name_template "prod6_LaPalma-{zenith}deg\\
            _gamma_cone.N.Am-4LSTs09MSTs_ID0_reduced.fits" \\
        --zeniths 20 40 52 60 \\
        --offsets 0 \\
        --azimuths 180 \\
        --nsb 0.0 \\
        --plot_production_statistics

Output
------
The output will be a file containing the grid points with the derived number of required events
added.
"""

from simtools.application_control import get_application_label, startup_application
from simtools.configuration import configurator
from simtools.production_configuration.derive_production_statistics_handler import (
    ProductionStatisticsHandler,
)


def _parse():
    """Parse command line arguments."""
    config = configurator.Configurator(
        label=get_application_label(__file__),
        description=(
            "Evaluate statistical uncertainties from DL2 MC event files and interpolate results."
        ),
    )

    config.parser.add_argument(
        "--grid_points_production_file",
        type=str,
        required=True,
        help="Path to the JSON file containing grid points for a production.",
    )
    config.parser.add_argument(
        "--metrics_file",
        required=True,
        type=str,
        default=None,
        help="Metrics definition file. (default: production_simulation_config_metrics.yml)",
    )
    config.parser.add_argument(
        "--base_path",
        type=str,
        required=True,
        help="Path to the DL2 MC event files for interpolation.",
    )
    config.parser.add_argument(
        "--file_name_template",
        required=False,
        type=str,
        default=("prod6_LaPalma-{zenith}deg_gamma_cone.N.Am-4LSTs09MSTs_ID0_reduced.fits"),
        help=("Template for the DL2 MC event file name."),
    )
    config.parser.add_argument(
        "--zeniths",
        required=True,
        nargs="+",
        type=float,
        help="List of zenith angles in deg that describe the supplied DL2 files.",
    )
    config.parser.add_argument(
        "--azimuths",
        required=True,
        nargs="+",
        type=float,
        help="List of azimuth angles in deg that describe the supplied DL2 files.",
    )
    config.parser.add_argument(
        "--nsb",
        required=True,
        nargs="+",
        type=float,
        help="List of nsb values that describe the supplied DL2 files.",
    )
    config.parser.add_argument(
        "--offsets",
        required=True,
        nargs="+",
        type=float,
        help="List of camera offsets in deg that describe the supplied DL2 files.",
    )
    config.parser.add_argument(
        "--plot_production_statistics",
        required=False,
        action="store_true",
        default=False,
        help="Plot production statistics.",
    )

    return config.initialize(db_config=False, output=True)


def main():
    """Run the ProductionStatisticsHandler."""
    args_dict, _, _, _io_handler = startup_application(_parse)

    manager = ProductionStatisticsHandler(args_dict, output_path=_io_handler.get_output_directory())
    manager.run()


if __name__ == "__main__":
    main()
