#!/usr/bin/python3

r"""
Plot production grid points on sky coordinate projections.

This application visualizes production grid points on a polar sky projection using
altitude / azimuth coordinates. The input is an ECSV production-grid table generated
by ``simtools-production-generate-grid``.

Command line arguments
----------------------
grid_points_file (str, required)
    Path to the ECSV file containing grid points.
site (str, required)
    Observatory site name used to read the reference coordinates from the site model.
model_version (str, required)
    Model version used to read the site reference coordinates.
observation_time (str, optional)
    Observation time in UTC ISO format used for Alt/Az <-> RA/Dec transformations.
    If omitted, the application uses ``metadata.observing_time_utc`` from the
    grid file when available.
plot_ra_dec_tracks (flag, optional)
    If provided, plot RA/Dec guide tracks on top of the sky projection. When native
    RA/Dec grid points are present (grid file contains explicit ``ra`` and ``dec``
    columns), thin grid lines are inferred automatically.
    Default: False.
dec_values (list of float, optional)
    Optional list of declination values in degrees to plot as manual tracks. If not
    provided, tracks are inferred from native RA/Dec grid points when possible.
    Default: None.

Example
-------
To plot grid points on a sky projection:

.. code-block:: console

    simtools-plot-production-grid \
        --grid_points_file path/to/grid_points_production.ecsv \
        --site North \
        --model_version 6.0.2 \
        --observation_time "2025-06-01 00:00:00" \
        --plot_ra_dec_tracks

Output
------
The output figure shows both local Alt/Az (polar projection) and equatorial
RA/Dec panels.
"""

import logging

from simtools.application_control import get_application_label, startup_application
from simtools.configuration import configurator
from simtools.model.site_model import SiteModel
from simtools.production_configuration.plot_production_grid import ProductionGridPlotter

logger = logging.getLogger(__name__)


def _parse():
    """Parse command line arguments."""
    config = configurator.Configurator(
        label=get_application_label(__file__),
        description="Plot production grid points on sky coordinate projections.",
    )

    config.parser.add_argument(
        "--grid_points_file",
        type=str,
        required=True,
        help="Path to the ECSV file containing grid points.",
    )
    config.parser.add_argument(
        "--observation_time",
        type=str,
        default=None,
        help=(
            "Observation time in UTC ISO format for coordinate transforms. "
            "If not provided, uses observing time stored in the grid file metadata when present."
        ),
    )
    config.parser.add_argument(
        "--plot_ra_dec_tracks",
        action="store_true",
        default=False,
        help="Plot manual or inferred RA/Dec guide tracks on the sky projection.",
    )
    config.parser.add_argument(
        "--dec_values",
        nargs="+",
        type=float,
        default=None,
        help="Optional list of declination values in degrees to plot as manual tracks.",
    )

    return config.initialize(
        db_config=True,
        output=True,
        simulation_model=["version", "site", "model_version"],
    )


def main():
    """Run the ProductionGridPlotter."""
    app_context = startup_application(_parse)
    site_model = SiteModel(
        model_version=app_context.args["model_version"],
        site=app_context.args["site"],
    )

    plotter = ProductionGridPlotter(
        grid_points_file=app_context.args["grid_points_file"],
        site_location_lat=site_model.get_parameter_value_with_unit("reference_point_latitude"),
        site_location_lon=site_model.get_parameter_value_with_unit("reference_point_longitude"),
        site_location_height=site_model.get_parameter_value_with_unit("reference_point_altitude"),
        observation_time=app_context.args["observation_time"],
        output_path=app_context.io_handler.get_output_directory(),
    )

    plotter.plot_sky_projection(
        plot_ra_dec_tracks=app_context.args["plot_ra_dec_tracks"],
        dec_values=app_context.args["dec_values"],
    )


if __name__ == "__main__":
    main()
