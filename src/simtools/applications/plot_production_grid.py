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
        --plot_ra_dec_tracks

Output
------
The output figure shows local Alt/Az (polar projection). The equatorial
RA/Dec panel is added when RA/Dec columns are available in the grid file.
"""

from simtools.application_control import build_application
from simtools.visualization.plot_production_grid import (
    PLOT_VALUE_KEYS,
    ProductionGridPlotter,
    azimuth_zenith_output_file_stem,
    zenith_profile_output_file_stem,
)


def _add_arguments(parser):
    """Register application-specific command line arguments."""
    parser.add_argument(
        "--grid_points_file",
        type=str,
        required=True,
        help="Path to the ECSV file containing grid points.",
    )
    parser.add_argument(
        "--plot_ra_dec_tracks",
        action="store_true",
        default=False,
        help="Plot manual or inferred RA/Dec guide tracks on the sky projection.",
    )
    parser.add_argument(
        "--dec_values",
        nargs="+",
        type=float,
        default=None,
        help="Optional list of declination values in degrees to plot as manual tracks.",
    )


def main():
    """Run the ProductionGridPlotter."""
    app_context = build_application(initialization_kwargs={"db_config": False, "output": True})

    plotter = ProductionGridPlotter(
        grid_points_file=app_context.args["grid_points_file"],
        output_path=app_context.io_handler.get_output_directory(),
    )

    plotter.plot_sky_projection(
        plot_ra_dec_tracks=app_context.args["plot_ra_dec_tracks"],
        dec_values=app_context.args["dec_values"],
    )
    for value_key in PLOT_VALUE_KEYS:
        plotter.plot_azimuth_zenith_projection_with_color_scale(
            value_key=value_key,
            value_label=value_key,
            output_file_stem=azimuth_zenith_output_file_stem(value_key),
        )
        plotter.plot_zenith_limits_for_azimuths(
            value_key=value_key,
            value_label=value_key,
            output_file_stem=zenith_profile_output_file_stem(value_key),
        )


if __name__ == "__main__":
    main()
