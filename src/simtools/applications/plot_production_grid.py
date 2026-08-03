#!/usr/bin/python3

"""Plot production-grid points on sky coordinate projections."""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.visualization.plot_production_grid import ProductionGridPlotter

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "grid_points_file",
        type=str,
        required=True,
        help="Path to the ECSV file containing grid points.",
    ),
    cli.ArgumentDefinition(
        "plot_ra_dec_tracks",
        action="store_true",
        default=False,
        help="Plot manual or inferred RA/Dec guide tracks on the sky projection.",
    ),
    cli.ArgumentDefinition(
        "dec_values",
        nargs="+",
        type=float,
        default=None,
        help="Optional list of declination values in degrees to plot as manual tracks.",
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        *cli.OUTPUT_PATH_ARGUMENTS,
        *cli.OUTPUT_ARGUMENTS,
    ),
    initialize_output=True,
)


def main():
    """Run the ProductionGridPlotter."""
    app_context = APPLICATION.start()

    plotter = ProductionGridPlotter(
        grid_points_file=app_context.args["grid_points_file"],
        output_path=app_context.io_handler.get_output_directory(),
    )

    plotter.plot_sky_projection(
        plot_ra_dec_tracks=app_context.args["plot_ra_dec_tracks"],
        dec_values=app_context.args["dec_values"],
    )
    plotter.plot_limit_projections()


if __name__ == "__main__":
    main()
