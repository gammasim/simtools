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
        "azimuth_center",
        type=float,
        default=180.0,
        help="Azimuth-bin center in degrees for the selected shower-statistics plot.",
    ),
    cli.ArgumentDefinition(
        "azimuth_width",
        type=float,
        default=1.0,
        help="Azimuth-bin width in degrees for the selected shower-statistics plot.",
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        *cli.OUTPUT_PATH_ARGUMENTS,
    ),
    needs=("corsika",),
)


def main():
    """Run the ProductionGridPlotter."""
    app_context = APPLICATION.start()
    output_path = app_context.paths.get_output_directory()

    plotter = ProductionGridPlotter(
        grid_points_file=app_context.args["grid_points_file"],
        output_path=output_path,
    )

    plotter.plot_sky_projection()
    plotter.plot_limit_projections()
    plotter.plot_shower_statistics_vs_zenith(
        azimuth_center=app_context.args["azimuth_center"],
        azimuth_width=app_context.args["azimuth_width"],
    )


if __name__ == "__main__":
    main()
