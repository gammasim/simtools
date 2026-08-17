#!/usr/bin/python3

"""Calculate the camera FoV of the telescope requested and plot the camera."""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.visualization.plot_camera import plot_camera_pixel_layout_from_args

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "camera_in_sky_coor",
        help=(
            "Plot the camera layout in sky coordinates "
            "(akin to looking at it from behind for single-mirror telescopes)"
        ),
        action="store_true",
        default=False,
    ),
    cli.ArgumentDefinition(
        "print_pixels_id",
        help=(
            "Highest pixel ID to print. Use zero (--print_pixels_id 0) to suppress pixel "
            "IDs, or 'All' to print every pixel."
        ),
        default=50,
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        cli.MODEL_VERSION,
        cli.OVERWRITE_MODEL_PARAMETERS,
        cli.SITE,
        cli.TELESCOPE,
        *cli.OUTPUT_PATH_ARGUMENTS,
    ),
    database=True,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()
    plot_camera_pixel_layout_from_args(app_context)


if __name__ == "__main__":
    main()
