#!/usr/bin/python3

r"""
    Calculate the camera FoV of the telescope requested and plot the camera.

    The orientation for the plotting is "as seen for an observer facing the camera".

    An example of the camera plot can be found below.

    .. _camera_fov_plot:
    .. image:: images/validate_camera_fov_North-LST-1_pixelLayout.png
      :width: 50 %


    Command line arguments
    ----------------------
    site (str, required)
        North or South.
    telescope (str, required)
        Telescope model name (e.g. LST-1, SST-D, ...)
    model_version (str, optional)
        Model version
    camera_in_sky_coor (bool, optional)
        Plot the camera layout in sky coordinates akin to looking at it from behind for single \
         mirror telescopes
    print_pixels_id (bool, optional)
        Up to which pixel ID to print. To suppress printing of pixel IDs, set to zero\
         (--print_pixels_id 0). To print all pixels, set to 'All'."

    Example
    -------
    LST - 5.0.0

    .. code-block:: console

        simtools-validate-camera-fov --site North \\
            --telescope LSTN-01 --model_version 5.0.0

    The output is saved in simtools-output/validate_camera_fov.

    Expected final print-out message:

    .. code-block:: console

        Saved camera plot in /workdir/external/simtools/simtools-output/validate_camera_fov\\
        /application-plots/validate_camera_fov_LST-1_pixel_layout.png

"""

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
        *cli.PATH_ARGUMENTS,
    ),
    database=True,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()
    plot_camera_pixel_layout_from_args(app_context)


if __name__ == "__main__":
    main()
