#!/usr/bin/python3

"""Plot the camera of the requested telescope."""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.configuration.argument_helpers import bounded_int
from simtools.visualization.plot_camera import plot_camera_pixel_layout_from_args

_TRIGGER_SELECTION = "trigger selection"
_MAX_PIXEL_ID = (1 << 63) - 1

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "trigger_pixel",
        type=bounded_int(0, _MAX_PIXEL_ID),
        default=None,
        exclusive_group=_TRIGGER_SELECTION,
        help="Highlight every trigger patch containing this model pixel ID.",
    ),
    cli.ArgumentDefinition(
        "trigger_patches",
        type=int,
        nargs="+",
        default=None,
        exclusive_group=_TRIGGER_SELECTION,
        help=(
            "Highlight distinct model trigger-group IDs, with a labelled close-up. "
            "The maximum is the model's largest number of patches sharing a pixel (six for LST)."
        ),
    ),
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
    model_repository=True,
    arguments=(
        *_ARGUMENTS,
        cli.MODEL_VERSION,
        cli.OVERWRITE_MODEL_PARAMETERS,
        cli.SITE,
        cli.TELESCOPE,
        *cli.OUTPUT_PATH_ARGUMENTS,
    ),
    needs=("model", "corsika"),
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()
    plot_camera_pixel_layout_from_args(app_context, context=app_context)


if __name__ == "__main__":
    main()
