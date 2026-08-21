#!/usr/bin/python3

"""Validate the optical model parameters through ray tracing simulations of the whole telescope."""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.ray_tracing.optics_validation import validate_optics

_ARGUMENTS = (
    cli.SOURCE_DISTANCE,
    cli.RAY_TRACING_ZENITH_ANGLE,
    cli.MAX_OFFSET,
    cli.OFFSET_STEP,
    cli.ArgumentDefinition(
        "offset_file",
        help=(
            "Path to ECSV file with x, y offset columns (in degrees). If provided, "
            "overrides max_offset and offset_step."
        ),
        type=str,
        default=None,
    ),
    cli.ArgumentDefinition(
        "offset_directions",
        help=(
            "Cardinal directions for offset generation (comma-separated): N,S,E,W. "
            "Only used with max_offset. Default: all four directions."
        ),
        type=str,
        default="N,S,E,W",
    ),
    cli.ArgumentDefinition(
        "plot_images",
        help="Produce a multiple pages pdf file with the image plots.",
        action="store_true",
    ),
    cli.ArgumentDefinition(
        "plot_images_in_degrees",
        help=(
            "When plotting PSF images, convert X/Y positions from cm to degrees "
            "using the effective focal length. Requires --plot_images."
        ),
        action="store_true",
    ),
    cli.ArgumentDefinition(
        "save_photons",
        help="Retain compressed photon list files after analysis.",
        action="store_true",
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        cli.MODEL_VERSION,
        cli.PARAMETER_VERSION,
        cli.OVERWRITE_MODEL_PARAMETERS,
        cli.SITE,
        cli.TELESCOPE,
        *cli.OUTPUT_PATH_ARGUMENTS,
        *cli.SIM_TELARRAY_PATH_ARGUMENTS,
    ),
    database=True,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()
    validate_optics(app_context)


if __name__ == "__main__":
    main()
