#!/usr/bin/python3

"""Simulate the cumulative PSF and compare it with measured data."""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.ray_tracing.optics_validation import validate_cumulative_psf

_ARGUMENTS = (
    cli.SOURCE_DISTANCE,
    cli.RAY_TRACING_ZENITH_ANGLE,
    cli.ArgumentDefinition(
        "data",
        group="application",
        help="Measured cumulative PSF data file.",
        type=str,
        required=True,
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
        cli.DATA_SEARCH_PATH,
        *cli.OUTPUT_PATH_ARGUMENTS,
        *cli.SIM_TELARRAY_PATH_ARGUMENTS,
        *cli.RAY_TRACING_ARGUMENTS,
    ),
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()
    validate_cumulative_psf(app_context)


if __name__ == "__main__":
    main()
