#!/usr/bin/python3

"""Calculate on-axis telescope throughput and NSB pixels rates."""

from simtools.application.definition import ApplicationDefinition
from simtools.camera.camera_efficiency_workflow import run_camera_efficiency
from simtools.configuration import arguments as cli

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "nsb_spectrum",
        help=(
            "File with NSB spectrum for the efficiency simulation. Expected format: two "
            "columns with wavelength in nm and NSB flux in [1e9 * ph/m2/s/sr/nm]. If more "
            "than two columns are present, the first and third are used and the second is "
            "ignored (native sim_telarray behavior)."
        ),
        type=str,
        default=None,
        required=False,
    ),
    cli.ArgumentDefinition(
        "skip_correction_to_nsb_spectrum",
        help=(
            "Skip correction to the NSB spectrum for the altitude difference between the "
            "reference B&E spectrum and the observation level at the CTAO sites."
        ),
        required=False,
        action="store_true",
    ),
    cli.ArgumentDefinition(
        "write_reference_nsb_rate_as_parameter",
        help="Write the NSB pixel rate obtained for reference conditions as a model parameter ",
        action="store_true",
        required=False,
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    model_repository=True,
    arguments=(
        *_ARGUMENTS,
        cli.MODEL_VERSION,
        cli.PARAMETER_VERSION,
        cli.OVERWRITE_MODEL_PARAMETERS,
        cli.SITE,
        cli.TELESCOPE,
        cli.ZENITH_ANGLE,
        cli.AZIMUTH_ANGLE,
        *cli.OUTPUT_PATH_ARGUMENTS,
    ),
)


def _validate_required_args(args_dict):
    """Validate required arguments that must be explicitly provided."""
    if args_dict["site"] is None or args_dict["telescope"] is None:
        raise RuntimeError("Site and telescope must be provided")


def main():
    """See CLI description."""
    app_context = APPLICATION.start()
    _validate_required_args(app_context.args)

    run_camera_efficiency(
        app_context.args,
        app_context.io_handler.get_output_directory(),
        label=app_context.args.get("label"),
    )


if __name__ == "__main__":
    main()
