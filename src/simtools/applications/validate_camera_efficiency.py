#!/usr/bin/python3

"""Calculate on-axis telescope throughput and NSB pixels rates."""

from simtools.application.definition import ApplicationDefinition
from simtools.camera.camera_efficiency import CameraEfficiency
from simtools.configuration import arguments as cli
from simtools.io.ascii_handler import write_data_to_file
from simtools.utils import names

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
    database=True,
)


def _validate_required_args(args_dict):
    """Validate required arguments that must be explicitly provided."""
    if args_dict["site"] is None or args_dict["telescope"] is None:
        raise RuntimeError("Site and telescope must be provided")


def main():
    """See CLI description."""
    app_context = APPLICATION.start()
    _validate_required_args(app_context.args)

    results = {}
    for efficiency_type in ["Shower", "NSB", "Muon"]:
        ce = CameraEfficiency(
            label=app_context.args.get("label"),
            config_data=app_context.args,
            efficiency_type=efficiency_type,
        )
        ce.simulate()
        ce.analyze(force=True)
        results |= ce.results_summary()
        ce.plot_efficiency(save_fig=True)

        if ce.efficiency_type == "nsb":
            ce.dump_nsb_pixel_rate()
        if ce.efficiency_type == "muon":
            ce.calc_partial_efficiency()

    results_file = app_context.io_handler.get_output_directory() / names.generate_file_name(
        file_type="camera_efficiency_summary",
        suffix=".yml",
        site=app_context.args["site"],
        telescope_model_name=app_context.args["telescope"],
        zenith_angle=app_context.args["zenith_angle"].value,
        azimuth_angle=app_context.args["azimuth_angle"].value,
    )
    app_context.logger.info(f"Writing results summary to {results_file}")
    write_data_to_file(results, results_file, unique_lines=True)


if __name__ == "__main__":
    main()
