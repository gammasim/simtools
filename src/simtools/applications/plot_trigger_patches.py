"""Generate sim_telarray displays for all camera trigger patches."""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.configuration.argument_helpers import bounded_int
from simtools.model.model_utils import read_overwrite_model_parameter_dict
from simtools.model.site_model import SiteModel
from simtools.model.telescope_model import TelescopeModel
from simtools.simtel.trigger_patch_mapping import run_trigger_patch_mapping

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "output_name",
        type=str,
        default=None,
        help="PostScript file name; defaults to trigger-patches-<telescope>-run<run>.ps.",
    ),
    cli.ArgumentDefinition(
        "required_pixels",
        type=bounded_int(1, 2147483647),
        default=2,
        help="Number of LEDs required in each pixled trigger combination.",
    ),
    cli.ArgumentDefinition(
        "include_presum_slaves",
        action="store_true",
        help="Fire pre-sum slave pixels whenever their master pixel fires.",
    ),
    cli.ArgumentDefinition(
        "photons_per_pixel",
        type=bounded_int(1, 2147483647),
        default=500,
        help="Number of photons emitted by each selected pixled LED.",
    ),
    cli.ArgumentDefinition(
        "events_per_combination",
        type=bounded_int(1, 2147483647),
        default=1,
        help="Number of pixled events generated for each trigger combination.",
    ),
    cli.ArgumentDefinition(
        "run_number",
        type=bounded_int(1, 2147483647),
        default=1,
        help="Run number recorded in generated LED events.",
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        cli.MODEL_VERSION(required=True, nargs=None),
        cli.OVERWRITE_MODEL_PARAMETERS,
        cli.SITE(required=True),
        cli.TELESCOPE(required=True),
        *cli.SIM_TELARRAY_PATH_ARGUMENTS,
        *cli.OUTPUT_PATH_ARGUMENTS,
    ),
    model_repository=True,
)


def run(app_context):
    """Generate trigger-patch mapping files for the requested telescope model."""
    args = app_context.args
    output_directory = app_context.io_handler.get_output_directory()
    overrides = read_overwrite_model_parameter_dict(args.get("overwrite_model_parameters"))
    telescope = TelescopeModel(
        site=args["site"],
        telescope_name=args["telescope"],
        model_version=args["model_version"],
        label="plot_trigger_patches",
        model_reader=app_context.model_reader,
        model_directory=output_directory,
        overwrite_model_parameter_dict=overrides,
    )
    site = SiteModel(
        site=args["site"],
        model_version=args["model_version"],
        label="plot_trigger_patches",
        model_reader=app_context.model_reader,
        model_directory=output_directory,
        overwrite_model_parameter_dict=overrides,
    )
    files = run_trigger_patch_mapping(
        telescope,
        site,
        output_directory,
        required_pixels=args["required_pixels"],
        include_presum_slaves=args["include_presum_slaves"],
        photons_per_pixel=args["photons_per_pixel"],
        events_per_combination=args["events_per_combination"],
        run_number=args["run_number"],
        output_name=args["output_name"],
    )
    app_context.logger.info("Wrote trigger patch display to %s", files.postscript_file)
    app_context.logger.info("Wrote trigger patch simtel events to %s", files.simtel_file)
    return files


def main():
    """See CLI description."""
    run(APPLICATION.start())


if __name__ == "__main__":
    main()
