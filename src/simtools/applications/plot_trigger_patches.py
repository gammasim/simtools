"""Inspect camera trigger groups with a compact simulation report."""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.configuration.argument_helpers import bounded_int
from simtools.model.model_utils import read_overwrite_model_parameter_dict
from simtools.model.site_model import SiteModel
from simtools.model.telescope_model import TelescopeModel
from simtools.simtel.trigger_patch_mapping import run_trigger_patch_mapping

ACTION_STORE_TRUE = "store_true"


EVENTS_PER_COMBINATION = "events_per_combination"
INCLUDE_PRESUM_SLAVES = "include_presum_slaves"
MODEL_VERSION = "model_version"
OUTPUT_NAME = "output_name"
PHOTONS_PER_PIXEL = "photons_per_pixel"
PLOT_TRIGGER_PATCHES = "plot_trigger_patches"
REQUIRED_PIXELS = "required_pixels"
RUN_NUMBER = "run_number"
MAX_EVENTS = "max_events"
MAX_PLOTTED_EVENTS = "max_plotted_events"
MAX_PIXEL_RADIUS = "max_pixel_radius"
EXHAUSTIVE = "exhaustive"

_ARGUMENTS = (
    cli.ArgumentDefinition(
        MAX_PLOTTED_EVENTS,
        type=bounded_int(0, 2147483647),
        default=4,
        help="Maximum event pages in simulation order; 0 writes only the summary plots.",
    ),
    cli.ArgumentDefinition(
        MAX_EVENTS,
        type=bounded_int(1, 2147483647),
        default=None,
        help="Maximum generated trials, including repetitions; unselected groups remain reported.",
    ),
    cli.ArgumentDefinition(
        MAX_PIXEL_RADIUS,
        type=float,
        default=None,
        help="Select illuminated pixels within this camera-plane radius [m].",
    ),
    cli.ArgumentDefinition(
        EXHAUSTIVE,
        action=ACTION_STORE_TRUE,
        help="Enumerate all eligible combinations; default is one deterministic pattern per group.",
    ),
    cli.ArgumentDefinition(
        OUTPUT_NAME,
        type=str,
        default=None,
        help="PDF file name; defaults to trigger-patches-<telescope>-run<run>.pdf.",
    ),
    cli.ArgumentDefinition(
        REQUIRED_PIXELS,
        type=bounded_int(1, 2147483647),
        default=None,
        help="Illuminated masters; omit for whole groups. Independent of trigger_pixels.",
    ),
    cli.ArgumentDefinition(
        INCLUDE_PRESUM_SLAVES,
        action=ACTION_STORE_TRUE,
        help="Fire pre-sum slave pixels whenever their master pixel fires.",
    ),
    cli.ArgumentDefinition(
        PHOTONS_PER_PIXEL,
        type=bounded_int(1, 2147483647),
        default=500,
        help="Number of photons emitted by each selected pixled LED.",
    ),
    cli.ArgumentDefinition(
        EVENTS_PER_COMBINATION,
        type=bounded_int(1, 2147483647),
        default=1,
        help="Number of pixled events generated for each trigger combination.",
    ),
    cli.ArgumentDefinition(
        RUN_NUMBER,
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
        model_version=args[MODEL_VERSION],
        label=PLOT_TRIGGER_PATCHES,
        model_reader=app_context.model_reader,
        model_directory=output_directory,
        overwrite_model_parameter_dict=overrides,
    )
    site = SiteModel(
        site=args["site"],
        model_version=args[MODEL_VERSION],
        label=PLOT_TRIGGER_PATCHES,
        model_reader=app_context.model_reader,
        model_directory=output_directory,
        overwrite_model_parameter_dict=overrides,
    )
    files = run_trigger_patch_mapping(
        telescope,
        site,
        output_directory,
        required_pixels=args[REQUIRED_PIXELS],
        include_presum_slaves=args[INCLUDE_PRESUM_SLAVES],
        photons_per_pixel=args[PHOTONS_PER_PIXEL],
        events_per_combination=args[EVENTS_PER_COMBINATION],
        run_number=args[RUN_NUMBER],
        output_name=args[OUTPUT_NAME],
        max_events=args.get(MAX_EVENTS),
        max_pixel_radius=args.get(MAX_PIXEL_RADIUS),
        exhaustive=args.get(EXHAUSTIVE, False),
        max_plotted_events=args.get(MAX_PLOTTED_EVENTS, 4),
    )
    app_context.logger.info("Wrote trigger patch display to %s", files.pdf_file)
    app_context.logger.info("Wrote trigger patch simtel events to %s", files.simtel_file)
    return files


def main():
    """See CLI description."""
    run(APPLICATION.start())


if __name__ == "__main__":
    main()
