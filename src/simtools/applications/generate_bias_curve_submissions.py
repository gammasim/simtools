#!/usr/bin/python3

"""Generate scan grids for NSB and proton telescope trigger bias curves."""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.configuration import defaults
from simtools.configuration.argument_helpers import parse_quantity_pair
from simtools.production_configuration import parameter_scans

_ARGUMENTS = (
    cli.SITE(required=True),
    cli.MODEL_VERSION(required=True, nargs=None),
    cli.ARRAY_LAYOUT_NAME(required=True),
    cli.SIMULATION_SOFTWARE,
    cli.AZIMUTH_ANGLE(required=True, action="store", nargs=None, default=None),
    cli.ZENITH_ANGLE(required=True, action="store", nargs=None, default=None),
    cli.SHOWERS_PER_RUN(required=True),
    cli.CORE_SCATTER(required=True, default=None),
    cli.VIEW_CONE(required=True, default=None),
    cli.NUMBER_OF_RUNS(required=True),
    cli.CORSIKA_LE_INTERACTION(action="store", nargs=None, default=defaults.CORSIKA_LE_INTERACTION),
    cli.CORSIKA_HE_INTERACTION(action="store", nargs=None, default=defaults.CORSIKA_HE_INTERACTION),
    cli.CORSIKA_HADRONIC_TRANSITION_ENERGY,
    cli.ArgumentDefinition(
        "nsb_energy_range",
        help="Energy range for the NSB gamma curve.",
        type=parse_quantity_pair,
        default=parse_quantity_pair("20 MeV 25 MeV"),
    ),
    cli.ArgumentDefinition(
        "proton_energy_range",
        help="Energy range for the proton curve.",
        type=parse_quantity_pair,
        default=parse_quantity_pair("2 GeV 2000 GeV"),
    ),
    cli.ArgumentDefinition(
        "nsb_scaling_factor",
        help="NSB scaling factor used for both curves.",
        type=float,
        default=2.0,
    ),
    cli.ArgumentDefinition(
        "trigger_thresholds",
        help=(
            "Define evenly spaced trigger thresholds for both curves as "
            "MIN_THRESHOLD NUMBER_OF_THRESHOLDS STEP_SIZE."
        ),
        type=float,
        nargs=3,
        metavar=("MIN_THRESHOLD", "NUMBER_OF_THRESHOLDS", "STEP_SIZE"),
        default=None,
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    model_repository=True,
    arguments=(
        *_ARGUMENTS,
        *cli.OUTPUT_PATH_ARGUMENTS,
    ),
    needs=("model", "corsika"),
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()
    parameter_scans.generate_scan_grids(app_context.args, app_context.paths, context=app_context)


if __name__ == "__main__":
    main()
