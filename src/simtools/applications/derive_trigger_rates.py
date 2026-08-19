"""Derive cosmic-ray trigger rates for a single telescope or an array of telescopes."""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.telescope_trigger_rates import telescope_trigger_rates

_ARGUMENTS = (
    cli.TELESCOPE_CONFIG_FILE,
    cli.EVENT_DATA_FILE,
    cli.ArgumentDefinition(
        "plot_histograms",
        help="Plot histograms of the event data.",
        action="store_true",
        default=False,
    ),
    cli.ArgumentDefinition(
        "cr_spectrum",
        type=str,
        default=None,
        help=(
            "Path to a YAML file defining the cosmic-ray spectrum. Supported types: "
            "PowerLaw, LogParabola, PowerLawWithExponentialGaussian. If not given, "
            "the spectrum is selected from the CTAO spectrum library."
        ),
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        cli.MODEL_VERSION,
        cli.OVERWRITE_MODEL_PARAMETERS,
        cli.SITE,
        *cli.layout_selection_arguments(),
        *cli.OUTPUT_PATH_ARGUMENTS,
        *cli.OUTPUT_ARGUMENTS,
    ),
    database=True,
    initialize_output=True,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

    telescope_trigger_rates(app_context.args)


if __name__ == "__main__":
    main()
