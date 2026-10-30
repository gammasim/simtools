#!/usr/bin/python3
"""Plot peak, ADC sum and width versus illumination from model pulse tables."""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.model.model_utils import initialize_simulation_models
from simtools.simtel.pulse_response import template_responses, write_response

ILLUMINATION = "illumination"
WIDTH_FRACTION = "width_fraction"
THRESHOLD_ADC = "threshold_adc"

_ARGUMENTS = (
    cli.ArgumentDefinition(
        ILLUMINATION,
        type=float,
        nargs="+",
        required=True,
        help="Illumination levels in photoelectrons",
    ),
    cli.ArgumentDefinition(
        WIDTH_FRACTION, type=float, default=0.95, help="Fraction of pulse peak defining its width"
    ),
    cli.ArgumentDefinition(
        THRESHOLD_ADC,
        type=float,
        help="Absolute width threshold above pedestal, overriding the fraction",
    ),
)

APPLICATION = ApplicationDefinition.for_module(
    __name__,
    model_repository=True,
    arguments=(
        *_ARGUMENTS,
        cli.MODEL_VERSION,
        cli.SITE,
        cli.TELESCOPE,
        cli.OVERWRITE_MODEL_PARAMETERS,
        *cli.OUTPUT_PATH_ARGUMENTS,
    ),
)


def main():
    """See CLI description."""
    context = APPLICATION.start()
    args = context.args
    model, _, _ = initialize_simulation_models(
        label=args.get("label"),
        model_version=args["model_version"],
        site=args["site"],
        telescope_name=args["telescope"],
        model_reader=context.model_reader,
    )
    table, waves = template_responses(
        model, args[ILLUMINATION], args[WIDTH_FRACTION], args.get(THRESHOLD_ADC)
    )
    write_response(table, waves, context.paths.get_output_directory(), "pulse_response")


if __name__ == "__main__":
    main()
