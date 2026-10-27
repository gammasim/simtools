#!/usr/bin/python3
"""Plot gain-resolved pulse response from a simulated flasher intensity sequence."""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.simtel.pulse_response import flasher_responses, write_response

SIMTEL_FILES = "simtel_files"
ILLUMINATION = "illumination"
EVENTS_PER_LEVEL = "events_per_level"
PIXELS = "pixels"
PEDESTAL = "pedestal"
WIDTH_FRACTION = "width_fraction"
THRESHOLD_ADC = "threshold_adc"

_ARGUMENTS = (
    cli.ArgumentDefinition(
        SIMTEL_FILES,
        nargs="+",
        required=True,
        help="One file per intensity, or one ordered scan file",
    ),
    cli.ArgumentDefinition(
        ILLUMINATION,
        type=float,
        nargs="+",
        required=True,
        help="Requested photons per intensity step, in file/event order",
    ),
    cli.ArgumentDefinition(
        EVENTS_PER_LEVEL,
        type=int,
        nargs="+",
        required=True,
        help="Expected events per step; one count broadcasts to all steps",
    ),
    cli.ArgumentDefinition(
        PIXELS, type=int, nargs="+", default=[0], help="Pixel IDs to analyze independently"
    ),
    cli.ArgumentDefinition(
        PEDESTAL,
        type=float,
        help="Explicit per-sample pedestal for both gains; otherwise monitoring",
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
    arguments=(*_ARGUMENTS, cli.TELESCOPE, *cli.OUTPUT_PATH_ARGUMENTS),
    initialize_model_reader=False,
)


def main():
    """See CLI description."""
    context = APPLICATION.start()
    args = context.args
    table, waves = flasher_responses(
        args[SIMTEL_FILES],
        args[ILLUMINATION],
        args[EVENTS_PER_LEVEL],
        args["telescope"],
        args[PIXELS],
        args[WIDTH_FRACTION],
        args.get(THRESHOLD_ADC),
        args.get(PEDESTAL),
    )
    write_response(table, waves, context.io_handler.get_output_directory(), "flasher_response")


if __name__ == "__main__":
    main()
