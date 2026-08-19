#!/usr/bin/python3

"""Create trigger-histogram products from reduced event lists."""

import astropy.units as u

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.configuration.argument_helpers import positive_quantity
from simtools.production_configuration.trigger_histograms import write_trigger_histograms

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "event_data_files",
        help=(
            "Reduced event-data file or glob pattern. Provide one or more patterns to build "
            "histograms for multiple productions."
        ),
        nargs="+",
        action="extend",
        required=True,
    ),
    cli.ArgumentDefinition(
        "energy_bins_per_decade",
        help="Number of logarithmic energy bins per decade.",
        type=int,
        default=10,
    ),
    cli.ArgumentDefinition(
        "angular_distance_bin_width",
        help="Angular-distance bin width. The range is taken from broad-range viewcone limits.",
        type=positive_quantity("deg"),
        default=0.5 * u.deg,
    ),
    cli.ArgumentDefinition(
        "skip_invalid_event_data_files",
        help=(
            "Skip malformed or incomplete reduced event-data files inside each input "
            "pattern. By default, stop at the first invalid file."
        ),
        action="store_true",
        default=False,
    ),
    cli.ArgumentDefinition(
        "max_workers",
        help=(
            "Number of worker processes to use for execution "
            "(default: 1; set to 0 for auto-detection of available cores)."
        ),
        type=int,
        default=1,
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        *cli.BACKEND_ARGUMENTS,
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
    """Run the trigger-histogram writer CLI application."""
    app_context = APPLICATION.start()
    write_trigger_histograms(app_context.args)


if __name__ == "__main__":
    main()
