#!/usr/bin/python3

r"""Derive CORSIKA limits from trigger histograms."""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.configuration.argument_helpers import efficiency_interval
from simtools.production_configuration.derive_corsika_limits import (
    generate_corsika_limits_grid,
)

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "trigger_histogram_file",
        help="Precomputed trigger-histogram HDF5 file from simtools-write-trigger-histograms. ",
        type=str,
        required=True,
    ),
    cli.ArgumentDefinition(
        "array_layout_name",
        help=(
            "Optional array layout name(s) to select from a precomputed trigger-histogram "
            "file. If omitted, derive limits for all layouts available in the file."
        ),
        nargs="+",
        type=str,
        required=False,
        default=None,
    ),
    cli.ArgumentDefinition(
        "allowed_losses",
        type=str,
        required=True,
        nargs="+",
        action="extend",
        metavar="AXIS,FRACTION,MIN_EVENTS",
        help=(
            "Per-axis allowed losses as axis,fraction,min_events. Repeat for each axis "
            "using core_distance, angular_distance, or all to set both. Example: "
            "--allowed_losses core_distance,1e-6,10"
        ),
    ),
    cli.ArgumentDefinition(
        "energy_threshold_fraction",
        help="Fraction of the stable energy-peak count used to derive ERANGE ",
        type=efficiency_interval,
        required=False,
        default=0.01,
    ),
    cli.ArgumentDefinition(
        "plot_histograms",
        help="Plot histograms of the event data.",
        action="store_true",
        default=False,
    ),
    cli.ArgumentDefinition(
        "differential_loss_bins_per_decade",
        help=(
            "Number of differential energy bins per decade for per-bin limit computation. "
            "Set to 0 (default) to use integrated limits."
        ),
        type=int,
        required=False,
        default=0,
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        *cli.OUTPUT_PATH_ARGUMENTS,
        *cli.OUTPUT_ARGUMENTS,
    ),
    initialize_output=True,
)


def main():
    """See CLI description."""
    APPLICATION.start()
    generate_corsika_limits_grid()


if __name__ == "__main__":
    main()
