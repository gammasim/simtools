#!/usr/bin/python3

"""Derive CORSIKA limits from trigger histograms."""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.configuration.argument_helpers import efficiency_interval
from simtools.production_configuration.derive_corsika_limits import (
    generate_corsika_limits_grid,
    parse_allowed_losses,
    validate_differential_loss_bins_per_decade,
)

_OUTPUT_ARGUMENTS = (
    cli.OUTPUT_FILE,
    cli.ArgumentDefinition(
        "output_file_format",
        help=(
            "Accepted for shared configuration compatibility; this application always writes ECSV."
        ),
        type=str,
        default="ecsv",
    ),
    cli.SKIP_OUTPUT_VALIDATION,
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
            "Per-axis allowed losses as axis,fraction,min_events. Provide both "
            "core_distance and angular_distance, or use all to set both. The fraction "
            "must be in [0,1] and min_events must be non-negative. Example: "
            "--allowed_losses core_distance,1e-6,10 angular_distance,1e-6,10"
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
            "Set to 0 (default) to use integrated limits; must be non-negative."
        ),
        type=validate_differential_loss_bins_per_decade,
        required=False,
        default=0,
    ),
)


def _post_parse(args_dict, _config_sources, parser):
    """Validate the complete allowed-loss configuration after merging inputs."""
    try:
        parse_allowed_losses(args_dict.get("allowed_losses"))
    except ValueError as exc:
        parser.error(str(exc))


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        *cli.OUTPUT_PATH_ARGUMENTS,
        *_OUTPUT_ARGUMENTS,
    ),
    initialize_output=True,
    post_parse=_post_parse,
)


def main():
    """See CLI description."""
    APPLICATION.start()
    generate_corsika_limits_grid()


if __name__ == "__main__":
    main()
