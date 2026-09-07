#!/usr/bin/env python3

"""Derive and plot bias curves from NSB and proton trigger rates."""

from pathlib import Path

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.simtel.bias_curve_generator import generate_bias_curves

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "data_dir",
        type=Path,
        required=True,
        help=(
            "Directory containing NSB/proton reduced event-data HDF5 files "
            "(e.g. gamma* and proton*)."
        ),
    ),
    cli.ArgumentDefinition(
        "scaling_factor",
        type=float,
        required=True,
        help=("Scaling factor to account for ions we didn't simulate"),
        default=1.35,
    ),
    cli.ArgumentDefinition(
        "figure_file",
        type=Path,
        default=Path("bias_curve.png"),
        help="Output plot file path or output directory. Default: bias_curve.png",
    ),
    cli.ArgumentDefinition(
        "nsb_table_file",
        type=Path,
        required=False,
        help="Output ECSV table file for NSB trigger rates. If not specified, no table is written.",
    ),
    cli.ArgumentDefinition(
        "proton_table_file",
        type=Path,
        required=False,
        help="Output ECSV table file for proton rates. If not specified, no table is written.",
    ),
    cli.ArgumentDefinition(
        "title",
        type=str,
        default="Trigger Rate Bias Curves",
        help="Title for the bias curve plot. Default: 'Trigger Rate Bias Curves'",
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        cli.MODEL_VERSION,
        cli.PARAMETER_VERSION(required=True),
        cli.OVERWRITE_MODEL_PARAMETERS,
        cli.SITE,
        cli.TELESCOPE,
        *cli.OUTPUT_PATH_ARGUMENTS,
    ),
    database=True,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

    generate_bias_curves(app_context.args)


if __name__ == "__main__":
    main()
