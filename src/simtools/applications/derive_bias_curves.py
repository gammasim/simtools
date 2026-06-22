#!/usr/bin/env python3

r"""Derive and plot bias curves from NSB and proton trigger rates.

Description
-----------

This application combines NSB (Night Sky Background) and proton trigger rates
to generate bias curves showing how trigger rates vary with threshold.

The tool:
1. Extracts NSB trigger rates from SIMTEL log files
2. Calculates proton trigger rates from simulation HDF5 files organized by threshold
3. Plots both curves on the same figure for comparison

The input directory structure should contain:
- NSB log files: **/*.simtel.log.gz with threshold information in path
- Proton sim files: threshold subdirectories (e.g., 220/, 230/) containing *.hdf5 files

Command line arguments
----------------------

root_dir (str, optional)
    Root directory containing both NSB logs and proton simulation files.
    Can be overridden by --nsb_dir and --proton_dir.
nsb_dir (str, optional)
    Directory containing NSB log files. If not specified, uses --root_dir.
proton_dir (str, optional)
    Directory containing proton simulation files. If not specified, uses --root_dir.
output (str, optional)
    Output plot file path. Default: bias_curve.png
nsb_table_output (str, optional)
    Output ECSV table file for NSB trigger rates. If not specified, no table is written.
site (str, required)
    Site name (North/South) for telescope configuration.
model_version (str, required)
    Model version for telescope configuration.
array_layout_name (str, optional)
    Array layout name for telescope configuration (alternative to telescope_ids).
telescope_ids (str, optional)
    Path to telescope configuration file (alternative to array_layout_name).
title (str, optional)
    Plot title. Default: "Trigger Rate Bias Curves"
ymin (float, optional)
    Minimum y-axis value for plot. Default: 1e2
ymax (float, optional)
    Maximum y-axis value for plot. Default: 5e5



Example
-------

Generate bias curves with data in same directory:

.. code-block:: console

    simtools-derive-bias-curves \\
        --root_dir /path/to/data \\
        --site North \\
        --model_version 7.0.0 \\
        --array_layout_name LSTN-01 \\
        --output bias_curves.png

Generate bias curves with NSB and proton data in separate directories:

.. code-block:: console

    simtools-derive-bias-curves \\
        --nsb_dir /path/to/nsb/data \\
        --proton_dir /path/to/proton/data \\
        --site North \\
        --model_version 7.0.0 \\
        --array_layout_name LSTN-01 \\
        --output bias_curves.png

"""

from pathlib import Path

from simtools.application_control import build_application
from simtools.simtel.bias_curve_generator import generate_bias_curves


def _add_arguments(parser):
    """Register application-specific command line arguments."""
    parser.add_argument(
        "--root_dir",
        type=Path,
        required=False,
        help="Root directory containing both NSB logs and proton simulation files. "
        "Can be overridden by --nsb_dir and --proton_dir.",
    )

    parser.add_argument(
        "--nsb_dir",
        type=Path,
        required=False,
        help="Directory containing NSB log files. If not specified, uses --root_dir.",
    )

    parser.add_argument(
        "--proton_dir",
        type=Path,
        required=False,
        help="Directory containing proton simulation files. If not specified, uses --root_dir.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("bias_curve.png"),
        help="Output plot file path. Default: bias_curve.png",
    )

    parser.add_argument(
        "--nsb_output",
        type=Path,
        required=False,
        help="Output ECSV table file for NSB trigger rates. If not specified, no table is written.",
    )

    parser.add_argument(
        "--proton_output",
        type=Path,
        required=False,
        help="Output ECSV table file for proton rates. If not specified, no table is written.",
    )

    parser.initialize_application_arguments(["telescope_ids"])

    parser.add_argument(
        "--title",
        type=str,
        default="Trigger Rate Bias Curves",
        help="Plot title.",
    )

    parser.add_argument(
        "--ymin",
        type=float,
        default=1e2,
        help="Minimum y-axis value. Default: 1e2",
    )

    parser.add_argument(
        "--ymax",
        type=float,
        default=5e5,
        help="Maximum y-axis value. Default: 5e5",
    )


def main():
    """See CLI description."""
    app_context = build_application(
        initialization_kwargs={
            "db_config": True,
            "simulation_model": [
                "site",
                "model_version",
                "layout",
            ],
        },
    )

    # Validate directory arguments
    args = app_context.args
    if not args.get("root_dir") and not (args.get("nsb_dir") and args.get("proton_dir")):
        raise ValueError("Must specify either --root_dir or both --nsb_dir and --proton_dir")

    # Set defaults: use specific dirs if provided, otherwise fall back to root_dir
    if not args.get("nsb_dir"):
        args["nsb_dir"] = args["root_dir"]
    if not args.get("proton_dir"):
        args["proton_dir"] = args["root_dir"]

    generate_bias_curves(args)


if __name__ == "__main__":
    main()
