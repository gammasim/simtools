#!/usr/bin/env python3

r"""Derive and plot bias curves from NSB and proton trigger rates.

Description
-----------

This application combines NSB (Night Sky Background) and proton trigger rates
to generate bias curves showing how trigger rates vary with threshold.

The tool:
1. Extracts NSB trigger rates from SIMTEL log files
2. Calculates proton trigger rates from simulation HDF5 files
3. Plots both curves on the same figure for comparison

The input directory should contain both:
- NSB log files or log_hist archives
- Proton simulation HDF5 files

Command line arguments
----------------------

data_dir (str, required)
    Directory containing both NSB logs/log_hist archives and proton simulation files.
output (str, optional)
    Output plot file path or output directory. Default: bias_curve.png
nsb_output (str, optional)
    Output ECSV table file for NSB trigger rates. If not specified, no table is written.
proton_output (str, optional)
    Output ECSV table file for proton rates. If not specified, no table is written.
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

.. code-block:: console

    simtools-derive-bias-curves \\
        --data_dir /path/to/data \\
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
        "--data_dir",
        type=Path,
        required=True,
        help="Directory containing both NSB logs/log_hist archives and proton simulation files.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("bias_curve.png"),
        help="Output plot file path or output directory. Default: bias_curve.png",
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

    generate_bias_curves(app_context.args)


if __name__ == "__main__":
    main()
