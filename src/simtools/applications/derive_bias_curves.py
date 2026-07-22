#!/usr/bin/env python3

r"""Derive and plot bias curves from NSB and proton trigger rates.

Description
-----------

This application combines NSB (Night Sky Background) and proton trigger rates
to generate bias curves showing how trigger rates vary with threshold.

The tool:
1. Extracts NSB trigger rates from sim_telarray log files
2. Calculates proton trigger rates from reduced event-data HDF5 files
3. Plots both curves on the same figure for comparison
4. Outputs ecsv files for runwise nsb simulation,
runwise proton simulation, nsb rate and proton rate vs threshold

The input directory should contain both:
- NSB log files or log_hist archives
- Proton simulation reduced event-data HDF5 files

The input files can be generated using simtools-generate-bias-curve-submissions.

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
telescope (str, required)
    Telescope name for configuration.
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
        --telescope LSTN-01 \\
        --output bias_curves.png

"""

from pathlib import Path

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.simtel.bias_curve_generator import generate_bias_curves

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "data_dir",
        type=Path,
        required=True,
        help="Directory containing both NSB logs/log_hist archives and proton simulation files.",
    ),
    cli.ArgumentDefinition(
        "output",
        type=Path,
        default=Path("bias_curve.png"),
        help="Output plot file path or output directory. Default: bias_curve.png",
    ),
    cli.ArgumentDefinition(
        "nsb_output",
        type=Path,
        required=False,
        help="Output ECSV table file for NSB trigger rates. If not specified, no table is written.",
    ),
    cli.ArgumentDefinition(
        "proton_output",
        type=Path,
        required=False,
        help="Output ECSV table file for proton rates. If not specified, no table is written.",
    ),
    cli.ArgumentDefinition(
        "title", type=str, default="Trigger Rate Bias Curves", help="Plot title."
    ),
    cli.ArgumentDefinition(
        "ymin",
        type=float,
        default=100.0,
        help="Minimum trigger rate value for plotting. Default: 1e2",
    ),
    cli.ArgumentDefinition(
        "ymax",
        type=float,
        default=500000.0,
        help="Maximum trigger rate value for plotting. Default: 5e5",
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        cli.MODEL_VERSION(),
        cli.OVERWRITE_MODEL_PARAMETERS(),
        cli.SITE(),
        cli.TELESCOPE(),
        *cli.PATH_ARGUMENTS,
    ),
    database=True,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

    generate_bias_curves(app_context.args)


if __name__ == "__main__":
    main()
