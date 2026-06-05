#!/usr/bin/python3

r"""
Plot CORSIKA limits from a ECSV table.

This application reads a CORSIKA limits table and plots the limits
as function of zenith angle.


Command line arguments
----------------------
input (str, required)
    Path to a CORSIKA limits table in ECSV format.

Example
-------

.. code-block:: console

   simtools-production-plot-corsika-limits \
       --input simtools-output/merged_corsika_limits.ecsv \
       --output_path simtools-output
"""

from simtools.application_control import build_application
from simtools.data_model import data_reader
from simtools.visualization.plot_corsika_limits import plot_limits


def _add_arguments(parser):
    """Register application-specific command line arguments."""
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to a merged CORSIKA limits table in ECSV format.",
    )


def main():
    """Run CORSIKA limits plotting."""
    app_context = build_application(initialization_kwargs={"output": True})

    plot_limits(
        data_reader.read_table_from_file(app_context.args["input"]),
        app_context.io_handler.get_output_directory(),
    )


if __name__ == "__main__":
    main()
