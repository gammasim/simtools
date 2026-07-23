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

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.data_model import data_reader
from simtools.visualization.plot_corsika_limits import plot_limits

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "input",
        type=str,
        required=True,
        help="Path to a merged CORSIKA limits table in ECSV format.",
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        *cli.PATH_ARGUMENTS,
        *cli.OUTPUT_ARGUMENTS,
    ),
    initialize_output=True,
)


def main():
    """Run CORSIKA limits plotting."""
    app_context = APPLICATION.start()

    plot_limits(
        data_reader.read_table_from_file(app_context.args["input"]),
        app_context.io_handler.get_output_directory(),
    )


if __name__ == "__main__":
    main()
