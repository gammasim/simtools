#!/usr/bin/python3

"""
Inspect simulation-related files.

This application currently supports HDF5 files and is intended as a generic
inspection entry point that can later be extended to additional simulation
file types.

For known simulation products, the application can append specialized
inspection sections on top of the generic file-structure report.

Command line arguments
----------------------
input_file (str, required)
    Simulation-related file to inspect.
max_entries (int, optional)
    Maximum number of HDF5 groups and datasets to print.
"""

from simtools.application_control import build_application
from simtools.io.file_inspector import format_inspection_report, inspect_file
from simtools.production_configuration.trigger_histograms import (
    format_trigger_histogram_inspection,
    inspect_trigger_histogram_file,
)


def _add_arguments(parser):
    """Register application-specific command line arguments."""
    parser.add_argument(
        "--input_file",
        help="Simulation-related file to inspect.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--max_entries",
        help="Maximum number of HDF5 entries to print.",
        required=False,
        default=50,
        type=int,
    )


def main():
    """Run the simulation-file inspector."""
    app_context = build_application(
        initialization_kwargs={"db_config": False},
        startup_kwargs={"setup_io_handler": False},
    )
    report = inspect_file(
        app_context.args["input_file"],
        max_entries=app_context.args["max_entries"],
    )
    output_sections = [format_inspection_report(report)]
    trigger_histogram_report = inspect_trigger_histogram_file(app_context.args["input_file"])
    if trigger_histogram_report is not None:
        output_sections.append(format_trigger_histogram_inspection(trigger_histogram_report))
    print("\n".join(output_sections))


if __name__ == "__main__":
    main()
