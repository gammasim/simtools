#!/usr/bin/python3

"""
Inspect simulation-related files and prints out a structured report of their contents.

For known simulation products, the application can append specialized
inspection sections on top of the generic file-structure report.

Command line arguments
----------------------
input_file (str, required)
    Simulation-related file to inspect.
max_entries (int, optional)
    Maximum number of entries or preview lines to print. Use 0 for no limit.
"""

from simtools.application_control import build_application
from simtools.io.file_inspector import inspect_file


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
        help="Maximum number of entries or preview lines to print; use 0 for no limit.",
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
    reports = inspect_file(
        app_context.args["input_file"],
        max_entries=app_context.args["max_entries"],
        format_report=True,
    )
    print("\n".join(reports))


if __name__ == "__main__":
    main()
