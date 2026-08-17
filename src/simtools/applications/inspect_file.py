#!/usr/bin/python3

"""Inspect simulation-related files and prints out a structured report of their contents."""

from pathlib import Path

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.io.file_inspector import inspect_file

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "input_file", help="Simulation-related file to inspect.", required=True, type=Path
    ),
    cli.ArgumentDefinition(
        "max_entries",
        help="Maximum number of entries or preview lines to print; use 0 for no limit.",
        required=False,
        default=50,
        type=int,
    ),
    cli.ArgumentDefinition(
        "show_entry",
        help="Print one HDF5 root dataset, for example SIMULATION_METADATA or SHOWERS.",
        required=False,
        default=None,
        type=str,
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(*_ARGUMENTS,),
    setup_io_handler=False,
)


def main():
    """Run the simulation-file inspector."""
    app_context = APPLICATION.start()
    reports = inspect_file(
        app_context.args["input_file"],
        max_entries=app_context.args["max_entries"],
        format_report=True,
        entry_name=app_context.args["show_entry"],
    )
    print("\n".join(reports))


if __name__ == "__main__":
    main()
