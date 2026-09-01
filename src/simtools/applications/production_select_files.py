#!/usr/bin/python3

"""Select production files from production metadata manifests."""

import sys

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.production_configuration.production_file_selection import (
    select_file_groups,
    selection_summary,
    write_selection_file,
)

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "production_path",
        help="Directory containing simulate_prod job output directories.",
        type=str,
        required=True,
    ),
    cli.ArgumentDefinition(
        "select",
        help="Selection expression as dotted.path=value. Can be repeated.",
        action="append",
        default=[],
    ),
    cli.ArgumentDefinition(
        "file_type",
        help="Manifest file type to select.",
        type=str,
        default="reduced_event_data",
    ),
    cli.ArgumentDefinition(
        "require_complete_runs",
        help="Fail when selected run numbers are not contiguous within each group.",
        action="store_true",
        default=False,
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(*_ARGUMENTS, *cli.OUTPUT_ARGUMENTS),
    initialize_output=False,
)


def main():
    """Run the production-file selection CLI application."""
    app_context = APPLICATION.start()
    result = select_file_groups(
        app_context.args["production_path"],
        selections=app_context.args.get("select"),
        file_type=app_context.args["file_type"],
        require_complete_runs=app_context.args.get("require_complete_runs", False),
    )
    sys.stdout.write(selection_summary(result) + "\n")
    if app_context.args.get("output_file") and not app_context.args.get("output_file_from_default"):
        write_selection_file(result, app_context.args["output_file"])


if __name__ == "__main__":
    main()
