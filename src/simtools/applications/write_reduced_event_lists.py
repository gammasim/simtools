#!/usr/bin/python3

"""Write reduced event lists from sim_telarray output files.

This application supports the ``local`` (default) and ``htcondor`` execution backends.
"""

from pathlib import Path

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.simulator import Simulator
from simtools.utils import general

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "input_files",
        exclusive_group="input group",
        exclusive_group_required=True,
        nargs="+",
        help="sim_telarray output file(s) to process (e.g., '*.simtel.zst').",
    ),
    cli.ArgumentDefinition(
        "input_file_list",
        exclusive_group="input group",
        exclusive_group_required=True,
        help="Text file containing one sim_telarray output file per line.",
    ),
    cli.ArgumentDefinition(
        "input_file_list_pattern",
        exclusive_group="input group",
        exclusive_group_required=True,
        help="Glob pattern matching text files containing sim_telarray output file lists.",
    ),
    cli.ArgumentDefinition(
        "files_per_reduced_event_file",
        type=int,
        default=1,
        help="Number of input files combined into each reduced event file (default: 1).",
    ),
    cli.ArgumentDefinition(
        "max_workers",
        type=int,
        default=None,
        help=(
            "Maximum parallel output-file workers. Default: 60%% of CPU cores; use 1 for "
            "serial execution or 0 for all cores."
        ),
    ),
    cli.ArgumentDefinition(
        "wait",
        action="store_true",
        default=False,
        help="Wait for HTCondor jobs and validate outputs before exiting.",
    ),
)


def _resolve_input_file_list_pattern(pattern):
    """Return sorted regular files matching an input-list glob pattern."""
    matches = sorted(
        path for path in general.resolve_file_patterns(pattern) if Path(path).is_file()
    )
    if not matches:
        raise FileNotFoundError(f"No input file lists found for pattern: {pattern}")
    return matches


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        *cli.BACKEND_ARGUMENTS,
        *cli.OUTPUT_PATH_ARGUMENTS,
    ),
    resolve_sim_software_executables=False,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()
    input_file_lists = None
    if app_context.args.get("input_file_list_pattern"):
        input_file_lists = _resolve_input_file_list_pattern(
            app_context.args["input_file_list_pattern"]
        )

    Simulator.write_reduced_event_lists(
        input_files=app_context.args["input_files"],
        input_file_list=app_context.args["input_file_list"],
        input_file_lists=input_file_lists,
        files_per_reduced_event_file=app_context.args["files_per_reduced_event_file"],
        max_workers=app_context.args["max_workers"],
        backend=app_context.args.get("backend", "local"),
        backend_config=app_context.args.get("backend_config"),
        wait_for_completion=app_context.args.get("wait", False),
        output_path=app_context.io_handler.get_output_directory(),
        metadata_args=app_context.args,
    )


if __name__ == "__main__":
    main()
