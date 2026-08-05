#!/usr/bin/python3

"""Write reduced event lists from sim_telarray output files."""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.simulator import Simulator

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
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        *cli.OUTPUT_PATH_ARGUMENTS,
    ),
    resolve_sim_software_executables=False,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

    Simulator.write_reduced_event_lists(
        input_files=app_context.args["input_files"],
        input_file_list=app_context.args["input_file_list"],
        files_per_reduced_event_file=app_context.args["files_per_reduced_event_file"],
        max_workers=app_context.args["max_workers"],
        output_path=app_context.io_handler.get_output_directory(),
        metadata_args=app_context.args,
    )


if __name__ == "__main__":
    main()
