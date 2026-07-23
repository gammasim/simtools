#!/usr/bin/python3

r"""
Write reduced event lists from sim_telarray output files.

Processes one or more sim_telarray output files and writes reduced event
lists (HDF5). Input files can be given directly or read from a text file and
processed in batches.

Command line arguments
----------------------
input_files (list of str, optional)
    One or more sim_telarray output files (e.g., ``*.simtel.zst``).
input_file_list (str, optional)
    Text file containing one sim_telarray output file per line.
files_per_reduced_event_file (int, optional)
    Number of input files combined into each reduced event file. Defaults to 1.
max_workers (int, optional)
    Maximum number of parallel output-file workers.
output_path (str, optional)
    Directory for the output files. Defaults to './simtools-output/'.

Example
-------
Write reduced event lists for a set of sim_telarray output files:

.. code-block:: console

    simtools-write-reduced-event-lists \\
        --input_files run000001.simtel.zst run000002.simtel.zst \\
        --output_path /path/to/output/

"""

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
        *cli.PATH_ARGUMENTS,
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
    )


if __name__ == "__main__":
    main()
