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

from simtools.application_control import build_application
from simtools.simulator import Simulator


def _add_arguments(parser):
    """Register application-specific command line arguments."""
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input_files",
        nargs="+",
        help="sim_telarray output file(s) to process (e.g., '*.simtel.zst').",
    )
    input_group.add_argument(
        "--input_file_list",
        help="Text file containing one sim_telarray output file per line.",
    )
    parser.add_argument(
        "--files_per_reduced_event_file",
        "--files_per_reduced_events_file",
        type=int,
        default=1,
        help="Number of input files combined into each reduced event file (default: 1).",
    )


def main():
    """See CLI description."""
    app_context = build_application(
        initialization_kwargs={"db_config": False},
        startup_kwargs={"setup_io_handler": True, "resolve_sim_software_executables": False},
    )

    Simulator.write_reduced_event_lists(
        input_files=app_context.args["input_files"],
        input_file_list=app_context.args["input_file_list"],
        files_per_reduced_event_file=app_context.args["files_per_reduced_event_file"],
        output_path=app_context.io_handler.get_output_directory(),
    )


if __name__ == "__main__":
    main()
