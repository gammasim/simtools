#!/usr/bin/python3

r"""
Write reduced event lists from sim_telarray output files.

Processes one or more sim_telarray output files and writes one reduced event
list (HDF5) per input file.  The output file names are derived from the
input file names by replacing the sim_telarray suffix with
'.reduced_event_data.hdf5'.

Command line arguments
----------------------
input_files (list of str, required)
    One or more sim_telarray output files (e.g., '*.simtel.zst').
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
    parser.add_argument(
        "--input_files",
        nargs="+",
        required=True,
        help="sim_telarray output file(s) to process (e.g., '*.simtel.zst').",
    )


def main():
    """See CLI description."""
    app_context = build_application(
        initialization_kwargs={"db_config": False},
        startup_kwargs={"setup_io_handler": False, "resolve_sim_software_executables": False},
    )

    Simulator.write_reduced_event_lists(
        input_files=app_context.args["input_files"],
        output_path=app_context.args["output_path"],
    )


if __name__ == "__main__":
    main()
