r"""
Merge tables from multiple input files into single tables.

Allows to merge tables (e.g., astropy tables) from multiple input files into a single file.
The input files can be of HDF5 or FITS format.
specified output file.

Merging large tables in FITS are not recommended, as it may lead to
performance issues.

Command line arguments
----------------------
input str
    Input file(s) (e.g., 'file1 file2').
input_list str
    File with list of input files with tables.
table_names list of str
    Names of tables to merge from each input file.
output_file str
    Output file name.
output_path str
    Path to the output file for the merged tables.

Example
-------

Merge tables from two files generated with 'simtools-generate-simtel-event-data' into a single file.

.. code-block:: console

    simtools-merge-tables \\
        --input_files file1 file2' \\
        --table_names 'SHOWERS TRIGGERS FILE_INFO' \\
        --output_file merged_tables.hdf5


"""

import simtools.utils.general as gen
from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.io import table_handler

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "input_files",
        exclusive_group="input group",
        exclusive_group_required=True,
        type=str,
        nargs="+",
        help="Input file(s) (e.g., 'file1 file2') or a file with a list of input files.",
    ),
    cli.ArgumentDefinition(
        "table_names", type=str, nargs="+", help="Names of tables to merge from each input file."
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
    """See CLI description."""
    app_context = APPLICATION.start()

    app_context.logger.info(f"Loading input files: {app_context.args['input_files']}")

    # accept fits.gz files (.gz)
    input_files = gen.get_list_of_files_from_command_line(
        app_context.args["input_files"], [".hdf5", ".gz"]
    )

    output_filepath = app_context.io_handler.get_output_file(app_context.args["output_file"])

    table_handler.merge_tables(
        input_files,
        input_table_names=app_context.args["table_names"],
        output_file=output_filepath,
    )


if __name__ == "__main__":
    main()
