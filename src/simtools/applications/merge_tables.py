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

from pathlib import Path

import simtools.utils.general as gen
from simtools.application_control import get_application_label, startup_application
from simtools.configuration import configurator
from simtools.io import io_handler, table_handler


def _parse():
    """Parse command line arguments."""
    config = configurator.Configurator(
        label=get_application_label(__file__),
        description="Merge tables from multiple input files into single tables.",
    )

    input_group = config.parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input_files",
        type=str,
        nargs="+",
        help="Input file(s) (e.g., 'file1 file2') or a file with a list of input files.",
    )
    config.parser.add_argument(
        "--table_names",
        type=str,
        nargs="+",
        help="Names of tables to merge from each input file.",
    )

    return config.initialize(db_config=False, output=True)


def main():
    """Merge tables from multiple input files into single tables."""
    app_context = startup_application(_parse)

    app_context.logger.info(f"Loading input files: {app_context.args['input_files']}")

    # accept fits.gz files (.gz)
    input_files = gen.get_list_of_files_from_command_line(
        app_context.args["input_files"], [".hdf5", ".gz"]
    )

    output_path = io_handler.IOHandler().get_output_directory()
    output_filepath = Path(output_path).joinpath(f"{app_context.args['output_file']}")

    table_handler.merge_tables(
        input_files,
        input_table_names=app_context.args["table_names"],
        output_file=output_filepath,
    )


if __name__ == "__main__":
    main()
