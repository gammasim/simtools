#!/usr/bin/python3

"""
Generates a reduced dataset from input data.

Command line arguments
----------------------
--prefix (str, required)
    Path prefix for the input files.
--output_file (str, required)
    Path to save the output file.
--max_files (int, optional, default=100)
    Maximum number of files to process.

Example
-------
Generate a reduced dataset from input files and save the result.

.. code-block:: console

    generate-reduced-dataset \
        --prefix path/to/input_files \
        --output_file path/to/output_file.hdf5
"""

import logging
from pathlib import Path

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.io_operations import io_handler
from simtools.production_configuration.generate_reduced_datasets import ReducedDatasetGenerator


def _parse(label, description):
    """
    Parse command line arguments.

    Returns
    -------
    dict
        Parsed command-line arguments.
    """
    config = configurator.Configurator(label=label, description=description)

    config.parser.add_argument(
        "--prefix", type=str, required=True, help="Prefix path for input files."
    )
    config.parser.add_argument(
        "--wildcard",
        type=str,
        required=True,
        help="Wildcard for querying the files in the directory i.e. 'gamma_*dark*.simtel.zst'",
    )
    config.parser.add_argument(
        "--output_file", type=str, required=True, help="Path to output HDF5 file."
    )
    config.parser.add_argument(
        "--max_files", type=int, default=100, help="Maximum number of files to process."
    )
    config.parser.add_argument("--print_hdf5", action="store_true",
                                help="Print additionally HDF5 file contents after file generation.")

    return config.initialize(db_config=False)


def main():
    """Process EventIO files and store data in HDF5 reduced dataset."""
    label = Path(__file__).stem
    args_dict, _ = _parse(label=label,
                description="Process EventIO files and store data in HDF5 reduced dataset.")

    _logger = logging.getLogger()
    _logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))
    _logger.info(f"Loading input files with prefix: {args_dict['prefix']}")

    input_path = Path(args_dict["prefix"])
    files = list(input_path.glob(args_dict["wildcard"]))
    if not files:
        _logger.warning("No matching input files found.")
        return


    output_path = io_handler.IOHandler().get_output_directory(label)
    output_filepath = Path(output_path).joinpath(f"{args_dict['output_file']}")


    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    generator = ReducedDatasetGenerator(files, output_filepath, args_dict["max_files"])
    generator.process_files()
    _logger.info(f"reduced dataset saved to: {output_filepath}")
    if args_dict["print_hdf5"]:
        ReducedDatasetGenerator(files, output_filepath, args_dict["max_files"]).print_hdf5_file()


if __name__ == "__main__":
    main()
