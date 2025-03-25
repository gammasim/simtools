#!/usr/bin/python3

"""
Reduces and compiles event data from multiple input files into a structured dataset with event info.

Command line arguments
----------------------
prefix (str, required)
    Path prefix for the input files.
output_file (str, required)
    Path to save the output file.
max_files (int, optional, default=100)
    Maximum number of files to process.
print_dataset_information (flag)
    Print information about the datasets in the generated reduced event dataset.

Example
-------
Generate a reduced dataset from input files and save the result.

.. code-block:: console

    simtools-production-extract-mc-event-data \
    simtools-generate-simtel-event-data \
        --prefix path/to/input_files/ \
        --wildcard 'gamma_*dark*.simtel.zst' \
        --output_file output_file.hdf5 \
        --max_files 50 \
        --print_dataset_information
"""

import logging
from pathlib import Path

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.io_operations import io_handler
from simtools.simtel.simtel_io_event_reader import SimtelIOEventDataReader
from simtools.simtel.simtel_io_event_writer import SimtelIOEventDataWriter


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
        help="Wildcard for querying the files in the directory (e.g., 'gamma_*dark*.simtel.zst')",
    )
    config.parser.add_argument("--output_file", type=str, required=True, help="Output filename.")
    config.parser.add_argument(
        "--max_files", type=int, default=100, help="Maximum number of files to process."
    )

    config.parser.add_argument(
        "--print_dataset_information",
        type=int,
        help="Print given number of rows of the dataset.",
        default=0,
    )

    return config.initialize(db_config=False)


def main():
    """
    Process event data files and store data in reduced dataset.

    The reduced dataset contains shower information, array information and triggered telescopes.
    """
    label = Path(__file__).stem

    args_dict, _ = _parse(
        label=label,
        description=(
            "Process files and store reduced dataset with event information, "
            "array information and triggered telescopes."
        ),
    )

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
    generator = SimtelIOEventDataWriter(files, output_filepath, args_dict["max_files"])
    generator.process_files()
    _logger.info(f"reduced dataset saved to: {output_filepath}")

    if args_dict["print_dataset_information"] > 0:
        reader = SimtelIOEventDataReader(output_filepath)
        reader.print_dataset_information(args_dict.get("print_dataset_information"))


if __name__ == "__main__":
    main()
