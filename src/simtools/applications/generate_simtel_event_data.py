#!/usr/bin/python3

r"""
Generate a reduced dataset of event data from simulated files.

Processes sim_telarray output files (typically of type '.simtel.zst') and creates a
reduced dataset containing shower information, array-level parameters, and data about
triggered telescopes.

Command line arguments
----------------------
prefix (str, required)
    Path prefix for the input files.
output_file (str, required)
    Output file path.
max_files (int, optional, default=100)
    Maximum number of input files to process.
print_dataset_information (int, optional, default=0)
    Print information about the datasets in the generated reduced event dataset.

Example
-------
Generate a reduced dataset from input files and save the result.

.. code-block:: console

    simtools-production-extract-mc-event-data \\
    simtools-generate-simtel-event-data \\
        --input 'path/to/input_files/gamma_*dark*.simtel.zst' \\
        --output_file output_file.hdf5 \\
        --max_files 50 \\
        --print_dataset_information 10
"""

import logging
from pathlib import Path

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.data_model.metadata_collector import MetadataCollector
from simtools.io_operations import io_handler
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
        "--input",
        type=str,
        required=True,
        help="Input file path (wildcards allowed; e.g., '/data_path/gamma_*dark*.simtel.zst')",
    )
    config.parser.add_argument(
        "--max_files", type=int, default=100, help="Maximum number of input files to process."
    )
    config.parser.add_argument(
        "--print_dataset_information",
        type=int,
        help="Print data set information for the given number of events.",
        default=0,
    )
    return config.initialize(db_config=False, output=True)


def main():  # noqa: D103
    label = Path(__file__).stem

    args_dict, _ = _parse(
        label=label,
        description=(
            "Process files and store reduced dataset with event information, "
            "array information and triggered telescopes."
        ),
    )

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))
    logger.info(f"Loading input files from: {args_dict['input']}")

    input_pattern = Path(args_dict["input"])
    files = list(input_pattern.parent.glob(input_pattern.name))
    if not files:
        logger.warning("No matching input files found.")
        return

    output_filepath = io_handler.IOHandler().get_output_file(args_dict["output_file"])
    generator = SimtelIOEventDataWriter(files, args_dict["max_files"])
    tables = generator.process_files()
    generator.write(
        output_filepath,
        tables=tables,
    )
    MetadataCollector.dump(args_dict=args_dict, output_file=output_filepath.with_suffix(".yml"))

    if args_dict["print_dataset_information"] > 0:
        for table in tables:
            table.pprint(max_lines=args_dict["print_dataset_information"], max_width=-1)


if __name__ == "__main__":
    main()
