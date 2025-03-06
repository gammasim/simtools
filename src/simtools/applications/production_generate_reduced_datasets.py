#!/usr/bin/python3

"""
Reduces and compiles event data from multiple input files into a structured dataset.

Command line arguments
----------------------
--prefix (str, required)
    Path prefix for the input files.
--output_file (str, required)
    Path to save the output file.
--max_files (int, optional, default=100)
    Maximum number of files to process.
--print_dataset_information (flag)
    Print information about the datasets in the generated reduced event dataset.

Example
-------
Generate a reduced dataset from input files and save the result.

.. code-block:: console

    simtools-production-generate-reduced-dataset \
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
        help="Wildcard for querying the files in the directory (e.g., 'gamma_*dark*.simtel.zst')",
    )
    config.parser.add_argument("--output_file", type=str, required=True, help="Output filename.")
    config.parser.add_argument(
        "--max_files", type=int, default=100, help="Maximum number of files to process."
    )

    config.parser.add_argument(
        "--print_dataset_information",
        action="store_true",
        help="Print information about the datasets in the generated reduced event dataset.",
    )

    return config.initialize(db_config=False)


def main():
    """
    Process event data files and store data in reduced dataset.

    The reduced dataset contains the following information:
        - simulated: List of simulated events.
        - shower_id_triggered: List of triggered shower IDs
            (as in the telescope definition file used for simulations).
        - triggered_energies: List of energies for triggered events.
        - num_triggered_telescopes: Number of triggered telescopes for each event.
        - core_x: X-coordinate of the shower core.
        - core_y: Y-coordinate of the shower core.
        - trigger_telescope_list_list: List of lists containing triggered telescope IDs.
        - file_names: List of input file names.
        - shower_sim_azimuth: Simulated azimuth angle of the shower.
        - shower_sim_altitude: Simulated altitude angle of the shower.
        - array_altitudes: List of altitudes for the array.
        - array_azimuths: List of azimuths for the array.
    """
    label = Path(__file__).stem

    args_dict, _ = _parse(
        label=label,
        description=(
            "Process EventIO files and store reduced dataset with event information, "
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
    generator = ReducedDatasetGenerator(files, output_filepath, args_dict["max_files"])
    generator.process_files()
    _logger.info(f"reduced dataset saved to: {output_filepath}")
    if args_dict["print_dataset_information"]:
        generator.print_hdf5_file()


if __name__ == "__main__":
    main()
