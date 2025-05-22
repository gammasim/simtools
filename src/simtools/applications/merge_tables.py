"""
Merge tables from multiple input files into single tables.

Command line arguments
----------------------

"""

import logging
from pathlib import Path

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.io_operations import io_table_handler


def _parse(label, description):
    """
    Parse command line arguments.

    Returns
    -------
    dict
        Parsed command-line arguments.
    """
    config = configurator.Configurator(label=label, description=description)

    input_group = config.parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        type=str,
        nargs="+",
        help="Input file(s) (e.g., 'file1 file2')",
    )
    input_group.add_argument(
        "--input_list",
        type=str,
        help="File with list of input files with tables.",
    )
    config.parser.add_argument(
        "--table_names",
        type=str,
        nargs="+",
        help="Names of tables to merge from each input file.",
    )

    return config.initialize(db_config=False, output=True)


def main():  # noqa: D103
    label = Path(__file__).stem

    args_dict, _ = _parse(
        label=label,
        description=("Merge tables from multiple input files into single tables."),
    )
    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))
    logger.info(f"Loading input files from: {args_dict['input']}")

    input_files = args_dict.get("input") or gen.collect_data_from_file(args_dict["input_list"])

    io_table_handler.merge_tables(
        input_files,
        input_table_names=args_dict["table_names"],
        output_file=args_dict["output_file"],
    )


if __name__ == "__main__":
    main()
