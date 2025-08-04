#!/usr/bin/python3
r"""
    Generate a default simtools metadata file from a json schema.

    Command line arguments
    ----------------------
    schema (str, optional)
        Schema file describing the input data
        (default: simtools/schemas/metadata.metaschema.yml)
    output_file (str, optional)
        Output file name.

    Example
    -------
    .. code-block:: console

        simtools-generate-default-metadata \\
            --schema simtools/schemas/metadata.metaschema.yml \\
            --output_file default_metadata.yml


    """

import json
import logging
from pathlib import Path

import yaml

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.data_model import metadata_model
from simtools.io import io_handler


def _parse(label, description):
    """
    Parse command line configuration.

    Parameters
    ----------
    label: str
        Label describing application.
    description: str
        Description of application.

    Returns
    -------
    CommandLineParser
        Command line parser object

    """
    config = configurator.Configurator(label=label, description=description)

    config.parser.add_argument(
        "--schema",
        help="schema file describing input data",
        type=str,
        required=True,
    )
    config.parser.add_argument(
        "--output_file",
        help="output file name (if not given: print to stdout)",
        type=str,
        required=False,
    )

    return config.initialize(output=False, require_command_line=True)


def main():  # noqa: D103
    label = Path(__file__).stem
    args_dict, _ = _parse(
        label, description="Generate a default simtools metadata file from a json schema."
    )

    _logger = logging.getLogger()
    _logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    default_values = metadata_model.get_default_metadata_dict(args_dict["schema"])

    if args_dict["output_file"] is None:
        print(default_values)
    else:
        _io_handler = io_handler.IOHandler()
        _out_file = _io_handler.get_output_file(args_dict["output_file"])
        _logger.info(f"Writing default values to {_out_file}")
        if args_dict["output_file"].endswith((".yml", ".yaml")):
            with open(_out_file, "w", encoding="utf-8") as file:
                yaml.dump(
                    default_values,
                    file,
                    default_flow_style=False,
                    sort_keys=False,
                )
        if args_dict["output_file"].endswith(".json"):
            with open(_out_file, "w", encoding="utf-8") as file:
                json.dump(
                    default_values,
                    file,
                    indent=4,
                    sort_keys=False,
                )


if __name__ == "__main__":
    main()
