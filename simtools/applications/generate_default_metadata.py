#!/usr/bin/python3
"""
    Summary
    -------
    Generate a default simtools metadata file from a json schema.

    Command line arguments
    ----------------------
    schema (str, optional)
        Schema file describing the input data
        (default: simtools/schemas/metadata.schema.yml)
    output_file (str, optional)
        Output file name.

    Example
    -------
    .. code-block:: console

        simtools-generate-default-metadata
            --schema simtools/schemas/metadata.schema.yml
            --output_file default_metadata.yml


    """

import logging
from pathlib import Path

import yaml

import simtools.data_model.metadata_model as metadata_model
import simtools.utils.general as gen
from simtools.configuration import configurator


def _parse(label, description):
    """
    Parse command line configuration

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

    return config.initialize(output=False)


def main():
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
        _logger.info(f"Writing default values to {args_dict['output_file']}")
        with open(args_dict["output_file"], "w", encoding="utf-8") as file:
            yaml.dump(
                default_values,
                file,
                default_flow_style=False,
                sort_keys=False,
            )


if __name__ == "__main__":
    main()
