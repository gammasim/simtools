#!/usr/bin/python3

"""
    Summary
    -------
    Generate a default simtools metadata file from a json schema.

    Command line arguments
    ----------------------
    schema (str, optional)
        Schema describing the input data
        (default: simtools/schemas/metadata.schema.yml)
    output_file (str, optional)
        Output file name.
    required_only (bool, optional)
        Output required fields only.

    TODO:
    - output required fields only

    """

import logging
from importlib.resources import files
from pathlib import Path

import yaml

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
        required=False,
    )
    config.parser.add_argument(
        "--output_file",
        help="output file name",
        type=str,
        required=False,
    )
    config.parser.add_argument(
        "--required_only",
        help="output required fields only",
        action="store_true",
        required=False,
    )

    return config.initialize(output=False)


def fill_defaults(schema, required_only=False):
    defaults = {}

    def fill_defaults_recursive(subschema, current_dict):
        if "properties" in subschema:
            for prop, prop_schema in subschema["properties"].items():
                if "default" in prop_schema:
                    current_dict[prop] = prop_schema["default"]
                elif "type" in prop_schema and prop_schema["type"] == "object":
                    current_dict[prop] = {}
                    fill_defaults_recursive(prop_schema, current_dict[prop])
                # You might need to handle other types like arrays, etc., if present

    fill_defaults_recursive(schema, defaults)
    return defaults


def main():
    label = Path(__file__).stem
    args_dict, _ = _parse(
        label, description="Generate a default simtools metadata file from a json schema."
    )

    _logger = logging.getLogger()
    _logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    if args_dict["schema"] is None:
        with files("simtools").joinpath("schemas/metadata.schema.yml").open(
            "r", encoding="utf-8"
        ) as file:
            schema = yaml.safe_load(file)
            _logger.info("Reading default schema from simtools/schemas/metadata.schema.yml")
    else:
        schema = gen.collect_data_from_yaml_or_dict(in_yaml=args_dict["schema"], in_dict=None)
        _logger.info(f"Reading schema from {args_dict['schema']}")

    default_values = fill_defaults(
        schema["definitions"]["CTA"], required_only=args_dict["required_only"]
    )

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
