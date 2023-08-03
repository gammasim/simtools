#!/usr/bin/python3
"""
    Summary
    -------
    Validate parameter description files using a json schema file.

    Command line arguments
    ----------------------
    file-name (str)
      input file to be validated
    schema (str)
      schema file (jsonschema format) used for validation

    Example
    -------

    .. code-block:: console

        simtools-validate-schema-file \
         --file-name tests/resources/MST_mirror_2f_measurements.schema.yml \
         --schema jsonschema.yml

"""

import logging
from pathlib import Path

import jsonschema
import yaml

import simtools.util.general as gen
from simtools.configuration import configurator


def _parse(label, description):
    """
    Parse command line configuration

    Parameters
    ----------
    label (str)
        application label
    description (str)
        application description

    Returns
    -------
    config (Configurator)
        application configuration

    """

    config = configurator.Configurator(label=label, description=description)
    config.parser.add_argument("-f", "--file_name", help="file to be validated", required=True)
    config.parser.add_argument("-s", "--schema", help="json schema file", required=True)
    return config.initialize()


def load_schema(schema_file):
    """
    Load parameter schema from file.

    Parameters
    ----------
    schema_file (str)
        schema file

    Returns
    -------
    parameter_schema (dict)
        parameter schema

    Raises
    ------
    FileNotFoundError
        if schema file is not found

    """

    try:
        with open(schema_file, "r", encoding="utf-8") as file:
            parameter_schema = yaml.safe_load(file)
    except FileNotFoundError:
        logging.error(f"Schema file {schema_file} not found")
        raise

    return parameter_schema


def validate_schema_file(input_file, schema):
    """
    Validate parameter file against schema.

    Parameters
    ----------
    input_file (str)
        input file to be validated
    schema (dict)
        schema used for validation

    Raises
    ------
    FileNotFoundError
        if input file is not found

    """

    try:
        with open(input_file, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except FileNotFoundError:
        logging.error(f"Input file {input_file} not found")
        raise

    try:
        jsonschema.validate(data, schema=schema)
    except jsonschema.exceptions.ValidationError:
        logging.error(f"Schema validation failed for {input_file} using {schema}")
        raise

    print(f"Schema validation successful for {input_file}")


def main():
    label = Path(__file__).stem
    args_dict, _ = _parse(label, description="Parameter file schema checking")
    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    validate_schema_file(args_dict["file_name"], load_schema(args_dict["schema"]))


if __name__ == "__main__":
    main()
