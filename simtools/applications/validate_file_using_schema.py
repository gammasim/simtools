#!/usr/bin/python3
"""
    Summary
    -------
    Validate yaml or ecsv file using a json schema file.

    Command line arguments
    ----------------------
    file_name (str)
      input file to be validated
    schema (str)
      schema file (jsonschema format) used for validation

    Raises
    ------
    FileNotFoundError
      if file to be validated is not found

    Example
    -------

    .. code-block:: console

        simtools-validate-file-using-schema \
         --file_name tests/resources/MLTdata-preproduction.meta.yml \
         --schema simtools/schemas/metadata.metaschema.yml

"""

import logging
from pathlib import Path

import yaml

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.data_model import metadata_model, validate_data


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
    config.parser.add_argument("--file_name", help="file to be validated", required=True)
    config.parser.add_argument("--schema", help="json schema file", required=True)
    return config.initialize(paths=False)


def _validate_yaml_file(args_dict, logger):
    """
    Validate a yaml file

    """

    try:
        with open(args_dict["file_name"], "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"Input file {args_dict['file_name']} not found")
        raise

    metadata_model.validate_schema(data, args_dict["schema"])


def _validate_ecsv_file(args_dict):
    """
    Validate an ecsv file

    """

    data_validator = validate_data.DataValidator(
        schema_file=args_dict["schema"],
        data_file=args_dict["file_name"],
    )
    data_validator.validate_and_transform()


def main():
    label = Path(__file__).stem
    args_dict, _ = _parse(label, description="Validate yaml or ecsv file using a json schema file.")

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    # check if file name ends with yml or yaml
    if args_dict["file_name"].endswith(".yml") or args_dict["file_name"].endswith(".yaml"):
        _validate_yaml_file(args_dict, logger)
    elif args_dict["file_name"].endswith(".ecsv"):
        _validate_ecsv_file(args_dict)
    else:
        logger.error(f"File extension not supported for {args_dict['file_name']}")


if __name__ == "__main__":
    main()
