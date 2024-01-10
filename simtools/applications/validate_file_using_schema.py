#!/usr/bin/python3
"""
    Summary
    -------
    Validate yaml, json, or ecsv file using a json schema file.

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

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.data_model import data_reader, metadata_collector, validate_data


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
    config.parser.add_argument("--schema", help="json schema file", required=False)
    config.parser.add_argument(
        "--validate_metadata",
        help="validate metadata",
        action="store_true",
        default=False,
    )
    return config.initialize(paths=False)


def _validate_yaml_or_json_file(args_dict, logger):
    """
    Validate a yaml or json file.
    Schema is either given as command line argument, read from the meta_schema_url or from
    the metadata section of the data dictionary.

    """

    data_reader.read_value_from_file(
        file_name=args_dict["file_name"],
        schema_file=args_dict["schema"],
        validate=True,
    )
    logger.debug("Successful validation of json/yaml file")


def _validate_ecsv_file(args_dict, logger):
    """
    Validate an ecsv file

    """

    data_validator = validate_data.DataValidator(
        schema_file=args_dict["schema"],
        data_file=args_dict["file_name"],
    )
    data_validator.validate_and_transform()
    logger.debug("Successful validation of escv file")


def _validate_metadata(args_dict, logger):
    """
    Validate metadata.

    """

    # metadata_collector runs the metadata validation by default, no need to do anything else
    metadata_collector.MetadataCollector(None, metadata_file_name=args_dict["file_name"])
    logger.debug("Successful validation of metadata")


def main():
    label = Path(__file__).stem
    args_dict, _ = _parse(label, description="Validate yaml or ecsv file using a json schema file.")

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    if args_dict["validate_metadata"]:
        _validate_metadata(args_dict, logger)
    elif any(args_dict["file_name"].endswith(ext) for ext in (".yml", ".yaml", ".json")):
        _validate_yaml_or_json_file(args_dict, logger)
    elif args_dict["file_name"].endswith(".ecsv"):
        _validate_ecsv_file(args_dict, logger)
    else:
        logger.error(f"File extension not supported for {args_dict['file_name']}")


if __name__ == "__main__":
    main()
