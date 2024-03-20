#!/usr/bin/python3
"""
    Summary
    -------
    Validate a file using a schema.
    Input files can be metadata, schema, or data files in yaml, json, or ecsv format.

    Command line arguments
    ----------------------
    file_name (str)
      input file to be validated
    schema (str)
      schema file (jsonschema format) used for validation
    data_type (str)
        type of input data (allowed types: metadata, schema, data)

    Raises
    ------
    FileNotFoundError
      if file to be validated is not found

    Example
    -------

    .. code-block:: console

        simtools-validate-file-using-schema \
         --file_name tests/resources/MLTdata-preproduction.meta.yml \
         --schema simtools/schemas/metadata.metaschema.yml \
         --data_type metadata

"""

import logging
from pathlib import Path

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.data_model import metadata_collector, metadata_model, validate_data


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
        "--data_type",
        help="type of input data",
        choices=["metadata", "schema", "data"],
        default="data",
    )
    return config.initialize(paths=False)


def _get_schema_file_name(args_dict, data_dict=None):
    """
    Get schema file name from metadata, data dict, or from command line argument.

    Parameters
    ----------
    args_dict (dict)
        command line arguments
    data_dict (dict)
        dictionary with metaschema information

    Returns
    -------
    schema_file: str
        schema file name

    """

    schema_file = args_dict.get("schema")
    if schema_file is None and data_dict is not None:
        schema_file = data_dict.get("meta_schema_url")
    if schema_file is None:
        metadata = metadata_collector.MetadataCollector(
            None, metadata_file_name=args_dict["file_name"]
        )
        schema_file = metadata.get_data_model_schema_file_name()
    return schema_file


def validate_schema(args_dict, logger):
    """
    Validate a schema file given in yaml or json format.
    Schema is either given as command line argument, read from the meta_schema_url or from
    the metadata section of the data dictionary.

    """

    try:
        data = gen.collect_data_from_file_or_dict(file_name=args_dict["file_name"], in_dict=None)
    except FileNotFoundError as exc:
        logger.error(f"Error reading schema file from {args_dict['file_name']}")
        raise exc
    metadata_model.validate_schema(data, _get_schema_file_name(args_dict, data))
    logger.info(f"Successful validation of schema file {args_dict['file_name']}")


def validate_data_file(args_dict, logger):
    """
    Validate a data file (e.g., in ecsv, json, yaml format)

    """
    data_validator = validate_data.DataValidator(
        schema_file=_get_schema_file_name(args_dict),
        data_file=args_dict["file_name"],
    )
    data_validator.validate_and_transform()
    logger.info(f"Successful validation of data file {args_dict['file_name']}")


def validate_metadata(args_dict, logger):
    """
    Validate metadata.

    """
    # metadata_collector runs the metadata validation by default, no need to do anything else
    metadata_collector.MetadataCollector(None, metadata_file_name=args_dict["file_name"])
    logger.info(f"Successful validation of metadata {args_dict['file_name']}")


def main():
    label = Path(__file__).stem
    args_dict, _ = _parse(
        label, description="Validate a file (metadata, schema, or data file) using a schema."
    )

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    if args_dict["data_type"].lower() == "metadata":
        validate_metadata(args_dict, logger)
    elif args_dict["data_type"].lower() == "schema":
        validate_schema(args_dict, logger)
    else:
        validate_data_file(args_dict, logger)


if __name__ == "__main__":
    main()
