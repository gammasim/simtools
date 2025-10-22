#!/usr/bin/python3
r"""
    Validate a file or files in a directory using a schema.

    Input files can be metadata, schema, or data files in yaml, json, or ecsv format.

    Command line arguments
    ----------------------
    file_name (str)
      input file to be validated
    model_parameters_directory (str)
        directory with json files of model parameters to be validated
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

        simtools-validate-file-using-schema \\
         --file_name tests/resources/MLTdata-preproduction.meta.yml \\
         --schema simtools/schemas/metadata.metaschema.yml \\
         --data_type metadata

"""

import re

from simtools.application_control import get_application_label, startup_application
from simtools.configuration import configurator
from simtools.constants import MODEL_PARAMETER_SCHEMA_PATH
from simtools.data_model import metadata_collector, schema, validate_data
from simtools.io import file_operations


def _parse():
    """Parse command line configuration."""
    config = configurator.Configurator(
        label=get_application_label(__file__),
        description="Validate a file (metadata, schema, or data file) using a schema.",
    )
    config.parser.add_argument(
        "--file_name",
        help="File to be validated (full path or name pattern, e.g., '*.json')",
        default="*.json",
    )
    config.parser.add_argument(
        "--file_directory",
        help=(
            "Directory with files to be validated. "
            "If no schema file is provided, the assumption is that model "
            "parameters are validated and the schema files are taken from "
            f"{MODEL_PARAMETER_SCHEMA_PATH}."
        ),
    )
    config.parser.add_argument("--schema", help="Json schema file", required=False)
    config.parser.add_argument(
        "--data_type",
        help="Type of input data",
        choices=["metadata", "schema", "data", "model_parameter"],
        default="data",
    )
    config.parser.add_argument(
        "--require_exact_data_type",
        help="Require exact data type for validation",
        action="store_true",
    )
    config.parser.add_argument(
        "--ignore_software_version",
        help="Ignore software version check.",
        action="store_true",
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


def validate_data_files(args_dict, logger):
    """Validate data files."""
    if args_dict.get("file_directory") is not None:
        tmp_args_dict = {}
        for file_name in file_operations.get_file_list(args_dict.get("file_directory")):
            tmp_args_dict["file_name"] = file_name
            parameter_name = re.sub(r"-\d+\.\d+\.\d+", "", file_name.stem)
            schema_file = schema.get_model_parameter_schema_file(f"{parameter_name}")
            tmp_args_dict["schema"] = schema_file
            tmp_args_dict["data_type"] = "model_parameter"
            tmp_args_dict["require_exact_data_type"] = args_dict["require_exact_data_type"]
            validate_data_file(tmp_args_dict, logger)
    else:
        validate_data_file(args_dict, logger)


def validate_data_file(args_dict, logger):
    """Validate a data file (e.g., in ecsv, json, yaml format)."""
    data_validator = validate_data.DataValidator(
        schema_file=_get_schema_file_name(args_dict),
        data_file=args_dict["file_name"],
        check_exact_data_type=args_dict["require_exact_data_type"],
    )
    data_validator.validate_and_transform(
        is_model_parameter=(args_dict["data_type"].lower() == "model_parameter")
    )

    logger.info(f"Successful validation of data file {args_dict['file_name']}")


def main():
    """Validate a file or files in a directory using a schema."""
    app_context = startup_application(_parse)

    file_name = app_context.args.get("file_name")

    if app_context.args["data_type"].lower() == "metadata":
        # metadata_collector runs the metadata validation by default, no need to do anything else
        metadata_collector.MetadataCollector(None, metadata_file_name=file_name)
        app_context.logger.info(f"Successful validation of metadata {file_name}")

    elif app_context.args["data_type"].lower() == "schema":
        schema.validate_schema_from_files(
            file_directory=app_context.args.get("file_directory"),
            file_name=file_name,
            schema_file=app_context.args.get("schema"),
            ignore_software_version=app_context.args.get("ignore_software_version", False),
        )
    else:
        validate_data_files(app_context.args, app_context.logger)


if __name__ == "__main__":
    main()
