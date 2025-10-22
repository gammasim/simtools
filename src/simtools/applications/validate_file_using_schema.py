#!/usr/bin/python3
r"""
    Validate a file or files in a directory using a schema.

    Input files can be metadata, schema, or data files in yaml, json, or ecsv format.
    For model parameters, the schema files are taken from the simtools model parameter
    schema directory by default.

    Command line arguments
    ----------------------
    file_name (str)
      input file to be validated
    file_directory (str)
        directory with json files of model parameters to be validated
    schema (str)
      schema file (jsonschema format) used for validation
    data_type (str)
        type of input data (allowed types: metadata, schema, data, model_parameter)

    Example
    -------

    Validate metadata of a file:

    .. code-block:: console

        simtools-validate-file-using-schema \\
         --file_name tests/resources/MLTdata-preproduction.meta.yml \\
         --schema simtools/schemas/metadata.metaschema.yml \\
         --data_type metadata

    Validate schema of a file:

    .. code-block:: console

        simtools-validate-file-using-schema \\
         --file_name tests/resources/model_parameters/schema-0.3.0/num_gains-1.0.0.json \\
         --schema src/simtools/schemas/model_parameter.metaschema.yml \\
         --data_type schema

    Validate all model parameter files in a directory:

    .. code-block:: console

        simtools-validate-file-using-schema \\
         --file_directory tests/resources/model_parameters \\
         --data_type model_parameter

"""

from simtools.application_control import get_application_label, startup_application
from simtools.configuration import configurator
from simtools.constants import MODEL_PARAMETER_SCHEMA_PATH
from simtools.data_model import metadata_collector, schema, validate_data


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
    config.parser.add_argument("--schema", help="Schema file", required=False)
    config.parser.add_argument(
        "--data_type",
        help="Type of input data",
        choices=["metadata", "schema", "data", "model_parameter"],
        default="data",
    )
    config.parser.add_argument(
        "--check_exact_data_type",
        help="Require exact data type for validation",
        action="store_true",
    )
    config.parser.add_argument(
        "--ignore_software_version",
        help="Ignore software version check.",
        action="store_true",
    )
    return config.initialize(paths=False)


def main():
    """Validate a file or files in a directory using a schema."""
    app_context = startup_application(_parse)

    file_name = app_context.args.get("file_name")
    file_directory = app_context.args.get("file_directory")
    data_type = app_context.args.get("data_type").lower()

    if data_type == "metadata":
        # metadata_collector runs the metadata validation by default, no need to do anything else
        metadata_collector.MetadataCollector(None, metadata_file_name=file_name)
        app_context.logger.info(f"Successful validation of metadata {file_name}")

    elif data_type == "schema":
        schema.validate_schema_from_files(
            file_directory=file_directory,
            file_name=file_name,
            schema_file=app_context.args.get("schema"),
            ignore_software_version=app_context.args.get("ignore_software_version", False),
        )
    else:
        validate_data.DataValidator.validate_data_files(
            file_name=file_name,
            file_directory=file_directory,
            is_model_parameter=(data_type == "model_parameter"),
            check_exact_data_type=app_context.args.get("require_exact_data_type", False),
            schema_file=app_context.args.get("schema"),
        )


if __name__ == "__main__":
    main()
