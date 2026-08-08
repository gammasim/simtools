#!/usr/bin/python3
r"""Validate data, metadata, and schemas against simtools schemas."""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.data_model import metadata_collector, schema, validate_data

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "file_name",
        help="File to be validated (full path or name pattern, e.g., '*.json')",
        default="*.json",
    ),
    cli.ArgumentDefinition(
        "file_directory",
        help=(
            "Directory with files to validate. Without a schema file, model parameters are "
            "assumed and the bundled model-parameter schemas are used."
        ),
    ),
    cli.ArgumentDefinition("schema_file", help="Schema file", required=False),
    cli.ArgumentDefinition(
        "data_type",
        help="Type of input data",
        choices=["metadata", "schema", "data", "model_parameter"],
        default="data",
    ),
    cli.ArgumentDefinition(
        "check_exact_data_type", help="Require exact data type for validation", action="store_true"
    ),
    cli.ArgumentDefinition(
        "ignore_software_version", help="Ignore software version check.", action="store_true"
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(*_ARGUMENTS,),
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

    file_name = app_context.args.get("file_name")
    file_directory = app_context.args.get("file_directory")
    schema_file = app_context.args.get("schema_file")
    data_type = app_context.args.get("data_type").lower()

    if data_type == "metadata":
        # metadata_collector runs the metadata validation by default, no need to do anything else
        metadata_collector.MetadataCollector(None, metadata_file_name=file_name)
        app_context.logger.info(f"Successful validation of metadata {file_name}")

    elif data_type == "schema":
        schema.validate_schema_from_files(
            file_directory=file_directory,
            file_name=file_name,
            schema_file=schema_file,
            ignore_software_version=app_context.args.get("ignore_software_version", False),
        )
    else:
        validate_data.DataValidator.validate_data_files(
            file_name=file_name,
            file_directory=file_directory,
            is_model_parameter=(data_type == "model_parameter"),
            check_exact_data_type=app_context.args.get("check_exact_data_type", False),
            schema_file=schema_file,
        )


if __name__ == "__main__":
    main()
