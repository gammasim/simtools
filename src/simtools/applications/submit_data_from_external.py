#!/usr/bin/python3
"""Submit data file through the command line."""

import simtools.data_model.model_data_writer as writer
from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.data_model import validate_data
from simtools.data_model.metadata_collector import MetadataCollector

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "input_meta", help="meta data file associated to input data", type=str, required=False
    ),
    cli.ArgumentDefinition("input_data_file", help="Input data file", type=str, required=True),
    cli.ArgumentDefinition(
        "schema_file", help="Schema file describing input data", type=str, required=False
    ),
    cli.ArgumentDefinition(
        "ignore_metadata", help="Ignore metadata", action="store_true", required=False
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        *cli.OUTPUT_PATH_ARGUMENTS,
        *cli.OUTPUT_ARGUMENTS,
    ),
    initialize_output=True,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

    _metadata = (
        None if app_context.args.get("ignore_metadata") else MetadataCollector(app_context.args)
    )

    data_validator = validate_data.DataValidator(
        schema_file=(
            _metadata.get_data_model_schema_file_name()
            if _metadata
            else app_context.args.get("schema_file")
        ),
        data_file=app_context.args["input_data_file"],
    )

    writer.ModelDataWriter.write_product_data(
        output_file=app_context.args["output_file"],
        output_file_format=app_context.args.get("output_file_format"),
        metadata=_metadata,
        product_data=data_validator.validate_and_transform(),
    )


if __name__ == "__main__":
    main()
