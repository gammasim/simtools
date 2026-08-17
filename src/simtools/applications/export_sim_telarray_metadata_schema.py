#!/usr/bin/python3
"""Export the expanded sim_telarray metadata schema."""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.io import ascii_handler
from simtools.simtel import simtel_validate_metadata

_ARGUMENTS = (
    cli.ArgumentDefinition("output_file", help="Output file name", type=str, required=False),
    cli.ArgumentDefinition(
        "source_type",
        help="Metadata source type to export",
        choices=simtel_validate_metadata.META_PARAMETER_SOURCE_TYPES,
        default="all",
    ),
    cli.ArgumentDefinition(
        "schema_version", help="Registry schema version", type=str, required=False
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        *cli.OUTPUT_PATH_ARGUMENTS,
    ),
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

    registry = simtel_validate_metadata.get_meta_parameter_registry(
        schema_version=app_context.args.get("schema_version"),
        source_type=app_context.args["source_type"],
    )

    output_file = app_context.io_handler.get_output_file(app_context.args.get("output_file"))
    app_context.logger.info(f"Writing sim_telarray metadata schema to {output_file}")
    ascii_handler.write_data_to_file(registry, output_file, sort_keys=False)


if __name__ == "__main__":
    main()
