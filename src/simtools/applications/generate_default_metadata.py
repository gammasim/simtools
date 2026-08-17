#!/usr/bin/python3
"""Generate a default simtools metadata file from a json schema."""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.data_model import metadata_model
from simtools.io import ascii_handler

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "schema_file", help="Schema file describing input data", type=str, required=True
    ),
    cli.ArgumentDefinition(
        "output_file",
        help="output file name (if not given: print to stdout)",
        type=str,
        required=False,
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

    default_values = metadata_model.get_default_metadata_dict(app_context.args["schema_file"])

    if app_context.args["output_file"] is None:
        print(default_values)
    else:
        output_file = app_context.io_handler.get_output_file(app_context.args["output_file"])
        app_context.logger.info(f"Writing default values to {output_file}")
        ascii_handler.write_data_to_file(
            data=default_values, output_file=output_file, sort_keys=False
        )


if __name__ == "__main__":
    main()
