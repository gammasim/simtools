#!/usr/bin/python3
r"""
    Generate a default simtools metadata file from a json schema.

    Command line arguments
    ----------------------
    schema (str, optional)
        Schema file describing the input data
        (default: simtools/schemas/metadata.metaschema.yml)
    output_file (str, optional)
        Output file name.

    Example
    -------
    .. code-block:: console

        simtools-generate-default-metadata \\
            --schema simtools/schemas/metadata.metaschema.yml \\
            --output_file default_metadata.yml


    """

from simtools.application_control import build_application
from simtools.data_model import metadata_model
from simtools.io import ascii_handler


def _add_arguments(parser):
    """Register application-specific command line arguments."""
    parser.add_argument(
        "--schema",
        help="schema file describing input data",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_file",
        help="output file name (if not given: print to stdout)",
        type=str,
        required=False,
    )


def main():
    """Generate a default simtools metadata file from a json schema."""
    app_context = build_application(
        __file__,
        description="Generate a default simtools metadata file from a json schema.",
        add_arguments_function=_add_arguments,
        initialization_kwargs={"output": False, "require_command_line": True},
    )

    default_values = metadata_model.get_default_metadata_dict(app_context.args["schema"])

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
