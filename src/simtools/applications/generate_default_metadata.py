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

from simtools.application_startup import get_application_label, startup_application
from simtools.configuration import configurator
from simtools.data_model import metadata_model
from simtools.io import ascii_handler, io_handler


def _parse():
    """Parse command line configuration."""
    config = configurator.Configurator(
        label=get_application_label(__file__),
        description="Generate a default simtools metadata file from a json schema.",
    )

    config.parser.add_argument(
        "--schema",
        help="schema file describing input data",
        type=str,
        required=True,
    )
    config.parser.add_argument(
        "--output_file",
        help="output file name (if not given: print to stdout)",
        type=str,
        required=False,
    )

    return config.initialize(output=False, require_command_line=True)


def main():  # noqa: D103
    args_dict, _, logger, _ = startup_application(_parse, setup_io_handler=False)

    default_values = metadata_model.get_default_metadata_dict(args_dict["schema"])

    if args_dict["output_file"] is None:
        print(default_values)
    else:
        _io_handler = io_handler.IOHandler()
        output_file = _io_handler.get_output_file(args_dict["output_file"])
        logger.info(f"Writing default values to {output_file}")
        ascii_handler.write_data_to_file(
            data=default_values, output_file=output_file, sort_keys=False
        )


if __name__ == "__main__":
    main()
