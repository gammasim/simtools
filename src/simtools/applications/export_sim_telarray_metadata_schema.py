#!/usr/bin/python3
r"""
Export the expanded sim_telarray metadata schema.

The exported schema combines generated metadata definitions from
``sim_telarray_meta_parameters.schema.yml`` with metadata derived from
model-parameter schemas.

Command line arguments
----------------------
output_file (str)
    Output file name.
source_type (str, optional)
    Export all metadata, only generated metadata, or only model-parameter-derived metadata.
schema_version (str, optional)
    Registry schema version.

Example
-------
.. code-block:: console

    simtools-export-sim-telarray-metadata-schema --output_file metadata.yml

"""

from simtools.application_control import build_application
from simtools.io import ascii_handler
from simtools.simtel import simtel_validate_metadata


def _add_arguments(parser):
    """Register application-specific command line arguments."""
    parser.add_argument(
        "--output_file",
        help="Output file name",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--source_type",
        help="Metadata source type to export",
        choices=simtel_validate_metadata.META_PARAMETER_SOURCE_TYPES,
        default="all",
    )
    parser.add_argument(
        "--schema_version",
        help="Registry schema version",
        type=str,
        required=False,
    )


def main():
    """See CLI description."""
    app_context = build_application(
        initialization_kwargs={"output": False, "require_command_line": True},
    )

    registry = simtel_validate_metadata.get_meta_parameter_registry(
        schema_version=app_context.args.get("schema_version"),
        source_type=app_context.args["source_type"],
    )

    output_file = app_context.io_handler.get_output_file(app_context.args.get("output_file"))
    app_context.logger.info(f"Writing sim_telarray metadata schema to {output_file}")
    ascii_handler.write_data_to_file(registry, output_file, sort_keys=False)


if __name__ == "__main__":
    main()
