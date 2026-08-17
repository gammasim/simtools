#!/usr/bin/python3
"""Convert and print a list of array element positions in different coordinate systems."""

import simtools.data_model.model_data_writer as writer
from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.data_model.metadata_collector import MetadataCollector
from simtools.layout import array_layout

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "array_element_positions_file", help="List of array element positions.", required=True
    ),
    cli.ArgumentDefinition(
        "input_meta", help="meta data file associated to input data", type=str, required=False
    ),
    cli.ArgumentDefinition(
        "print",
        help="print list of positions in requested coordinate system",
        required=False,
        default="",
        choices=["ground", "utm", "mercator"],
    ),
    cli.ArgumentDefinition(
        "export",
        help="export array element list to file (in requested coordinate system)",
        required=False,
        default=None,
        choices=["ground", "utm", "mercator"],
    ),
    cli.ArgumentDefinition(
        "select_assets",
        help="select a subset of assets (e.g., MSTN, LSTN)",
        required=False,
        default=None,
        nargs="+",
    ),
    cli.ArgumentDefinition(
        "skip_input_validation",
        help="skip input data validation against schema",
        default=False,
        required=False,
        action="store_true",
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        cli.MODEL_VERSION,
        cli.PARAMETER_VERSION,
        cli.OVERWRITE_MODEL_PARAMETERS,
        cli.SITE,
        *cli.OUTPUT_PATH_ARGUMENTS,
        *cli.OUTPUT_ARGUMENTS,
    ),
    database=True,
    initialize_output=True,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

    if app_context.args.get("array_element_positions_file", "").endswith(".json"):
        site = app_context.args.get("site", None)
        metadata, validate_schema_file = None, None
    else:
        metadata = MetadataCollector(
            args_dict=app_context.args, model_parameter_name="array_coordinates"
        )
        site = metadata.get_site(from_input_meta=True) or app_context.args.get("site")
        validate_schema_file = metadata.get_data_model_schema_file_name()

    layout = array_layout.ArrayLayout(
        model_version=app_context.args["model_version"],
        site=site,
        telescope_list_file=app_context.args["array_element_positions_file"],
        telescope_list_metadata_file=app_context.args["input_meta"],
        validate=not app_context.args["skip_input_validation"],
    )
    layout.select_assets(app_context.args["select_assets"])
    layout.convert_coordinates()

    if app_context.args["export"] is not None:
        product_data = (
            layout.export_one_telescope_as_json(
                crs_name=app_context.args["export"],
                parameter_version=app_context.args.get("parameter_version"),
            )
            if app_context.args.get("array_element_positions_file", "").endswith(".json")
            else layout.export_telescope_list_table(crs_name=app_context.args["export"])
        )
        writer.ModelDataWriter.write_product_data(
            output_file=app_context.args.get("output_file"),
            output_file_format=app_context.args.get("output_file_format", "ascii.ecsv"),
            metadata=metadata,
            product_data=product_data,
            validate_schema_file=validate_schema_file,
        )
    else:
        layout.print_telescope_list(
            crs_name=app_context.args["print"],
        )


if __name__ == "__main__":
    main()
