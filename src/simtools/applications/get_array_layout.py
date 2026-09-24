#!/usr/bin/python3

"""Get array layouts from a simulation-model repository or model repository."""

import simtools.data_model.model_data_writer as writer
from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.model.array_model import ArrayModel
from simtools.model.site_model import SiteModel

ARGUMENTS = (
    cli.ArgumentDefinition(
        "list_available_layouts",
        exclusive_group="input group",
        exclusive_group_required=False,
        help="List available layouts in the simulation-model source.",
        action="store_true",
        required=False,
    ),
    cli.ArgumentDefinition(
        "include_calibration_array_elements",
        help="Include calibration array elements in output table.",
        action="store_true",
        required=False,
    ),
    cli.ArgumentDefinition(
        "coordinate_system",
        help="Coordinate system for the array layout.",
        type=str,
        required=False,
        default="ground",
        choices=["ground", "utm"],
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    model_repository=True,
    arguments=(
        *ARGUMENTS,
        cli.MODEL_VERSION,
        cli.OVERWRITE_MODEL_PARAMETERS,
        cli.SITE,
        *cli.layout_selection_arguments(required=False),
        *cli.OUTPUT_PATH_ARGUMENTS,
        *cli.OUTPUT_ARGUMENTS,
    ),
    initialize_output=True,
)


def _layout_from_source(args_dict, model_reader):
    """Read array elements and positions using the configured model source."""
    array_model = ArrayModel(
        model_version=args_dict["model_version"],
        site=args_dict["site"],
        layout_name=args_dict.get("array_layout_name"),
        array_elements=args_dict.get("array_element_list"),
        model_reader=model_reader,
    )
    return array_model.export_array_elements_as_table(
        coordinate_system=args_dict["coordinate_system"],
        include_calibration_array_elements=args_dict["include_calibration_array_elements"],
    )


def run(app_context):
    """Run array-layout retrieval using an initialized application context."""
    if app_context.args.get("list_available_layouts", False):
        if app_context.args.get("site") is None:
            raise ValueError("Site must be provided to list available layouts.")
        site_model = SiteModel(
            model_version=app_context.args["model_version"],
            site=app_context.args["site"],
            model_reader=app_context.model_reader,
        )
        print(site_model.get_list_of_array_layouts())
        return

    app_context.logger.info("Array layout: %s", app_context.args["array_layout_name"])
    layout = _layout_from_source(app_context.args, app_context.model_reader)
    layout.pprint()
    if not app_context.args.get("output_file_from_default", False):
        writer.ModelDataWriter.write_product_data(
            output_file=app_context.args["output_file"],
            output_file_format=app_context.args.get("output_file_format"),
            metadata=None,
            product_data=layout,
        )


def main():
    """See CLI description."""
    run(APPLICATION.start())


if __name__ == "__main__":
    main()
