#!/usr/bin/python3

"""Submit an array-layout parameter after validating its telescope elements."""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.layout.array_layout_utils import (
    prepare_array_layouts_for_submission,
    validate_array_layouts_with_db,
    write_array_layouts,
)

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "array_layouts",
        type=str,
        required=False,
        help="Complete canonical array-layout parameter file (legacy input).",
    ),
    cli.ArgumentDefinition(
        "array_layout_name",
        type=str,
        required=False,
        help="Name of the new array layout.",
    ),
    cli.ArgumentDefinition(
        "array_element_list",
        type=str,
        required=False,
        nargs="+",
        help="Telescope names in the new array layout.",
    ),
    cli.ArgumentDefinition(
        "reference_array_layout",
        type=str,
        required=False,
        default="hyper_array",
        help="Existing layout that limits selectable telescopes.",
    ),
    cli.ArgumentDefinition(
        "input_meta",
        help="Meta data file(s) associated to input data.",
        type=str,
        required=False,
        nargs="+",
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        cli.MODEL_VERSION,
        cli.PARAMETER_VERSION,
        cli.UPDATED_PARAMETER_VERSION,
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
    args_dict = app_context.args
    model_reader = app_context.model_reader
    array_layouts, model_version = prepare_array_layouts_for_submission(model_reader, args_dict)
    array_layouts = validate_array_layouts_with_db(
        production_table=model_reader.read_production_table(
            collection_name="telescopes", model_version=model_version
        ),
        array_layouts=array_layouts,
    )

    write_array_layouts(array_layouts=array_layouts, args_dict=args_dict)


if __name__ == "__main__":
    main()
