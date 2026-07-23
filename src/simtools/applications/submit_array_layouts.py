#!/usr/bin/python3

r"""
    Read array layouts from file, validate with telescopes in DB, and prepare for submission.

    Validates that all telescope defined in the array layouts exist in the database for the
    specified model version. Prepares both JSON-style model parameters and corresponding
    metadata for submission.

    Command line arguments
    ----------------------
    array_layouts (str, required)
        Array layouts file.
    updated_parameter_version (str, optional)
        Updated parameter version.
    input_meta (str, optional)
        Input meta data file(s) associated to input data (wildcards or list of files allowed).
    model_version (str, required)
        Model version.

    Example
    -------

    Submit and validate a new array layout dictionary:

    .. code-block:: console

        simtools-submit-array-layouts \
            --array_layouts array_layouts.json \\
            --model_version 6.0.0 \\
            --updated_parameter_version 0.1.0 \\
            --input_meta array_layouts.metadata.yml


    """

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.db import db_handler
from simtools.io import ascii_handler
from simtools.layout.array_layout_utils import (
    validate_array_layouts_with_db,
    write_array_layouts,
)

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "array_layouts", type=str, required=True, help="Array layout dictionary file."
    ),
    cli.ArgumentDefinition(
        "input_meta",
        help="meta data file(s) associated to input data (wildcards or list of files allowed)",
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
        cli.UPDATED_PARAMETER_VERSION,
        cli.OVERWRITE_MODEL_PARAMETERS,
        cli.IGNORE_MISSING_DESIGN_MODEL,
        *cli.PATH_ARGUMENTS,
        *cli.OUTPUT_ARGUMENTS,
    ),
    database=True,
    initialize_output=True,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

    db = db_handler.DatabaseHandler()

    array_layouts = validate_array_layouts_with_db(
        production_table=db.read_production_table_from_db(
            collection_name="telescopes", model_version=app_context.args["model_version"]
        ),
        array_layouts=ascii_handler.collect_data_from_file(app_context.args["array_layouts"]),
    )

    write_array_layouts(array_layouts=array_layouts, args_dict=app_context.args)


if __name__ == "__main__":
    main()
