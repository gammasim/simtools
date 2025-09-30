#!/usr/bin/python3

r"""
    Submit array-layouts definition and corresponding metadata and validate list of telescopes.

    Includes validation that all defined telescope exists.

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

    Submit a new array layout dictionary:

    .. code-block:: console

        simtools-submit-array-layouts \
            --array_layouts array_layouts.json \\
            --model_version 6.0.0 \\
            --updated_parameter_version 0.1.0 \\
            --input_meta array_layouts.metadata.yml


    """

from simtools.application_control import get_application_label, startup_application
from simtools.configuration import configurator
from simtools.db import db_handler
from simtools.io import ascii_handler
from simtools.layout.array_layout_utils import validate_array_layouts_with_db, write_array_layouts


def _parse():
    """Parse command line configuration."""
    config = configurator.Configurator(
        label=get_application_label(__file__),
        description="Submit and validate array layouts.",
    )

    config.parser.add_argument(
        "--array_layouts",
        type=str,
        required=True,
        help="Array layout dictionary file.",
    )
    config.parser.add_argument(
        "--updated_parameter_version",
        help="Updated parameter version.",
        type=str,
        required=False,
    )
    config.parser.add_argument(
        "--input_meta",
        help="meta data file(s) associated to input data (wildcards or list of files allowed)",
        type=str,
        nargs="+",
        required=False,
    )

    return config.initialize(output=True, db_config=True, simulation_model=["model_version"])


def main():
    """Submit and validate array layouts."""
    args_dict, db_config, _, _ = startup_application(_parse)

    db = db_handler.DatabaseHandler(mongo_db_config=db_config)

    array_layouts = validate_array_layouts_with_db(
        production_table=db.read_production_table_from_mongo_db(
            collection_name="telescopes", model_version=args_dict["model_version"]
        ),
        array_layouts=ascii_handler.collect_data_from_file(args_dict["array_layouts"]),
    )

    write_array_layouts(array_layouts=array_layouts, args_dict=args_dict, db_config=db_config)


if __name__ == "__main__":
    main()
