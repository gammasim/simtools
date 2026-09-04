#!/usr/bin/python3
"""Add parameters and production tables from a simulation model repository to a new database."""

from pathlib import Path

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.db import db_handler, db_model_upload
from simtools.settings import config

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "input_path", help="Path to simulation model repository.", type=Path, required=True
    ),
    cli.ArgumentDefinition(
        "type",
        help="Type of data to be uploaded to the database.",
        type=str,
        required=False,
        default="model_parameters",
        choices=["model_parameters", "production_tables"],
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        *cli.OUTPUT_PATH_ARGUMENTS,
        *cli.OUTPUT_ARGUMENTS,
    ),
    database=True,
    initialize_model_reader=False,
    use_dependency_defaults=False,
    initialize_output=True,
    setup_io_handler=False,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

    if app_context.args.get("db_simulation_model") and app_context.args.get(
        "db_simulation_model_tag"
    ):
        app_context.db_config["db_simulation_model"] = app_context.args["db_simulation_model"]
        app_context.db_config["db_simulation_model_tag"] = app_context.args[
            "db_simulation_model_tag"
        ]
        config.load(app_context.args, app_context.db_config)
    else:
        raise ValueError("Both db_simulation_model and db_simulation_model_tag are required.")

    db = db_handler.DatabaseHandler()
    db.require_mongodb("Adding a simulation model to a database")

    if app_context.args.get("type") == "model_parameters":
        db_model_upload.add_model_parameters_to_db(
            input_path=Path(app_context.args["input_path"]), db=db
        )
    elif app_context.args.get("type") == "production_tables":
        db_model_upload.add_production_tables_to_db(
            input_path=Path(app_context.args["input_path"]), db=db
        )


if __name__ == "__main__":
    main()
