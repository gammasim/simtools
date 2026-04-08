#!/usr/bin/python3
r"""
    Add parameters and production tables from a simulation model repository to a new database.

    Generates a new database with all required collections.
    Follows the structure of the CTAO gitlab simulation model repository.

    This is an application for DB maintainers and should not be used by the general user.

    Command line arguments

    input_path (str, required)
        Path of local copy of model parameter repository.
    db_simulation_model (str, required)
        Name of new DB to be created.
    db_simulation_model_version (str, required)
        Version of the new DB to be created.
    type (str, optional)
        Type of data to be uploaded to the DB. Options are: model_parameters, production_tables.

    Examples
    --------
    Upload model data repository to the DB
    Loops over all subdirectories in 'input_path' and uploads all json files to the
    database (or updates an existing database with the same name):

    * subdirectories starting with 'OBS' are uploaded to the 'sites' collection
    * json files from the subdirectory 'configuration_sim_telarray/configuration_corsika'
      are uploaded to the 'configuration_sim_telarray/configuration_corsika' collection
    * 'Files' are added to the 'files' collection
    * all other json files are uploaded to collection defined in the array element description
      in 'simtools/schemas/array_elements.yml'

    .. code-block:: console

        simtools-db-simulation-model-from-repository-to-db \
            --input_path /path/to/repository \
            --db_simulation_model database name \
            --db_simulation_model_version new database version \
            --type model_parameters

    Upload production tables to the DB:

    .. code-block:: console

        simtools-db-simulation-model-from-repository-to-db \
            --input_path /path/to/repository \
            --db_simulation_model database name \
            --db_simulation_model_version new database version \
            --type production_tables

"""

from pathlib import Path

from simtools.application_control import build_application
from simtools.db import db_handler, db_model_upload
from simtools.settings import config


def _add_arguments(parser):
    """Register application-specific command line arguments."""
    parser.add_argument(
        "--input_path",
        help="Path to simulation model repository.",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--type",
        help="Type of data to be uploaded to the database.",
        type=str,
        required=False,
        default="model_parameters",
        choices=["model_parameters", "production_tables"],
    )


def main():
    """See CLI description."""
    app_context = build_application(
        __file__,
        description=__doc__,
        add_arguments_function=_add_arguments,
        initialization_kwargs={
            "output": True,
            "require_command_line": True,
            "db_config": True,
        },
        startup_kwargs={"setup_io_handler": False},
    )

    if app_context.args.get("db_simulation_model") and app_context.args.get(
        "db_simulation_model_version"
    ):
        app_context.db_config["db_simulation_model"] = app_context.args["db_simulation_model"]
        app_context.db_config["db_simulation_model_version"] = app_context.args[
            "db_simulation_model_version"
        ]
        config.load(app_context.args, app_context.db_config)
    else:
        raise ValueError("Both db_simulation_model and db_simulation_model_version are required.")

    db = db_handler.DatabaseHandler()

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
