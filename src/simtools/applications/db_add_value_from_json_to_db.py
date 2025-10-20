#!/usr/bin/python3

r"""
    Add a new parameter / value to a collection in the DB using a json file as input.

    Command line arguments
    ----------------------
    file_name (str, required)
        Name of the file to upload including the full path.
    db_collection (str, required)
        The DB collection to which to add the file.
    db (str)
        The DB to insert the files to.

    Example
    -------

    Upload a file to sites collection:

    .. code-block:: console

        simtools-add-value-from-json-to-db \\
            --file_name new_value.json --db_collection sites


"""

import uuid
from pathlib import Path

import simtools.utils.general as gen
from simtools.application_control import get_application_label, startup_application
from simtools.configuration import configurator
from simtools.db import db_handler
from simtools.io import ascii_handler


def _parse():
    """Parse command line configuration."""
    config = configurator.Configurator(
        label=get_application_label(__file__), description="Add a new parameter to the DB."
    )
    group = config.parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file_name", help="file to be added", type=str)
    group.add_argument(
        "--input_path",
        help="A directory with json files to upload to the DB.",
        type=Path,
    )
    config.parser.add_argument(
        "--db_collection", help="DB collection to which to add new values.", required=True
    )
    config.parser.add_argument(
        "--test_db",
        help="Use sandbox database. Drop all data after the operation.",
        action="store_true",
    )
    return config.initialize(db_config=True)


def main():
    """Add value from JSON to database."""
    app_context = startup_application(_parse)

    if app_context.args.get("test_db", False):
        app_context.db_config["db_simulation_model_version"] = str(uuid.uuid4())
        app_context.logger.info(
            f"Using test database version {app_context.db_config['db_simulation_model_version']}"
        )
    db = db_handler.DatabaseHandler(db_config=app_context.db_config)

    files_to_insert = []
    if app_context.args.get("file_name", None) is not None:
        files_to_insert.append(app_context.args["file_name"])
    else:
        files_to_insert.extend(Path(app_context.args["input_path"]).glob("*json"))

    if len(files_to_insert) < 1:
        raise ValueError("No files were provided to upload")
    plural = "s" if len(files_to_insert) > 1 else ""

    print(
        f"Should the following parameter{plural} be inserted to the "
        f"{app_context.args['db_collection']} DB collection?:\n"
    )
    print(*files_to_insert, sep="\n")
    print()

    app_context.logger.info(f"DB {db.get_db_name()} selected.")

    if gen.user_confirm():
        for file_to_insert_now in files_to_insert:
            par_dict = ascii_handler.collect_data_from_file(file_name=file_to_insert_now)
            app_context.logger.info(
                f"Adding the following parameter to the DB: {par_dict['parameter']}"
            )
            db.add_new_parameter(
                par_dict=par_dict,
                collection_name=app_context.args["db_collection"],
                file_prefix="./",
            )
            app_context.logger.info(
                f"Value for {par_dict['parameter']} added to "
                f"{app_context.args['db_collection']} collection."
            )
    else:
        app_context.logger.info("Aborted, no change applied to the database")

    # drop test database; be safe and required DB name is sandbox
    if app_context.args.get("test_db", False) and "sandbox" in db.get_db_name():
        app_context.logger.info(f"Test database used. Dropping all data from {db.get_db_name()}")
        db.mongo_db_handler.db_client.drop_database(db.get_db_name())


if __name__ == "__main__":
    main()
