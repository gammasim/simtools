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

from pathlib import Path

import simtools.utils.general as gen
from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.db import db_handler
from simtools.io import ascii_handler

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "file_name",
        exclusive_group="group",
        exclusive_group_required=True,
        help="file to be added",
        type=str,
    ),
    cli.ArgumentDefinition(
        "input_path",
        exclusive_group="group",
        exclusive_group_required=True,
        help="A directory with json files to upload to the DB.",
        type=Path,
    ),
    cli.ArgumentDefinition(
        "db_collection", help="DB collection to which to add new values.", required=True
    ),
    cli.ArgumentDefinition(
        "test_db",
        help="Use sandbox database. Drop all data after the operation.",
        action="store_true",
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        *cli.PATH_ARGUMENTS,
    ),
    database=True,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

    if app_context.args.get("test_db", False):
        app_context.db_config["db_simulation_model_version"] = gen.get_uuid()
        app_context.logger.info(
            f"Using test database version {app_context.db_config['db_simulation_model_version']}"
        )
    db = db_handler.DatabaseHandler()

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
