#!/usr/bin/python3

"""Add a file to a DB."""

from pathlib import Path

import simtools.utils.general as gen
from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.db import db_handler

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "file_name",
        exclusive_group="group",
        exclusive_group_required=True,
        help="The file name to upload. A list of files is also allowed.",
        type=str,
        nargs="+",
    ),
    cli.ArgumentDefinition(
        "input_path",
        exclusive_group="group",
        exclusive_group_required=True,
        help="A directory with files to upload to the DB.",
        type=Path,
    ),
    cli.DATABASE_NAME(required=True),
    cli.ArgumentDefinition(
        "test_db",
        help="Use sandbox database. Drop all data after the operation.",
        action="store_true",
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(*_ARGUMENTS,),
    database=True,
    initialize_model_reader=False,
    setup_io_handler=False,
)


def collect_files_to_insert(args_dict, logger, db):
    """
    Collect the files to insert into the database based on the provided arguments.

    Parameters
    ----------
    args_dict : dict
        Dictionary of parsed command-line arguments.
    logger : logging.Logger
        Logger object for logging messages.
    db : DatabaseHandler
        Database handler object.

    Returns
    -------
    list
        List of files to be inserted into the database.

    Raises
    ------
    ValueError
        If no valid files are provided for uploading.
    """
    files_to_insert = []

    if args_dict.get("file_name", None) is not None:
        for file_now in args_dict["file_name"]:
            if Path(file_now).suffix in db.ALLOWED_FILE_EXTENSIONS:
                files_to_insert.append(file_now)
            else:
                logger.warning(
                    f"The file {file_now} will not be uploaded to the DB because its extension "
                    f"is not in the allowed extension list: {db.ALLOWED_FILE_EXTENSIONS}"
                )
    else:
        for ext_now in db.ALLOWED_FILE_EXTENSIONS:
            files_to_insert.extend(Path(args_dict["input_path"]).glob(f"*{ext_now}"))

    if not files_to_insert:
        raise ValueError("No files were provided to upload")

    return files_to_insert


def confirm_and_insert_files(files_to_insert, args_dict, db, logger):
    """
    Confirm the files to be inserted and insert them into the database.

    Parameters
    ----------
    files_to_insert : list
        List of files to be inserted into the database.
    args_dict : dict
        Dictionary of parsed command-line arguments.
    db : DatabaseHandler
        Database handler object.
    logger : logging.logger
        logger object for logging messages.
    """
    plural = "" if len(files_to_insert) == 1 else "s"

    if args_dict.get("test_db", False):
        args_dict["database_name"] = args_dict["database_name"] + gen.get_uuid()
        logger.info(f"Using test database: {args_dict['database_name']}")

    print(
        f"Should the following file{plural} be inserted to the {args_dict['database_name']} DB?:\n"
    )
    print(*files_to_insert, sep="\n")
    print()

    if gen.user_confirm():
        for file_to_insert_now in files_to_insert:
            db.insert_file_to_db(file_to_insert_now, args_dict["database_name"])
            logger.info(f"File {file_to_insert_now} inserted to {args_dict['database_name']} DB")
    else:
        logger.info(f"Aborted, did not insert file{plural} to the {args_dict['database_name']} DB")

    # drop test database; be safe and required DB name is sandbox
    if args_dict.get("test_db", False) and "sandbox" in args_dict["database_name"]:
        logger.info(f"Test database used. Dropping all data from {args_dict['database_name']}")
        db.mongo_db_handler.db_client.drop_database(args_dict["database_name"])


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

    db = db_handler.DatabaseHandler()
    db.require_mongodb("Adding files to a database")

    files_to_insert = collect_files_to_insert(app_context.args, app_context.logger, db)
    confirm_and_insert_files(files_to_insert, app_context.args, db, app_context.logger)


if __name__ == "__main__":
    main()
