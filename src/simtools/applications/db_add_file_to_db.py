#!/usr/bin/python3

"""
    Add a file to a DB.

    The name and location of the file are required.
    This application should complement the ones for updating parameters, \
    adding entries to the DB and getting files from the DB.

    Command line arguments
    ----------------------
    file_name (str or list of str, required)
        Name of the file to upload including the full path. \
        A list of files is also allowed, in which case only one -file_name is necessary, \
        i.e., python applications/db_add_file_to_db.py -file_name file_1.dat file_2.dat file_3.dat \
        If no path is given, the file is assumed to be in the CWD.
    input_path (str, required if file_name is not given)
        A directory with files to upload to the DB. \
        All files in the directory with a predefined list of extensions will be uploaded.
    db (str)
        The DB to insert the files to.

    Example
    -------
    uploading a dummy file.

    .. code-block:: console

        simtools-db-add-file-to-db --file_name test_application.dat --db test-data

    Expected final print-out message:

    .. code-block:: console

        INFO::get_file_from_db(l75)::main::Got file test_application.dat from DB test-data and
        saved into .

"""

import logging
from pathlib import Path

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.db import db_handler


def _parse():

    config = configurator.Configurator(
        description="Add file to the DB.",
        usage="simtools-add-file-to-db --file_name test_application.dat --db test-data",
    )

    group = config.parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--file_name",
        help=("The file name to upload. A list of files is also allowed."),
        type=str,
        nargs="+",
    )
    group.add_argument(
        "--input_path",
        help=("A directory with files to upload to the DB."),
        type=Path,
    )

    config.parser.add_argument(
        "--db",
        type=str,
        help=("The database to insert the files to."),
    )

    return config.initialize(paths=False, db_config=True)


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
    logger : logging.Logger
        Logger object for logging messages.
    """
    plural = "" if len(files_to_insert) == 1 else "s"

    print(f"Should the following file{plural} be inserted to the {args_dict['db']} DB?:\n")
    print(*files_to_insert, sep="\n")
    print()

    if gen.user_confirm():
        for file_to_insert_now in files_to_insert:
            db.insert_file_to_db(file_to_insert_now, args_dict["db"])
            logger.info(f"File {file_to_insert_now} inserted to {args_dict['db']} DB")
    else:
        logger.info(f"Aborted, did not insert file{plural} to the {args_dict['db']} DB")


def main():  # noqa: D103
    args_dict, db_config = _parse()

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    db = db_handler.DatabaseHandler(mongo_db_config=db_config)

    files_to_insert = collect_files_to_insert(args_dict, logger, db)
    confirm_and_insert_files(files_to_insert, args_dict, db, logger)


if __name__ == "__main__":
    main()
