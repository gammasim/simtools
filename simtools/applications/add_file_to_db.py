#!/usr/bin/python3

"""
    Summary
    -------
    This application adds a file to a DB.

    The name and location of the file are required.
    This application should complement the ones for updating parameters, \
    adding entries to the DB and getting files from the DB.

    Command line arguments
    ----------------------
    file_name (str or list of str, required)
        Name of the file to upload including the full path. \
        A list of files is also allowed, in which case only one -file_name is necessary, \
        i.e., python applications/add_file_to_db.py -file_name file_1.dat file_2.dat file_3.dat \
        If no path is given, the file is assumed to be in the CWD.
    input_path (str, required if file_name is not given)
        A directory with files to upload to the DB. \
        All files in the directory with a predefined list of extensions will be uploaded.
    db (str)
        The DB to insert the files to. \
        The choices are either the default CTA simulation DB or a sandbox for testing.
    verbosity (str, optional)
        Log level to print.

    Example
    -------
    uploading a dummy file.

    .. code-block:: console

        simtools-add-file-to-db --file_name test_application.dat --db test-data

    Expected final print-out message:

    .. code-block:: console

        INFO::get_file_from_db(l75)::main::Got file test_application.dat from DB test-data and
        saved into .

"""

import logging
from pathlib import Path

import simtools.utils.general as gen
from simtools import db_handler
from simtools.configuration import configurator


def _user_confirm():
    """
    Ask the user to enter y or n (case-insensitive).

    Returns
    -------
    bool: True if the answer is Y/y.
    """

    answer = ""
    while answer not in ["y", "n"]:
        try:
            answer = input("Is this OK? [y/n]").lower()
            return answer == "y"
        except EOFError:
            return False
    return False


def main():
    _db_tmp = db_handler.DatabaseHandler(mongo_db_config=None)

    config = configurator.Configurator(
        description="Add file to the DB.",
        usage="simtools-add-file-to-db --file_name test_application.dat --db test-data",
    )
    group = config.parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--file_name",
        help=(
            "The file name to upload. "
            "A list of files is also allowed, in which case only one -f is necessary, "
            "i.e., python applications/add_file_to_db.py --file_name file_1.dat file_2.dat "
            "file_3.dat. If no path is given, the file is assumed to be in the CWD."
        ),
        type=str,
        nargs="+",
    )
    group.add_argument(
        "--input_path",
        help=(
            "A directory with files to upload to the DB. "
            "All files in the directory with the following extensions "
            f"will be uploaded: {', '.join(_db_tmp.ALLOWED_FILE_EXTENSIONS)}"
        ),
        type=Path,
    )
    config.parser.add_argument(
        "--db",
        type=str,
        default=_db_tmp.DB_TABULATED_DATA,
        choices=[
            _db_tmp.DB_TABULATED_DATA,
            _db_tmp.DB_DERIVED_VALUES,
            _db_tmp.DB_REFERENCE_DATA,
            "sandbox",
            "test-data",
        ],
        help=("The database to insert the files to."),
    )
    args_dict, db_config = config.initialize(paths=False, db_config=True)

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    db = db_handler.DatabaseHandler(mongo_db_config=db_config)

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

    plural = "s"
    if len(files_to_insert) < 1:
        raise ValueError("No files were provided to upload")
    if len(files_to_insert) == 1:
        plural = ""
    else:
        pass

    print(f"Should the following file{plural} be inserted to the {args_dict['db']} DB?:\n")
    print(*files_to_insert, sep="\n")
    print()
    if _user_confirm():
        for file_to_insert_now in files_to_insert:
            db.insert_file_to_db(file_to_insert_now, args_dict["db"])
            logger.info(f"File {file_to_insert_now} inserted to {args_dict['db']} DB")
    else:
        logger.info(f"Aborted, did not insert file {plural} to the {args_dict['db']} DB")


if __name__ == "__main__":
    main()
