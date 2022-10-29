#!/usr/bin/python3

"""
    Summary
    -------
    This application adds a file to the DB.

    The name and location of the file are required.
    This application should complement the ones for updating parameters \
    and adding entries to the DB.

    Command line arguments
    ----------------------
    file_name (str or list of str, required)
        Name of the file to upload including the full path. \
        A list of files is also allowed, in which case only one -f is necessary, \
        i.e., python applications/add_file_to_db.py -f file_1.dat file_2.dat file_3.dat \
        If no path is given, the file is assumed to be in the CWD.
    input_path (str, required if file_name is not given)
        A directory with files to upload to the DB. \
        All files in the directory with a predefined list of extensions will be uploaded.
    db (str)
        The DB to insert the files to. \
        The choices are either the default CTA simulation DB or a sandbox for testing.
    verbosity (str, optional)
        Log level to print (default=INFO).

    Example
    -------
    uploading a dummy file.

    .. code-block:: console

        python applications/add_file_to_db.py --file_name data/data-to-upload/test-data.dat
"""

import logging
from pathlib import Path

import simtools.configuration as configurator
import simtools.util.general as gen
from simtools import db_handler


def _userConfirm():
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


def main():

    _db_tmp = db_handler.DatabaseHandler(mongoDBConfig=None)

    config = configurator.Configurator(
        label="Add file() to the DB.",
        description="python applications/add_file_to_db.py --file_name file_1.dat file_2.dat",
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
            "will be uploaded: {}".format(", ".join(_db_tmp.ALLOWED_FILE_EXTENSIONS))
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
    logger.setLevel(gen.getLogLevelFromUser(args_dict["log_level"]))

    db = db_handler.DatabaseHandler(mongoDBConfig=db_config)

    filesToInsert = list()
    if args_dict.get("file_name", None) is not None:
        for fileNow in args_dict["file_name"]:
            if Path(fileNow).suffix in db.ALLOWED_FILE_EXTENSIONS:
                filesToInsert.append(fileNow)
            else:
                logger.warning(
                    "The file {} will not be uploaded to the DB because its extension is not "
                    "in the allowed extension list: {}".format(fileNow, db.ALLOWED_FILE_EXTENSIONS)
                )
    else:
        for extNow in db.ALLOWED_FILE_EXTENSIONS:
            filesToInsert.extend(Path(args_dict["input_path"]).glob("*{}".format(extNow)))

    plural = "s"
    if len(filesToInsert) < 1:
        raise ValueError("No files were provided to upload")
    elif len(filesToInsert) == 1:
        plural = ""
    else:
        pass

    print(f"Should the following file{plural} be inserted to the {args_dict['db']} DB?:\n")
    print(*filesToInsert, sep="\n")
    print()
    if _userConfirm():
        for fileToInsertNow in filesToInsert:
            db.insertFileToDB(fileToInsertNow, args_dict["db"])
            logger.info("File {} inserted to {} DB".format(fileToInsertNow, args_dict["db"]))
    else:
        logger.info("Aborted, did not insert file {} to the {} DB".format(plural, args_dict["db"]))


if __name__ == "__main__":
    main()
