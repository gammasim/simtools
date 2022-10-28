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
    fileName (str or list of str, required)
        Name of the file to upload including the full path. \
        A list of files is also allowed, in which case only one -f is necessary, \
        i.e., python applications/add_file_to_db.py -f file_1.dat file_2.dat file_3.dat \
        If no path is given, the file is assumed to be in the CWD.
    directory (str, required if fileName isn't given)
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

        python applications/add_file_to_db.py -f data/data-to-upload/test-data.dat
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
        answer = input("Is this OK? [y/n]").lower()

    return answer == "y"


def main():

    db = db_handler.DatabaseHandler()

    parser = argparser.CommandLineParser(description=("Add a file or files to the DB."))
    parser.initialize_default_arguments()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-f",
        "--fileName",
        help=(
            "The file name to upload. "
            "A list of files is also allowed, in which case only one -f is necessary, "
            "i.e., python applications/add_file_to_db.py -f file_1.dat file_2.dat file_3.dat "
            "If no path is given, the file is assumed to be in the CWD."
        ),
        type=str,
        nargs="+",
    )
    group.add_argument(
        "-d",
        "--directory",
        help=(
            "A directory with files to upload to the DB. "
            "All files in the directory with the following extensions "
            "will be uploaded: {}".format(", ".join(_tmp_db_config.ALLOWED_FILE_EXTENSIONS))
        ),
        type=str,
    )
    config.parser.add_argument(
        "-db",
        dest="dbToInsertTo",
        type=str,
        default=_tmp_db_config.DB_TABULATED_DATA,
        choices=[
            _tmp_db_config.DB_TABULATED_DATA,
            _tmp_db_config.DB_DERIVED_VALUES,
            _tmp_db_config.DB_REFERENCE_DATA,
            "sandbox",
            "test-data",
        ],
        help=(
            "The DB to insert the files to. "
            'The choices are {0} or "sandbox", '
            "the default is {0}".format(_tmp_db_config.DB_TABULATED_DATA)
        ),
    )
    args_dict, db_config = config.initialize(db_config=True)

    logger = logging.getLogger()
    logger.setLevel(gen.getLogLevelFromUser(args_dict["log_level"]))

    db = db_handler.DatabaseHandler(mongoDBConfig=db_config)

    filesToInsert = list()
    if args_dict.get("fileName", None) is not None:
        for fileNow in args_dict["fileName"]:
            if Path(fileNow).suffix in db.ALLOWED_FILE_EXTENSIONS:
                filesToInsert.append(fileNow)
            else:
                logger.warning(
                    "The file {} will not be uploaded to the DB because its extension is not "
                    "in the allowed extension list: {}".format(fileNow, db.ALLOWED_FILE_EXTENSIONS)
                )
    else:
        for extNow in db.ALLOWED_FILE_EXTENSIONS:
            filesToInsert.extend(Path(args_dict["directory"]).glob("*{}".format(extNow)))

    plural = "s"
    if len(filesToInsert) < 1:
        raise ValueError("No files were provided to upload")
    elif len(filesToInsert) == 1:
        plural = ""
    else:
        pass

    print(
        f"Should the following file{plural} be inserted to the {args_dict['dbToInsertTo']} DB?:\n"
    )
    print(*filesToInsert, sep="\n")
    print()
    if _userConfirm():
        for fileToInsertNow in filesToInsert:
            db.insertFileToDB(fileToInsertNow, args_dict["dbToInsertTo"])
            logger.info(f"File {fileToInsertNow} inserted to {args.dbToInsertTo} DB")
    else:
        logger.info(
            "Aborted, did not insert the file{} to the {} DB".format(
                plural, args_dict["dbToInsertTo"]
            )
        )


if __name__ == "__main__":
    main()
