#!/usr/bin/python3

"""
    Summary
    -------
    This application gets a file from the DB.

    The name and location of the file are required.
    This application should complement the ones for getting parameters \
    and adding entries and files to the DB.

    Command line arguments
    ----------------------
    fileNames (str or list of str, required)
        Name of the file(s) to get including the full path. \
        i.e., python applications/get_file_from_db.py -f file_1.dat.
    outputFolder (str)
        Name of the local output directory where to save the files. \
        If it does not exist, it will be created.
    db (str)
        The DB to get the files from. \
        The choices are either the default CTA simulation DB or a sandbox for testing.
    verbosity (str, optional)
        Log level to print (default=INFO).

    Example
    -------
    retrieving a dummy file.

    .. code-block:: console

        python applications/get_file_from_db.py -f test-data.dat
"""

import logging
from pathlib import Path

import simtools.config as cfg
import simtools.util.commandline_parser as argparser
import simtools.util.general as gen
from simtools import db_handler


def main():

    db = db_handler.DatabaseHandler()

    parser = argparser.CommandLineParser(description=("Get a file or files from the DB."))
    parser.add_argument(
        "-f",
        "--fileNames",
        help=(
            "The file name(s) to download. "
            "i.e., python applications/get_file_from_db.py -f file_1.dat"
        ),
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "-out",
        dest="outputFolder",
        type=str,
        default=".",
        help=(
            "The local output folder DB to save the files from the DB. " "The default is the CWD"
        ),
    )
    parser.add_argument(
        "-db",
        dest="dbToGetFrom",
        type=str,
        default=db.DB_TABULATED_DATA,
        choices=[
            db.DB_TABULATED_DATA,
            db.DB_DERIVED_VALUES,
            db.DB_REFERENCE_DATA,
            "sandbox",
            "test-data",
        ],
        help=(
            "The DB to get the files from. "
            'The choices are {0} or "sandbox", '
            "the default is {0}".format(db.DB_TABULATED_DATA)
        ),
    )
    parser.initialize_default_arguments()
    args = parser.parse_args()
    cfg.setConfigFileName(args.configFile)

    logger = logging.getLogger()
    logger.setLevel(gen.getLogLevelFromUser(args.logLevel))

    output_path = Path(args.outputFolder)
    if output_path.exists() == False:
        print("Folder{} does not exist" "It will be created now".format(args.outputFolder))
        output_path.mkdir(parents=True, exist_ok=True)

    if args.fileNames is not None:
        if len(args.fileNames) > 0:
            db.exportFilesDB(args.dbToGetFrom, args.outputFolder, args.fileNames)
        elif len(args.fileNames) == 0:
            db.exportFileDB(args.dbToGetFrom, args.outputFolder, args.fileNames)
        else:
            print(
                "Aborted, {} invalid. Valid formats are: str or list of str.".format(args.fileNames)
            )

    else:
        raise ValueError("No files were provided to download")


if __name__ == "__main__":
    main()
