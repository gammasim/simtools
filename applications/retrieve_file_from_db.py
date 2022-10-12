#!/usr/bin/python3

"""
    Summary
    -------
    This application retrieves a file from the DB.

    The name and location of the file are required.
    This application should complement the ones for getting parameters \
    and adding entries and files to the DB.

    Command line arguments
    ----------------------
    fileName (str, required)
        Name of the file to retrieve including the full path. \
        i.e., python applications/retrieve_file_from_db.py -f file_1.dat.
    outputFolder (str)
        Name of the local output directory where to save the files. \
        If it does not exist, it will be created.
    db (str)
        The DB to retrieve the files from. \
        The choices are either the default CTA simulation DB or a sandbox for testing.
    verbosity (str, optional)
        Log level to print (default=INFO).

    Example
    -------
    retrieving a dummy file.

    .. code-block:: console

        python applications/retrieve_file_from_db.py -f data/data-to-download/test-data.dat
"""

import logging
from pathlib import Path

import simtools.config as cfg
import simtools.util.commandline_parser as argparser
import simtools.util.general as gen
from simtools import db_handler


def main():

    db = db_handler.DatabaseHandler()

    parser = argparser.CommandLineParser(description=("Retrieve a file or files from the DB."))
    parser.add_argument(
        "-f",
        "--fileName",
        help=(
            "The file name to download. "
            "i.e., python applications/retrieve_file_from_db.py -f file_1.dat"
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
        dest="dbToRetrieveFrom",
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
            "The DB to retrieve the files from. "
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

    if args.fileName is not None:
        print(
            "File{} is going to be downloaded from {} DB and saved to {}".format(
                args.fileName, args.dbToRetrieveFrom, args.outputFolder
            )
        )
        db.exportFileDB(args.dbToRetrieveFrom, args.outputFolder, args.fileName)
        if output_path.joinpath(args.fileName).exists():
            logger.info(
                "File{} was downloaded from {} DB".format(args.fileName, args.dbToRetrieveFrom)
            )
        else:
            logger.info(
                "File{} was not downloaded from the {} DB".format(args.fileName, args.dbToInsertTo)
            )
    else:
        raise ValueError("No files were provided to download")


if __name__ == "__main__":
    main()
