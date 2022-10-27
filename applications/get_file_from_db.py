#!/usr/bin/python3

"""
    Summary
    -------
    This application gets a file from the DB.

    The name of the file is required.
    This application should complement the ones for getting parameters, adding entries and files \
    to the DB.

    Command line arguments
    ----------------------
    fileName (str or list of str, required)
        Name of the file to get including its full directory. A list of files is also allowed.
        i.e., python applications/get_file_from_db.py -f mirror_CTA-N-LST1_v2019-03-31.dat.
    outputDirectory (str)
        Name of the local output directory where to save the files.
        Default it $CWD.
    verbosity (str, optional)
        Log level to print (default=INFO).

    Example
    -------
    getting a file from DB.

    .. code-block:: console

        python applications/get_file_from_db.py -f mirror_CTA-N-LST1_v2019-03-31.dat
"""

import logging
from pathlib import Path

import simtools.config as cfg
import simtools.util.commandline_parser as argparser
import simtools.util.general as gen
from simtools import db_handler


def main():

    parser = argparser.CommandLineParser(description=("Get a file or files from the DB."))

    parser.add_argument(
        "-f",
        "--fileName",
        help=(
            "The file name to download. "
            "i.e., python applications/get_file_from_db.py -f mirror_CTA-S-LST_v2020-04-07.dat"
        ),
        type=str,
    )
    parser.add_argument(
        "-out",
        "--outputDirectory",
        dest="outputDirectory",
        type=str,
        default=".",
        help=(
            "The local output directory DB to save the files from the DB. The default is the CWD"
        ),
    )

    parser.initialize_default_arguments()
    args = parser.parse_args()
    cfg.setConfigFileName(args.configFile)

    logger = logging.getLogger()
    logger.setLevel(gen.getLogLevelFromUser(args.logLevel))

    db = db_handler.DatabaseHandler()
    availableDbs = [
        db.DB_TABULATED_DATA,
        db.DB_CTA_SIMULATION_MODEL,
        db.DB_CTA_SIMULATION_MODEL_DESCRIPTIONS,
        db.DB_REFERENCE_DATA,
        db.DB_DERIVED_VALUES,
        "sandbox",
        "test-data",
    ]
    outputPath = Path(args.outputDirectory)
    fileId = None
    if outputPath.exists():
        for dbName in availableDbs:
            try:
                fileId = db.exportFileDB(dbName, args.outputDirectory, args.fileName)
                print(
                    "Got file {} from DB {} and saved into {}".format(
                        args.fileName, dbName, args.outputDirectory
                    )
                )
                break
            except FileNotFoundError:
                continue

        if fileId is None:
            raise FileNotFoundError(
                "The file {} was not found in any of the available DBs."
            ).format(args.fileName)

    else:
        print("Aborted, directory {} does not exist".format(args.outputDirectory))


if __name__ == "__main__":
    main()
