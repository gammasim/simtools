#!/usr/bin/python3

"""
    Summary
    -------
    Get a file from the DB.

    The name of the file is required.
    This application complements the ones for getting parameters, adding entries and files \
    to the DB.

    Command line arguments
    ----------------------
    file_name (str or list of str, required)
        Name of the file to get including its full directory. A list of files is also allowed.
        i.e., python applications/get_file_from_db.py -file_name mirror_CTA-N-LST1_v2019-03-31.dat.
    output_path (str)
        Name of the local output directory where to save the files.
        Default it $CWD.
    verbosity (str, optional)
        Log level to print (default=INFO).

    Example
    -------
    getting a file from the DB.

    .. code-block:: console

        python applications/get_file_from_db.py ---file_name mirror_CTA-N-LST1_v2019-03-31.dat
"""

import logging

import simtools.configuration as configurator
import simtools.util.general as gen
from simtools import db_handler


def main():

    config = configurator.Configurator(
        label="Get file(s) from the DB.",
        description="python applications/get_file_from_db.py "
        " --file_name mirror_CTA-S-LST_v2020-04-07.dat",
    )

    config.parser.add_argument(
        "--file_name",
        help=("The name of the file name to be downloaded."),
        type=str,
    )
    args_dict, db_config = config.initialize(db_config=True)

    logger = logging.getLogger()
    logger.setLevel(gen.getLogLevelFromUser(args_dict["log_level"]))

    db = db_handler.DatabaseHandler(mongoDBConfig=db_config)
    availableDbs = [
        db.DB_TABULATED_DATA,
        db.DB_CTA_SIMULATION_MODEL,
        db.DB_CTA_SIMULATION_MODEL_DESCRIPTIONS,
        db.DB_REFERENCE_DATA,
        db.DB_DERIVED_VALUES,
        "sandbox",
        "test-data",
    ]
    fileId = None
    if args_dict["output_path"].exists():
        for dbName in availableDbs:
            try:
                fileId = db.exportFileDB(dbName, args_dict["output_path"], args_dict["file_name"])
                logger.info(
                    "Got file {} from DB {} and saved into {}".format(
                        args_dict["file_name"], dbName, args_dict["output_path"]
                    )
                )
                break
            except FileNotFoundError:
                continue

        if fileId is None:
            logger.error(
                "The file {} was not found in any of the available DBs.".format(
                    args_dict["file_name"]
                )
            )
            raise FileNotFoundError
    else:
        logger.error("Aborted, directory {} does not exist".format(args_dict["output_path"]))


if __name__ == "__main__":
    main()
