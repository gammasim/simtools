#!/usr/bin/python3

"""
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

    Example
    -------
    getting a file from the DB.

    .. code-block:: console

        simtools-db-get-file-from-db --file_name mirror_CTA-N-LST1_v2019-03-31.dat

    Expected final print-out message:

    .. code-block:: console

        INFO::db_get_file_from_db(l82)::main::Got file mirror_CTA-N-LST1_v2019-03-31.dat from DB \
        CTA-Simulation-Model and saved into .

"""

import logging

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.db import db_handler
from simtools.io import io_handler


def _parse():
    config = configurator.Configurator(
        description="Get file(s) from the DB.",
        usage="simtools-get-file-from-db --file_name mirror_CTA-S-LST_v2020-04-07.dat",
    )

    config.parser.add_argument(
        "--file_name",
        help="The name of the file(s) to be downloaded (single file or space-separated list).",
        type=str,
        nargs="+",
        required=True,
    )
    return config.initialize(db_config=True, output=True)


def main():  # noqa: D103
    args_dict, db_config = _parse()

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))
    _io_handler = io_handler.IOHandler()

    db = db_handler.DatabaseHandler(mongo_db_config=db_config)
    file_id = db.export_model_files(
        dest=_io_handler.get_output_directory(),
        file_names=args_dict["file_name"],
    )
    if file_id is None:
        logger.error(f"The file {args_dict['file_name']} was not found in {db.db_name}.")
        raise FileNotFoundError
    logger.info(
        f"Got file {args_dict['file_name']} from DB {db.db_name} "
        f"and saved into {_io_handler.get_output_directory()}"
    )


if __name__ == "__main__":
    main()
