#!/usr/bin/python3

"""
    Summary
    -------
    This application adds a new parameter / value to a collection in the DB using a json
    file as input.

    Command line arguments
    ----------------------
    file_name (str, required)
        Name of the file to upload including the full path.
    db_collection (str, required)
        The DB collection to which to add the file.

    Example
    -------

    Upload a file to sites collection:

    .. code-block:: console

        simtools-add-value-from-json-to-db \
            --file_name new_value.json --db_collection sites


"""

import logging

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.db import db_handler


def main():
    config = configurator.Configurator(description="Add a new parameter to the DB.")
    config.parser.add_argument("--file_name", help="file to be added", required=True)
    config.parser.add_argument(
        "--db_collection", help="DB collection to which to add the file ", required=True
    )
    args_dict, db_config = config.initialize(db_config=True)

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    db = db_handler.DatabaseHandler(mongo_db_config=db_config)

    par_dict = gen.collect_data_from_file_or_dict(file_name=args_dict["file_name"], in_dict=None)

    logger.info(f"Adding the following parameter to the DB: {par_dict['parameter']}")

    print(f"Should {par_dict} be inserted to the {args_dict['db_collection']} collection?:\n")
    if gen.user_confirm():
        db.add_new_parameter(
            db_name=db.DB_CTA_SIMULATION_MODEL,
            telescope=par_dict["instrument"],
            parameter=par_dict["parameter"],
            version=par_dict["version"],
            value=par_dict["value"],
            site=par_dict["site"],
            type=par_dict["type"],
            collection_name=args_dict["db_collection"],
            applicable=par_dict["applicable"],
            file=par_dict["file"],
            unit=par_dict.get("unit", None),
            file_prefix="./",
        )
        logger.info(
            f"Value for {par_dict['parameter']} added to {args_dict['db_collection']} collection"
        )
    else:
        logger.info("Aborted, no change applied to the database")


if __name__ == "__main__":
    main()
