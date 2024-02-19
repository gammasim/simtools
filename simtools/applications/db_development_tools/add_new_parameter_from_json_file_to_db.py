#!/usr/bin/python3

"""
    Summary
    -------
    This application is used to add a new parameter to the sites collection in the DB
    using a json file as input.

    This application should not be used by anyone but expert users and not often.
    Therefore, no additional documentation about this applications will be given.

"""

import logging

import simtools.utils.general as gen
from simtools.db import db_handler
from simtools.configuration import configurator


def main():
    config = configurator.Configurator(
        description=("Add a new parameter to the DB.")
    )
    config.parser.add_argument("--file_name", help="file to be added", required=True)
    config.parser.add_argument(
        "--db_collection", help="DB collection file will be added", required=True)
    args_dict, db_config = config.initialize(db_config=True)

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    db = db_handler.DatabaseHandler(mongo_db_config=db_config)

    if args_dict["db_collection"] not in ("telescopes", "sites"):
        raise ValueError(f"DB collection {args_dict['db_collection']} not recognized")

    par_dict = gen.collect_data_from_file_or_dict(
        file_name=args_dict["file_name"], in_dict=None)

    logger.info(f"Adding the following parameters to the DB: {par_dict['parameter']}")

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
        unit=par_dict["unit"] if "unit" in par_dict and par_dict["unit"] is not None else None,
        file_prefix="./",
    )


if __name__ == "__main__":
    main()
