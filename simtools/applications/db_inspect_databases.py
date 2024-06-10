#!/usr/bin/python3

"""
Inspect databases and print (available database names and collections).

Command line arguments
----------------------
db_name (str, optional)
    Inspect a specific database.
"""

import logging

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.db import db_handler


def main():
    config = configurator.Configurator(description="Inspect databases")
    config.parser.add_argument(
        "--db_name",
        help="Inspect a specific database (use all to print all databases)",
        default="all",
        required=True,
    )
    args_dict, db_config = config.initialize(db_config=True, simulation_model="telescope")

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    db = db_handler.DatabaseHandler(mongo_db_config=db_config)

    databases = db.db_client.list_database_names()

    for db_name in databases:
        if args_dict["db_name"] != "all" and db_name != args_dict["db_name"]:
            continue
        # missing admin rights; skip config and admin
        if db_name in ("config", "admin", "local"):
            continue
        print("Database:", db_name)
        collections = db.get_collections(db_name=db_name)
        print("   Collections:", collections)


if __name__ == "__main__":
    main()
