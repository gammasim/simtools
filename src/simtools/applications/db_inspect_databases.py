#!/usr/bin/python3

"""
Inspect databases and print (available database names and collections).

Command line arguments
----------------------
db_name (str, optional)
    Database name (use "all" for all databases)
"""

import logging

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.db import db_handler


def _parse():
    config = configurator.Configurator(description="Inspect databases")
    config.parser.add_argument(
        "--db_name",
        help="Database name",
        default="all",
        required=True,
    )
    return config.initialize(db_config=True)


def main():  # noqa: D103
    args_dict, db_config = _parse()

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    db = db_handler.DatabaseHandler(mongo_db_config=db_config)
    # databases without internal databases we don't have rights to modify
    databases = [
        d for d in db.db_client.list_database_names() if d not in ("config", "admin", "local")
    ]
    requested = args_dict["db_name"]
    if requested != "all" and requested not in databases:
        raise ValueError(
            f"Requested database '{requested}' not found. "
            f"Following databases are available: {', '.join(databases)}"
        )

    databases = databases if requested == "all" else [requested]

    for db_name in databases:
        print("Database:", db_name)
        collections = db.get_collections(db_name=db_name)
        print("   Collections:", collections)
        print("   Indexes:")
        for collection_name in collections:
            db_collection = db.get_collection(collection_name=collection_name, db_name=db_name)
            for idx in db_collection.list_indexes():
                print(f"     {collection_name}: {idx}")


if __name__ == "__main__":
    main()
