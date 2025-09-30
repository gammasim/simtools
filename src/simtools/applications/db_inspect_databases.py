#!/usr/bin/python3

"""
Inspect databases and print (available database names and collections).

Command line arguments
----------------------
db_name (str, optional)
    Database name (use "all" for all databases)
"""

from simtools.application_control import get_application_label, startup_application
from simtools.configuration import configurator
from simtools.db import db_handler


def _parse():
    """Parse command line configuration."""
    config = configurator.Configurator(
        label=get_application_label(__file__), description="Inspect databases"
    )
    config.parser.add_argument(
        "--db_name",
        help="Database name",
        default="all",
        required=True,
    )
    return config.initialize(db_config=True)


def main():
    """Inspect databases."""
    args_dict, db_config, _, _ = startup_application(_parse, setup_io_handler=False)

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
