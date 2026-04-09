#!/usr/bin/python3

"""
Inspect databases and print (available database names and collections).

Command line arguments
----------------------
db_name (str)
    Database name (use "all" for all databases)
"""

from simtools.application_control import build_application
from simtools.db import db_handler


def _add_arguments(parser):
    """Register application-specific command line arguments."""
    parser.add_argument(
        "--db_name",
        help="Database name",
        default="all",
        required=True,
    )


def main():
    """See CLI description."""
    app_context = build_application(
        initialization_kwargs={"db_config": True},
        startup_kwargs={"setup_io_handler": False},
    )

    db = db_handler.DatabaseHandler()
    # databases without internal databases we don't have rights to modify
    databases = [
        d
        for d in db.mongo_db_handler.db_client.list_database_names()
        if d not in ("config", "admin", "local")
    ]
    requested = app_context.args["db_name"]
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
