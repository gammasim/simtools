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
    mongo_db_handler = db.require_mongodb("Inspecting databases")
    # databases without internal databases we don't have rights to modify
    databases = mongo_db_handler.get_accessible_database_names()
    requested = app_context.args["db_name"]
    databases = mongo_db_handler.resolve_requested_databases(requested, databases)

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
