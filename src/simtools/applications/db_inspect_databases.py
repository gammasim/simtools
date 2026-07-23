#!/usr/bin/python3

"""
Inspect databases and print (available database names and collections).

Command line arguments
----------------------
db_name (str)
    Database name (use "all" for all databases)
"""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.db import db_handler

_ARGUMENTS = (
    cli.ArgumentDefinition("db_name", help="Database name", default="all", required=True),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        *cli.PATH_ARGUMENTS,
    ),
    database=True,
    setup_io_handler=False,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

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
