#!/usr/bin/python3

"""
Generate compound indexes for the specified database.

This needs to be done once after a database has been set up.
Significantly accelerates database querying (at least a factor
of 5 in query time with a factor of 10 less documents examined).

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
        description="Generate compound indexes for a specific database",
        label=get_application_label(__file__),
    )
    config.parser.add_argument(
        "--db_name",
        help="Database name",
        default=None,
        required=False,
    )
    return config.initialize(db_config=True)


def main():
    """Generate compound indexes for the specified database."""
    args_dict, db_config, logger, _ = startup_application(_parse, setup_io_handler=False)

    db = db_handler.DatabaseHandler(mongo_db_config=db_config)
    # databases without internal databases we don't have rights to modify
    databases = [
        d for d in db.db_client.list_database_names() if d not in ("config", "admin", "local")
    ]
    logger.debug(f"Available databases: {databases}")
    requested = db.get_db_name(
        db_name=args_dict["db_name"],
        db_simulation_model_version=args_dict.get("db_simulation_model_version"),
        model_name=args_dict.get("db_simulation_model"),
    )
    if requested != "all" and requested not in databases:
        raise ValueError(
            f"Requested database '{requested}' not found. "
            f"Following databases are available: {', '.join(databases)}"
        )

    databases = databases if requested == "all" else [requested]
    for db_name in databases:
        logger.info(f"Generating compound indexes for database: {db_name}")
        db.generate_compound_indexes(db_name=db_name)


if __name__ == "__main__":
    main()
