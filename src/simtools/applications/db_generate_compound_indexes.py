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
    app_context = startup_application(_parse, setup_io_handler=False)

    db = db_handler.DatabaseHandler()

    db.generate_compound_indexes_for_databases(
        db_name=app_context.args["db_name"],
        db_simulation_model=app_context.args.get("db_simulation_model"),
        db_simulation_model_version=app_context.args.get("db_simulation_model_version"),
    )


if __name__ == "__main__":
    main()
