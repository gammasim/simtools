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

from simtools.application_control import build_application
from simtools.db import db_handler


def _add_arguments(parser):
    """Register application-specific command line arguments."""
    parser.add_argument(
        "--db_name",
        help="Database name",
        default=None,
        required=False,
    )


def main():
    """Generate compound indexes for the specified database."""
    app_context = build_application(
        __file__,
        description="Generate compound indexes for a specific database",
        add_arguments_function=_add_arguments,
        initialization_kwargs={"db_config": True},
        startup_kwargs={"setup_io_handler": False},
    )

    db = db_handler.DatabaseHandler()

    db.generate_compound_indexes_for_databases(
        db_name=app_context.args["db_name"],
        db_simulation_model=app_context.args.get("db_simulation_model"),
        db_simulation_model_version=app_context.args.get("db_simulation_model_version"),
    )


if __name__ == "__main__":
    main()
