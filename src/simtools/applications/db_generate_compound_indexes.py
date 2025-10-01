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

import logging

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.db import db_handler


def _parse():
    config = configurator.Configurator(
        description="Generate compound indexes for a specific database"
    )
    config.parser.add_argument(
        "--db_name",
        help="Database name",
        default=None,
        required=False,
    )
    return config.initialize(db_config=True)


def main():  # noqa: D103
    args_dict, db_config = _parse()

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    db = db_handler.DatabaseHandler(mongo_db_config=db_config)

    db.generate_compound_indexes_for_databases(
        db_name=args_dict["db_name"],
        db_simulation_model=args_dict.get("db_simulation_model"),
        db_simulation_model_version=args_dict.get("db_simulation_model_version"),
    )


if __name__ == "__main__":
    main()
