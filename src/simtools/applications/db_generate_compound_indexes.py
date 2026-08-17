#!/usr/bin/python3

"""Generate compound indexes for the specified database."""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.db import db_handler

_ARGUMENTS = (cli.DATABASE_NAME,)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(*_ARGUMENTS,),
    database=True,
    setup_io_handler=False,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

    db = db_handler.DatabaseHandler()

    db.generate_compound_indexes_for_databases(
        db_name=app_context.args["database_name"],
        db_simulation_model=app_context.args.get("db_simulation_model"),
        db_simulation_model_version=app_context.args.get("db_simulation_model_version"),
    )


if __name__ == "__main__":
    main()
