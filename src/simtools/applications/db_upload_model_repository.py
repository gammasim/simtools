#!/usr/bin/python3
"""Upload a simulation-model repository to MongoDB."""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.constants import DEFAULT_SIMULATION_MODELS
from simtools.db import db_handler, db_model_upload
from simtools.settings import config

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "branch",
        help="Repository branch to clone (optional, defaults to using version tag).",
        type=str,
        required=False,
    ),
    cli.ArgumentDefinition(
        "tmp_dir",
        help="Temporary directory for cloning the repository (default: ./tmp_model_parameters).",
        type=str,
        default="tmp_model_parameters",
        required=False,
    ),
    cli.ArgumentDefinition(
        "max_attempts",
        help="Maximum number of attempts to clone the repository (default: 3).",
        type=int,
        default=3,
        required=False,
    ),
    cli.ArgumentDefinition(
        "repository_dir",
        help="Path to existing simulation model repository directory (optional).",
        type=str,
        required=False,
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        *cli.OUTPUT_PATH_ARGUMENTS,
        *cli.OUTPUT_ARGUMENTS,
    ),
    database=True,
    use_dependency_defaults=False,
    initialize_output=True,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

    if app_context.args.get("db_simulation_model_tag"):
        app_context.db_config["db_simulation_model"] = app_context.args.get(
            "db_simulation_model", "CTAO-Simulation-Model"
        )
        app_context.db_config["db_simulation_model_tag"] = app_context.args[
            "db_simulation_model_tag"
        ]
        config.load(app_context.args, app_context.db_config)
    else:
        raise ValueError("Setting of db_simulation_model_tag is required.")

    db = db_handler.DatabaseHandler()
    db.require_mongodb("Uploading a simulation model to a database")
    db.print_connection_info()

    db_model_upload.add_complete_model(
        tmp_dir=app_context.args.get("tmp_dir"),
        db=db,
        db_simulation_model=app_context.args.get("db_simulation_model"),
        db_simulation_model_version=app_context.args.get("db_simulation_model_tag"),
        repository_url=(
            None if app_context.args.get("repository_dir") else DEFAULT_SIMULATION_MODELS
        ),
        repository_branch=app_context.args.get("branch"),
        max_attempts=app_context.args.get("max_attempts"),
        repository_dir=app_context.args.get("repository_dir"),
    )


if __name__ == "__main__":
    main()
