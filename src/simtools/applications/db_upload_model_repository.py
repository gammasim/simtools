#!/usr/bin/python3
r"""
Upload model parameters from simulation model repository to a local or remote database.

This script clones the CTAO simulation model repository and uploads model parameters
and production tables to a MongoDB database. It includes retry functionality for
network operations and confirmation prompts for remote database uploads.

Command line arguments
----------------------
db_simulation_model (str, required)
    Name of the database simulation model.
db_simulation_model_version (str, required)
    Version of the database simulation model.
branch (str, optional)
    Repository branch to clone (if not provided, uses the version tag).
tmp_dir (str, optional)
    Temporary directory for cloning the repository (default: ./tmp_model_parameters).

Examples
--------
Upload model repository to database using version tag:

.. code-block:: console

    simtools-db-upload-model-repository \\
        --db_simulation_model CTAO-Simulation-Model \\
        --db_simulation_model_version 6.2.0

Upload model repository using specific branch:

.. code-block:: console

    simtools-db-upload-model-repository \\
        --db_simulation_model CTAO-Simulation-Model \\
        --db_simulation_model_version 6.2.0 \\
        --branch main

"""

import logging
from pathlib import Path

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.db import db_handler, db_model_upload

DEFAULT_REPOSITORY_URL = (
    "https://gitlab.cta-observatory.org/cta-science/simulations/"
    "simulation-model/simulation-models.git"
)


def _parse(label=None, description=None):
    """
    Parse command line configuration.

    Parameters
    ----------
    label : str
        Label describing application.
    description : str
        Description of application.

    Returns
    -------
    tuple
        Command line parser object and database configuration.
    """
    config = configurator.Configurator(label=label, description=description)
    config.parser.add_argument(
        "--branch",
        help="Repository branch to clone (optional, defaults to using version tag).",
        type=str,
        required=False,
    )
    config.parser.add_argument(
        "--tmp_dir",
        help="Temporary directory for cloning the repository (default: ./tmp_model_parameters).",
        type=str,
        default="tmp_model_parameters",
        required=False,
    )

    args_dict, db_config = config.initialize(output=True, require_command_line=True, db_config=True)

    if args_dict.get("db_simulation_model_version"):
        db_config["db_simulation_model"] = args_dict.get(
            "db_simulation_model", "CTAO-Simulation-Model"
        )
        db_config["db_simulation_model_version"] = args_dict["db_simulation_model_version"]
    else:
        raise ValueError("Setting of db_simulation_model_version is required.")

    return args_dict, db_config


def main():
    """Application main."""
    args_dict, db_config = _parse(
        label=Path(__file__).stem, description="Upload model parameters from repository to database"
    )
    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    db = db_handler.DatabaseHandler(mongo_db_config=db_config)
    db.print_connection_info()

    db_model_upload.add_complete_model(
        tmp_dir=args_dict.get("tmp_dir"),
        db=db,
        db_simulation_model=args_dict.get("db_simulation_model"),
        db_simulation_model_version=args_dict.get("db_simulation_model_version"),
        repository_url=DEFAULT_REPOSITORY_URL,
        repository_branch=args_dict.get("branch"),
    )


if __name__ == "__main__":
    main()
