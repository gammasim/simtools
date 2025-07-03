#!/usr/bin/python3
r"""
    Add parameters and production tables from a simulation model repository to a new database.

    Generates a new database with all required collections.
    Follows the structure of the CTAO gitlab simulation model repository.

    This is an application for experts and should not be used by the general user.

    Command line arguments

    input_path (str, required)
        Path of local copy of model parameter repository.
    db_name (str, required)
        Name of new DB to be created.
    type (str, optional)
        Type of data to be uploaded to the DB. Options are: model_parameters, production_tables.

    Examples
    --------
    Upload model data repository to the DB
    Loops over all subdirectories in 'input_path' and uploads all json files to the
    database 'new_db_name' (or updates an existing database with the same name):

    * subdirectories starting with 'OBS' are uploaded to the 'sites' collection
    * json files from the subdirectory 'configuration_sim_telarray/configuration_corsika'
      are uploaded to the 'configuration_sim_telarray/configuration_corsika' collection
    * 'Files' are added to the 'files' collection
    * all other json files are uploaded to collection defined in the array element description
      in 'simtools/schemas/array_elements.yml'

    .. code-block:: console

        simtools-db-simulation-model-from-repository-to-db \
            --input_path /path/to/repository \
            --db_name new_db_name \
            --type model_parameters

    Upload production tables to the DB:

    .. code-block:: console

        simtools-db-simulation-model-from-repository-to-db \
            --input_path /path/to/repository \
            --db_name new_db_name \
            --type production_tables

"""

import logging
from pathlib import Path

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.db import db_handler, db_model_upload


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
    CommandLineParser
        Command line parser object.
    """
    config = configurator.Configurator(label=label, description=description)
    config.parser.add_argument(
        "--input_path",
        help="Path to simulation model repository.",
        type=Path,
        required=True,
    )
    config.parser.add_argument(
        "--db_name",
        help="Name of the new simulation model database to be created.",
        type=str.strip,
        required=True,
    )
    config.parser.add_argument(
        "--type",
        help="Type of data to be uploaded to the database.",
        type=str,
        required=False,
        default="model_parameters",
        choices=["model_parameters", "production_tables"],
    )

    args_dict, db_config = config.initialize(output=True, require_command_line=True, db_config=True)
    db_config["db_simulation_model"] = args_dict["db_name"]  # overwrite explicitly DB configuration
    return args_dict, db_config


def main():
    """Application main."""
    label = Path(__file__).stem
    args_dict, db_config = _parse(
        label, description="Add or update a model parameter database to the DB"
    )
    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    db = db_handler.DatabaseHandler(mongo_db_config=db_config)

    if args_dict.get("type") == "model_parameters":
        db_model_upload.add_model_parameters_to_db(args_dict, db)
    elif args_dict.get("type") == "production_tables":
        db_model_upload.add_production_tables_to_db(args_dict, db)


if __name__ == "__main__":
    main()
