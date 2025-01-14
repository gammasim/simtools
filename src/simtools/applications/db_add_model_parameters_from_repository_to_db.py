#!/usr/bin/python3
r"""
    Add parameters found in a model parameter repository to a new database.

    Generates a new database with all required collections.
    Follows the structure of the CTAO gitlab model parameters repository.
    file as input.

    This is an application for experts and should not be used by the general user.

    Command line arguments

    input_path (str, required)
        Path of local copy of model parameter repository.
    db_name (str, required)
        Name of new DB to be created.
    type (str, optional)
        Type of data to be uploaded to the DB. Options are: model_parameters

    Examples
    --------
    Upload model data repository to the DB:

    .. code-block:: console

        simtools-db-add_model-parameters-from-repository-to-db \
            --input_path /path/to/repository \
            --db_name new_db_name \
            --type model_parameters
"""

import logging
from pathlib import Path

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.db import db_handler
from simtools.utils import names


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
        help="Path to model parameter repository.",
        type=Path,
        required=True,
    )
    config.parser.add_argument(
        "--db_name",
        help="Name of the new model parameter database to be created.",
        type=str.strip,
        required=True,
    )
    config.parser.add_argument(
        "--type",
        help="Type of data to be uploaded to the database.",
        type=str,
        required=False,
        default="model_parameters",
        choices=["model_parameters"],
    )

    args_dict, db_config = config.initialize(
        output=True, require_command_line=True, db_config=True, simulation_model="version"
    )
    db_config["db_simulation_model"] = args_dict["db_name"]  # overwrite explicitly DB configuration
    return args_dict, db_config


def add_values_from_json_to_db(file, collection, db, db_name, file_prefix, logger):
    """
    Upload data from json files to db.

    Parameters
    ----------
    file : list
        Json file to be uploaded to the DB.
    collection : str
        The DB collection to which to add the file.
    db : DatabaseHandler
        Database handler object.
    db_name : str
        Name of the database to be created.
    file_prefix : str
        Path to location of all additional files to be uploaded.
    logger : logging.Logger
        Logger object.
    """
    par_dict = gen.collect_data_from_file(file_name=file)
    logger.info(
        f"Adding the following parameter to the DB: {par_dict['parameter']} "
        f"(collection {collection} in database {db_name})"
    )
    db.add_new_parameter(
        db_name=db_name,
        par_dict=par_dict,
        collection_name=collection,
        file_prefix=file_prefix,
    )


def _add_model_parameters_to_db(args_dict, db, logger):
    """
    Add model parameters to the DB.

    Parameters
    ----------
    args_dict : dict
        Command line arguments.
    db : DatabaseHandler
        Database handler object.
    logger : logging.Logger
        Logger object.

    """
    input_path = Path(args_dict["input_path"])
    array_elements = [d for d in input_path.iterdir() if d.is_dir()]
    for element in array_elements:
        try:
            collection = names.get_collection_name_from_array_element_name(element.name)
        except ValueError:
            if element.name.startswith("OBS"):
                collection = "sites"
            elif element.name in {"configuration_sim_telarray", "configuration_corsika"}:
                collection = element.name
            elif element.name == "Files":
                logger.info("Files are uploaded with the corresponding model parameters")
                continue
        logger.info(f"Reading model parameters for {element.name} into collection {collection}")
        files_to_insert = list(Path(element).rglob("*json"))
        for file in files_to_insert:
            if collection == "files":
                logger.info("Not yet implemented files")
            else:
                add_values_from_json_to_db(
                    file=file,
                    collection=collection,
                    db=db,
                    db_name=args_dict["db_name"],
                    file_prefix=input_path / "Files",
                    logger=logger,
                )


def main():
    """Application main."""
    label = Path(__file__).stem
    args_dict, db_config = _parse(
        label, description="Add or update a model parameter database to the DB"
    )
    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    db = db_handler.DatabaseHandler(mongo_db_config=db_config)

    _add_model_parameters_to_db(args_dict, db, logger)


if __name__ == "__main__":
    main()
