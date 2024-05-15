#!/usr/bin/python3
"""
    This application adds all parameters found in a repository to the DB.

    Generates a new data with all required collections.
    Follows the structure of the CTAO gitlab model parameters repository.
    file as input.

    This is an application for experts and should not be used by the general user.

    Command line arguments

    input_path (str, required)
        Path of local copy of model parameter repository.
    db_name (str, required)
        Name of new DB to be created.

Examples
--------
    Upload a repository to the DB:

    .. code-block:: console

        simtools-add_model-parameters-from-repository-to-db \
            --input_path /path/to/repository \
            --db_name new_db_name
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
        help="Name of the new DB to be created.",
        type=str,
        required=True,
    )

    return config.initialize(
        output=True, require_command_line=True, db_config=True, simulation_model="version"
    )


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
    par_dict = gen.collect_data_from_file_or_dict(file_name=file, in_dict=None)
    logger.info(f"Adding the following parameter to the DB: {par_dict['parameter']}")
    db.add_new_parameter(
        db_name=db_name,
        telescope=par_dict["instrument"],
        parameter=par_dict["parameter"],
        version=par_dict["version"],
        value=par_dict["value"],
        site=par_dict["site"],
        type=par_dict["type"],
        collection_name=collection,
        applicable=par_dict["applicable"],
        file=par_dict["file"],
        unit=par_dict.get("unit", None),
        file_prefix=file_prefix,
    )


def _add_metadata_to_db(db, db_name, logger):
    """
    Add metadata to the DB.

    Parameters
    ----------
    db : DatabaseHandler
        Database handler object.
    db_name : str
        Name of the database to be created.
    logger : logging.Logger
        Logger object.
    """
    logger.info("Adding metadata to the DB")
    db.add_tagged_version(
        db_name=db_name,
        released_version="2020-06-28",
        released_label="Prod5",
        latest_version="2020-06-28",
        latest_label="Prod5",
    )


def main():
    """Application main."""
    label = Path(__file__).stem
    args_dict, db_config = _parse(label, description="Add a new model parameter database to the DB")
    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    db = db_handler.DatabaseHandler(mongo_db_config=db_config)

    input_path = Path(args_dict["input_path"])
    array_elements = [d for d in input_path.iterdir() if d.is_dir()]
    for element in array_elements:
        try:
            collection = names.get_collection_name_from_array_element_name(element.name)
        except ValueError:
            if element.name.startswith("OBS"):
                collection = "sites"
            elif element.name == "configuration_sim_telarray":
                collection = "configuration_sim_telarray"
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

    _add_metadata_to_db(db, db_name=args_dict["db_name"], logger=logger)


if __name__ == "__main__":
    main()
