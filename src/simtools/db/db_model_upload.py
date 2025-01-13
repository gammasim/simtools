"""Module for uploading a simulation model (parameters and production tables) to the database."""

import logging
from pathlib import Path

import simtools.utils.general as gen
from simtools.utils import names

logger = logging.getLogger(__name__)


def add_values_from_json_to_db(file, collection, db, db_name, file_prefix):
    """
    Upload new model parameter from json files to db.

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
    """
    par_dict = gen.collect_data_from_file(file_name=file)
    logger.info(
        f"Adding the following parameter to the DB: {par_dict['parameter']} "
        f"(collection {collection} in database {db_name})"
    )
    db.add_new_parameter(
        db_name=db_name,
        array_element_name=par_dict["instrument"],
        parameter=par_dict["parameter"],
        parameter_version=par_dict["parameter_version"],
        value=par_dict["value"],
        site=par_dict["site"],
        type=par_dict["type"],
        collection_name=collection,
        applicable=par_dict["applicable"],
        file=par_dict["file"],
        unit=par_dict.get("unit", None),
        file_prefix=file_prefix,
    )


def add_model_parameters_to_db(args_dict, db):
    """
    Read model parameters from a directory and upload them to the database.

    Parameters
    ----------
    args_dict : dict
        Command line arguments.
    db : DatabaseHandler
        Database handler object.
    """
    input_path = Path(args_dict["input_path"])
    logger.info(f"Reading model parameters from repository path {input_path}")
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
                logger.info("Files (tables) are uploaded with the corresponding model parameters")
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
                )


def add_production_tables_to_db(args_dict, db):
    """
    Read production tables from a directory and upload them to the database.

    A single dictionary is prepared for each model version, containing the complete
    table of all array elements, sites, and parameters.

    Parameters
    ----------
    args_dict : dict
        Command line arguments.
    db : DatabaseHandler
        Database handler object.
    """
    input_path = Path(args_dict["input_path"])
    logger.info(f"Reading production tables from repository path {input_path}")
    model_versions = [d for d in input_path.iterdir() if d.is_dir()]

    for model in model_versions:
        logger.info(f"Reading production tables for model version {model.name}")
        files_to_insert = sorted(Path(model).rglob("*json"))
        model_dict = {
            "model_version": None,
            "parameters": {},
        }
        for file in files_to_insert:
            array_element = file.stem
            parameter_dict = gen.collect_data_from_file(file_name=file)
            logger.info(f"Reading production table for {array_element}")
            try:
                model_dict["model_version"] = parameter_dict["model_version"]
                model_dict["parameters"][array_element] = parameter_dict["parameters"][
                    array_element
                ]
            except KeyError as exc:
                logger.error(f"KeyError: {exc}")
                raise

        db.add_production_table(
            db_name=args_dict["db_name"],
            production_table=model_dict,
        )
