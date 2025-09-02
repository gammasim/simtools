"""Upload a simulation model (parameters and production tables) to the database."""

import logging
from pathlib import Path

from simtools.io import ascii_handler
from simtools.utils import names

logger = logging.getLogger(__name__)


def add_values_from_json_to_db(file, collection, db, file_prefix):
    """
    Upload new model parameter from json files to db.

    Parameters
    ----------
    file : list
        JSON file to be uploaded to the DB.
    collection : str
        The DB collection to which to add the file.
    db : DatabaseHandler
        Database handler object.
    file_prefix : str
        Path to location of all additional files to be uploaded.
    """
    par_dict = ascii_handler.collect_data_from_file(file_name=file)
    logger.debug(
        f"Adding the following parameter to the DB: {par_dict['parameter']} "
        f"version {par_dict['parameter_version']} "
        f"(collection {collection} in database {db.get_db_name()})"
    )

    db.add_new_parameter(
        par_dict=par_dict,
        collection_name=collection,
        file_prefix=file_prefix,
    )


def add_model_parameters_to_db(input_path, db):
    """
    Read model parameters from a directory and upload them to the database.

    Parameters
    ----------
    input_path : Path, str
        Path to the directory containing the model parameters.
    db : DatabaseHandler
        Database handler object.
    """
    input_path = Path(input_path)
    logger.info(f"Reading model parameters from repository path {input_path}")
    array_elements = [d for d in input_path.iterdir() if d.is_dir()]
    for element in array_elements:
        collection = names.get_collection_name_from_array_element_name(element.name, False)
        if collection == "Files":
            logger.info("Files (tables) are uploaded with the corresponding model parameters")
            continue
        logger.info(f"Reading model parameters for {element.name} into collection {collection}")
        files_to_insert = list(Path(element).rglob("*json"))
        for file in files_to_insert:
            add_values_from_json_to_db(
                file=file,
                collection=collection,
                db=db,
                file_prefix=input_path / "Files",
            )


def add_production_tables_to_db(input_path, db):
    """
    Read production tables from a directory and upload them to the database.

    One dictionary per collection is prepared for each model version, containing
    tables of all array elements, sites, and configuration parameters.

    Parameters
    ----------
    input_path : Path, str
        Path to the directory containing the production tables.
    db : DatabaseHandler
        Database handler object.
    """
    input_path = Path(input_path)
    logger.info(f"Reading production tables from repository path {input_path}")

    for model in filter(Path.is_dir, input_path.iterdir()):
        logger.info(f"Reading production tables for model version {model.name}")
        model_dict = {}
        for file in sorted(model.rglob("*json")):
            _read_production_table(model_dict, file, model.name)

        for collection, data in model_dict.items():
            if not data["parameters"]:
                logger.info(f"No production table for {collection} in model version {model.name}")
                continue
            logger.info(f"Adding production table for {collection} to the database")
            db.add_production_table(
                production_table=data,
            )


def _read_production_table(model_dict, file, model_name):
    """Read a single production table from file."""
    array_element = file.stem
    collection = names.get_collection_name_from_array_element_name(array_element, False)
    model_dict.setdefault(
        collection,
        {
            "collection": collection,
            "model_version": model_name,
            "parameters": {},
            "design_model": {},
        },
    )
    parameter_dict = ascii_handler.collect_data_from_file(file_name=file)
    logger.debug(f"Reading production table for {array_element} (collection {collection})")
    try:
        if array_element in ("configuration_corsika", "configuration_sim_telarray"):
            model_dict[collection]["parameters"] = parameter_dict["parameters"]
        else:
            model_dict[collection]["parameters"][array_element] = parameter_dict["parameters"][
                array_element
            ]
    except KeyError as exc:
        logger.error(f"KeyError: {exc}")
        raise
    try:
        model_dict[collection]["design_model"][array_element] = parameter_dict["design_model"][
            array_element
        ]
    except KeyError:
        pass
