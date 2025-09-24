"""Upload a simulation model (parameters and production tables) to the database."""

import logging
from pathlib import Path

from packaging.version import Version

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
    for element in filter(Path.is_dir, input_path.iterdir()):
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

    for model in sorted(filter(Path.is_dir, input_path.iterdir())):
        logger.info(f"Reading production tables for model version {model.name}")
        model_dict = _read_production_tables(model)

        for collection, data in model_dict.items():
            if data["parameters"]:
                logger.info(
                    f"Adding production table for {collection} "
                    f"(model version {model.name}) to the database"
                )
                db.add_production_table(production_table=data)
            else:
                logger.info(f"No production table for {collection} in model version {model.name}")


def _read_production_tables(model_path):
    """
    Read production tables from a directory.

    Take into account that some productions include patch updates only. Read in this cases
    all models from the model version history, starting with the earliest one.

    Parameters
    ----------
    model_path : Path
        Path to the directory containing the production tables for a specific model version.
    """
    model_dict = {}
    models = [model_path.name]
    if (model_path / "info.yml").exists():
        info = ascii_handler.collect_data_from_file(file_name=model_path / "info.yml")
        models.extend(info.get("model_version_history", []))
    # sort oldest --> newest
    models = sorted(set(models), key=Version, reverse=False)
    for model in models:
        for file in sorted((model_path.parent / model).rglob("*json")):
            _read_production_table(model_dict, file, model)

    # ensure that the for patch updates the model version is set correctly
    for table in model_dict.values():
        table["model_version"] = model_path.name

    _remove_deprecated_model_parameters(model_dict)

    return model_dict


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
            "deprecated_parameters": [],
        },
    )
    parameter_dict = ascii_handler.collect_data_from_file(file_name=file)
    logger.debug(
        f"Reading production table for {array_element} "
        f"(model_version {model_name}, collection {collection})"
    )
    try:
        if array_element in ("configuration_corsika", "configuration_sim_telarray"):
            model_dict[collection]["parameters"] = parameter_dict["parameters"]
        else:
            model_dict[collection]["parameters"].setdefault(array_element, {}).update(
                parameter_dict["parameters"][array_element]
            )
    except KeyError as exc:
        logger.error(f"KeyError: {exc}")
        raise

    try:
        model_dict[collection]["design_model"][array_element] = parameter_dict["design_model"][
            array_element
        ]
    except KeyError:
        pass

    try:
        model_dict[collection]["deprecated_parameters"] = parameter_dict["deprecated_parameters"]
    except KeyError:
        pass

    model_dict[collection]["model_version"] = model_name


def _remove_deprecated_model_parameters(model_dict):
    """
    Remove deprecated parameters from all tables in a model dictionary.

    Parameters
    ----------
    model_dict : dict
        Production tables for a specific model version.
    """
    for table in model_dict.values():
        for params in table.get("parameters", {}).values():
            for param in table.get("deprecated_parameters", []):
                if param in params:
                    logger.info(
                        f"Deprecated parameter {param} in production table {table['collection']}"
                    )
                    params.pop(param)
