"""Upload a simulation model (parameters and production tables) to the database."""

import logging
import shutil
from pathlib import Path

from packaging.version import Version

from simtools.io import ascii_handler
from simtools.job_execution.job_manager import retry_command
from simtools.utils import names

logger = logging.getLogger(__name__)


def add_complete_model(
    tmp_dir,
    db,
    db_simulation_model,
    db_simulation_model_version,
    repository_url,
    repository_branch=None,
):
    """
    Upload a complete model including model parameters and production tables to the database.

    Generate compound indexes for the new database.

    Parameters
    ----------
    tmp_dir : Path or str
        Temporary directory to use for cloning the repository.
    db : DatabaseHandler
        Database handler object.
    db_simulation_model : str
        Name of the simulation model in the database.
    db_simulation_model_version : str
        Version of the simulation model in the database.
    repository_url : str
        URL of the simulation model repository to clone.
    repository_branch : str, optional
        Branch of the repository to use. If None, the default branch is used.

    Returns
    -------
    None
        This function does not return a value.
    """
    if not _confirm_remote_database_upload(db):
        return

    repository_dir = None
    try:
        repository_dir = clone_simulation_model_repository(
            tmp_dir,
            repository_url,
            db_simulation_model_version=db_simulation_model_version,
            repository_branch=repository_branch,
        )

        add_model_parameters_to_db(
            input_path=repository_dir / "simulation-models" / "model_parameters", db=db
        )

        add_production_tables_to_db(
            input_path=repository_dir / "simulation-models" / "productions", db=db
        )

        db.generate_compound_indexes_for_databases(
            db_name=None,
            db_simulation_model=db_simulation_model,
            db_simulation_model_version=db_simulation_model_version,
        )
    except Exception as exc:
        raise RuntimeError(f"Upload of simulation model failed: {exc}") from exc
    finally:
        if repository_dir is not None and repository_dir.exists():
            shutil.rmtree(repository_dir)

    logger.info("Upload of simulation model completed successfully")


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
        if info.get("model_update") == "patch_update":
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


def _confirm_remote_database_upload(db):
    """
    Confirm with the user that they want to upload to a remote database.

    Returns True if the user confirms, False otherwise. Returns also True
    if the database is not remote.

    Parameters
    ----------
    db : DatabaseHandler
        Database handler object.

    Returns
    -------
    bool
        True if user confirms upload, False otherwise.
    """
    abort_message = "Operation aborted."

    if not db.is_remote_database():
        return True

    db_config = db.db_config
    db_server = db_config.get("db_server", "unknown server") if db_config else "unknown server"

    try:
        # First confirmation
        user_input = input(
            f"Do you really want to upload to remote DB {db_server}? Type 'yes' to confirm: "
        )
        if user_input != "yes":
            logger.info(abort_message)
            return False

        # Second confirmation
        user_input = input(
            f"Let be sure: do you really want to upload to remote DB {db_server}? "
            "Type 'yes' to confirm: "
        )
        if user_input != "yes":
            logger.info(abort_message)
            return False

        return True

    except (EOFError, KeyboardInterrupt):
        logger.info(abort_message)
        return False


def clone_simulation_model_repository(
    target_dir, repository_url, db_simulation_model_version, repository_branch
):
    """
    Clone a git repository containing simulation model parameters and production tables.

    Parameters
    ----------
    target_dir : Path or str
        Target directory to clone the repository into.
    repository_url : str
        URL of the git repository to clone.
    db_simulation_model_version : str
        Version tag of the simulation model to checkout.
    repository_branch : str, optional
        Branch of the repository to use. If None, the version tag is used.

    Returns
    -------
    Path
        Path to the cloned repository.
    """
    if repository_branch:
        logger.info(f"Using branch: {repository_branch}")
    else:
        logger.info(f"Using version tag: {db_simulation_model_version}")

    target_dir = Path(target_dir)
    target_dir = target_dir if target_dir.is_absolute() else Path.cwd() / target_dir

    if target_dir.exists():
        shutil.rmtree(target_dir)

    logger.info(f"Cloning model parameters from {repository_url}")

    if repository_branch:
        command = f'git clone --depth=1 -b "{repository_branch}" "{repository_url}" "{target_dir}"'
    else:
        # Use version tag with detached head (generates warning but that's fine)
        command = (
            f'git clone --branch "{db_simulation_model_version}" '
            f'--depth 1 "{repository_url}" "{target_dir}"'
        )

    retry_command(command)

    return target_dir
