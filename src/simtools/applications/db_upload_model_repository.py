#!/usr/bin/python3
r"""
Upload model parameters from simulation model repository to a local or remote MongoDB.

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

Examples
--------
Upload model repository to database using version tag:

.. code-block:: console

    simtools-db-upload-model-repository \\
        --db_simulation_model ctao-prod6 \\
        --db_simulation_model_version 6.2.0

Upload model repository using specific branch:

.. code-block:: console

    simtools-db-upload-model-repository \\
        --db_simulation_model ctao-prod6 \\
        --db_simulation_model_version 6.2.0 \\
        --branch main

"""

import logging
import re
import shutil
import subprocess
import time
from pathlib import Path

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.db import db_handler, db_model_upload

DEFAULT_REPOSITORY_URL = (
    "https://gitlab.cta-observatory.org/cta-science/simulations/"
    "simulation-model/simulation-models.git"
)


def retry_command(command, max_attempts=3, delay=10):
    """
    Execute a shell command with retry logic for network-related failures.

    Parameters
    ----------
    command : str
        Shell command to execute.
    max_attempts : int
        Maximum number of retry attempts (default: 3).
    delay : int
        Delay in seconds between attempts (default: 10).

    Returns
    -------
    bool
        True if command succeeded, False if all attempts failed.

    Raises
    ------
    subprocess.CalledProcessError
        If command fails after all retry attempts.
    """
    logger = logging.getLogger()

    for attempt in range(1, max_attempts + 1):
        logger.info(f"Attempt {attempt} of {max_attempts}: {command}")
        try:
            subprocess.run(command, shell=True, check=True, text=True)
            logger.info(f"Command succeeded on attempt {attempt}")
            return True
        except subprocess.CalledProcessError as exc:
            logger.warning(f"Command failed on attempt {attempt}")
            if attempt < max_attempts:
                logger.info(f"Waiting {delay}s before retry...")
                time.sleep(delay)
            else:
                logger.error(f"Command failed after {max_attempts} attempts")
                raise exc from None
    return False


def clone_repository(repository_url, target_dir, version=None, branch=None):
    """
    Clone a Git repository with retry logic.

    Parameters
    ----------
    repository_url : str
        URL of the Git repository to clone.
    target_dir : Path
        Target directory for the clone.
    version : str, optional
        Version tag to checkout (used if branch is not provided).
    branch : str, optional
        Branch name to clone (takes precedence over version).

    Returns
    -------
    bool
        True if clone succeeded, False otherwise.
    """
    logger = logging.getLogger()

    # Remove target directory if it exists
    if target_dir.exists():
        shutil.rmtree(target_dir)

    logger.info(f"Cloning model parameters from {repository_url}")

    if branch:
        command = f'git clone --depth=1 -b "{branch}" "{repository_url}" "{target_dir}"'
    else:
        # Use version tag with detached head (generates warning but that's fine)
        command = f'git clone --branch "{version}" --depth 1 "{repository_url}" "{target_dir}"'

    return retry_command(command)


def confirm_remote_database_upload(db_server):
    """
    Ask for double confirmation before uploading to a remote database.

    Parameters
    ----------
    db_server : str
        Database server name/URL.

    Returns
    -------
    bool
        True if user confirms upload, False otherwise.
    """
    logger = logging.getLogger()
    abort_message = "Operation aborted."

    # Check if this looks like a remote server (has domain pattern)
    domain_pattern = r"^([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}$"
    if not re.match(domain_pattern, db_server):
        # Local database, no confirmation needed
        return True

    logger.info(f"DB_SERVER: {db_server}")

    # First confirmation
    try:
        user_input = input(
            f"Do you really want to upload to remote DB {db_server}? Type 'yes' to confirm: "
        )
        if user_input != "yes":
            logger.info(abort_message)
            return False

        # Second confirmation
        user_input = input(
            f"Let me ask again: do you really want to upload to remote DB {db_server}? "
            "Type 'yes' to confirm: "
        )
        if user_input != "yes":
            logger.info(abort_message)
            return False

        return True

    except (EOFError, KeyboardInterrupt):
        logger.info(abort_message)
        return False


def copy_env_file(source_dir, target_dir):
    """
    Copy .env file from source directory to target directory if it exists.

    Parameters
    ----------
    source_dir : Path
        Source directory (parent of database_scripts).
    target_dir : Path
        Target directory (cloned repository).
    """
    logger = logging.getLogger()

    env_file = source_dir / ".env"
    if env_file.exists():
        target_env = target_dir / ".env"
        shutil.copy2(env_file, target_env)
        logger.info(f"Copied .env file to {target_env}")


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

    args_dict, db_config = config.initialize(output=True, require_command_line=True, db_config=True)

    if args_dict.get("db_simulation_model") and args_dict.get("db_simulation_model_version"):
        # overwrite explicitly DB configuration
        db_config["db_simulation_model"] = args_dict["db_simulation_model"]
        db_config["db_simulation_model_version"] = args_dict["db_simulation_model_version"]
    else:
        raise ValueError("Both db_simulation_model and db_simulation_model_version are required.")

    return args_dict, db_config


def upload_model_data_to_database(
    repository_dir, db_config, db_simulation_model, db_simulation_model_version
):
    """
    Upload model parameters and production tables to database.

    Parameters
    ----------
    repository_dir : Path
        Directory containing the cloned repository.
    db_config : dict
        Database configuration.
    db_simulation_model : str
        Database simulation model name.
    db_simulation_model_version : str
        Database simulation model version.
    """
    logger = logging.getLogger()

    # Create database handler
    db = db_handler.DatabaseHandler(mongo_db_config=db_config)

    # Upload model parameters
    model_directory = repository_dir / "simulation-models" / "model_parameters"
    if model_directory.exists():
        logger.info(f"Uploading model parameters from {model_directory}")
        db_model_upload.add_model_parameters_to_db(input_path=model_directory, db=db)
    else:
        logger.warning(f"Model parameters directory not found: {model_directory}")

    # Upload production tables
    production_directory = repository_dir / "simulation-models" / "productions"
    if production_directory.exists():
        logger.info(f"Uploading production tables from {production_directory}")
        db_model_upload.add_production_tables_to_db(input_path=production_directory, db=db)
    else:
        logger.warning(f"Production tables directory not found: {production_directory}")

    # Generate compound indexes
    logger.info("Generating compound indexes")
    db_name = db.get_db_name(
        db_simulation_model_version=db_simulation_model_version, model_name=db_simulation_model
    )
    db.generate_compound_indexes(db_name=db_name)


def main():
    """Application main."""
    label = Path(__file__).stem
    args_dict, db_config = _parse(
        label, description="Upload model parameters from repository to database"
    )
    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    db_simulation_model = args_dict["db_simulation_model"]
    db_simulation_model_version = args_dict["db_simulation_model_version"]
    branch = args_dict.get("branch")

    logger.info("Starting upload from model repository to database")
    logger.info(f"Database model: {db_simulation_model}")
    logger.info(f"Database version: {db_simulation_model_version}")

    if branch:
        logger.info(f"Using branch: {branch}")
    else:
        logger.info(f"Using version tag: {db_simulation_model_version}")

    # Get server info for confirmation
    db_server = db_config.get("db_server", "localhost")

    # Print connection details for debugging
    logger.info("MongoDB connection details:")
    logger.info(f"Server: {db_server}")
    logger.info(f"Port: {db_config.get('db_api_port', 'default')}")

    # Ask for confirmation if uploading to remote database
    if not confirm_remote_database_upload(db_server):
        return

    # Set up paths
    current_dir = Path.cwd()
    tmp_dir = current_dir / "tmp_model_parameters"

    try:
        # Clone repository
        success = clone_repository(
            DEFAULT_REPOSITORY_URL, tmp_dir, version=db_simulation_model_version, branch=branch
        )

        if not success:
            logger.error("Failed to clone repository")
            return

        # Copy .env file if it exists
        parent_dir = current_dir.parent
        copy_env_file(parent_dir, tmp_dir)

        # Upload data to database
        upload_model_data_to_database(
            tmp_dir, db_config, db_simulation_model, db_simulation_model_version
        )

        logger.info("Upload completed successfully")

    except Exception as exc:
        logger.error(f"Upload failed: {exc}")
        raise
    finally:
        # Clean up temporary directory
        if tmp_dir.exists():
            logger.info(f"Cleaning up temporary directory: {tmp_dir}")
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
