"""Simtools dependencies version management."""

import logging
import os
import re
import subprocess
from pathlib import Path

from simtools.db.db_handler import DatabaseHandler

_logger = logging.getLogger(__name__)


def get_version_string(db_config=None):
    """Print the versions of the dependencies."""
    return (
        f"Database version: {get_database_version(db_config)}\n"
        f"sim_telarray version: {get_sim_telarray_version()}\n"
        f"CORSIKA version: {get_corsika_version()}\n"
    )


def get_database_version(db_config):
    """
    Get the version of the simulation model data base used.

    Parameters
    ----------
    db_config : dict
        Dictionary containing the database configuration.

    Returns
    -------
    str
        Version of the simulation model data base used.

    """
    if db_config is None:
        return None
    db = DatabaseHandler(db_config)
    return db.mongo_db_config.get("db_simulation_model")


def get_sim_telarray_version():
    """
    Get the version of the sim_telarray package using 'sim_telarray --version'.

    Returns
    -------
    str
        Version of the sim_telarray package.
    """
    sim_telarray_path = os.getenv("SIMTOOLS_SIMTEL_PATH")
    if sim_telarray_path is None:
        _logger.warning("Environment variable SIMTOOLS_SIMTEL_PATH is not set.")
        return None
    sim_telarray_path = Path(sim_telarray_path) / "bin" / "sim_telarray"

    # expect stdout with e.g. a line 'Release: 2024.271.0 from 2024-09-27'
    result = subprocess.run(
        [sim_telarray_path, "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    match = re.search(r"^Release:\s+(.+)", result.stdout, re.MULTILINE)

    if match:
        return match.group(1).split()[0]
    raise ValueError(f"sim_telarray release not found in {result.stdout}")


def get_corsika_version():
    """
    Get the version of the corsika package.

    Returns
    -------
    str
        Version of the corsika package.
    """
    _logger.warning("CORSIKA version not implemented yet.")
    return "7.7"
