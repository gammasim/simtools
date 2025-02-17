"""Simtools dependencies version management."""

import logging
import os
import re
from pathlib import Path

from simtools.db.db_handler import DatabaseHandler

_logger = logging.getLogger(__name__)


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
    Get the version of the sim_telarray package.

    Returns
    -------
    str
        Version of the sim_telarray package.
    """
    sim_telarray_path = os.getenv("SIMTOOLS_SIMTEL_PATH")
    if sim_telarray_path is None:
        _logger.warning("Environment variable SIMTOOLS_SIMTEL_PATH is not set.")
        return None
    version_file = Path(sim_telarray_path) / "sim_telarray" / "version.h"
    try:
        with open(version_file, encoding="utf-8") as file:
            content = file.read()
    except FileNotFoundError as exc:
        raise FileNotFoundError("sim_telarray version file not found.") from exc

    match = re.search(r'#define BASE_RELEASE\s+"([^"]+)"', content)

    if match:
        return match.group(1)
    raise ValueError("sim_telarray BASE_RELEASE not found in the file.")


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
