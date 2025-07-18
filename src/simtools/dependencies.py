"""
Simtools dependencies version management.

This modules provides two main functionalities:

- retrieve the versions of simtools dependencies (e.g., databases, sim_telarray, CORSIKA)
- provide space for future implementations of version management

"""

import logging
import os
import re
import subprocess
from pathlib import Path

from simtools.db.db_handler import DatabaseHandler
from simtools.io_operations import ascii_handler

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
    sim_telarray_path = Path(sim_telarray_path) / "sim_telarray" / "bin" / "sim_telarray"

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
    Get the version of the CORSIKA package.

    Returns
    -------
    str
        Version of the CORSIKA package.
    """
    version = None
    sim_telarray_path = os.getenv("SIMTOOLS_SIMTEL_PATH")
    if sim_telarray_path is None:
        _logger.warning("Environment variable SIMTOOLS_SIMTEL_PATH is not set.")
        return None
    corsika_command = Path(sim_telarray_path) / "corsika-run" / "corsika"

    # Below I do not use the standard context manager because
    # it makes mocking in the tests significantly more difficult
    process = subprocess.Popen(  # pylint: disable=consider-using-with
        corsika_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE,
        text=True,
    )

    # Capture output until it waits for input
    while True:
        line = process.stdout.readline()
        if not line:
            break
        # Extract the version from the line "NUMBER OF VERSION :  7.7550"
        if "NUMBER OF VERSION" in line:
            version = line.split(":")[1].strip()
            break
        # Check for a specific prompt or indication that the program is waiting for input
        if "DATA CARDS FOR RUN STEERING ARE EXPECTED FROM STANDARD INPUT" in line:
            break

    process.terminate()
    # Check it's a valid version string
    if version and re.match(r"\d+\.\d+", version):
        return version
    try:
        build_opts = get_build_options()
    except (FileNotFoundError, TypeError):
        _logger.warning("Could not get CORSIKA version.")
        return None
    _logger.debug("Getting the CORSIKA version from the build options.")
    return build_opts.get("corsika_version")


def get_build_options():
    """
    Return CORSIKA / sim_telarray build options.

    Expects a build_opts.yml file in the sim_telarray directory.
    """
    try:
        return ascii_handler.collect_data_from_file(
            Path(os.getenv("SIMTOOLS_SIMTEL_PATH")) / "build_opts.yml"
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError("No build_opts.yml file found.") from exc
    except TypeError as exc:
        raise TypeError("SIMTOOLS_SIMTEL_PATH not defined.") from exc
