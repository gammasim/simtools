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

import yaml

from simtools.io import ascii_handler
from simtools.version import __version__

_logger = logging.getLogger(__name__)


def get_version_string(db_config=None, run_time=None):
    """
    Print the versions of the dependencies.

    Parameters
    ----------
    db_config : dict, optional
        Database configuration dictionary.
    run_time : list, optional
        Runtime environment command (e.g., Docker).

    Returns
    -------
    str
        String containing the versions of the dependencies.

    """
    return (
        f"Database name: {get_database_version_or_name(db_config, version=False)}\n"
        f"Database version: {get_database_version_or_name(db_config, version=True)}\n"
        f"sim_telarray version: {get_sim_telarray_version(run_time)}\n"
        f"CORSIKA version: {get_corsika_version(run_time)}\n"
        f"Build options: {get_build_options(run_time)}\n"
        f"Runtime environment: {run_time if run_time else 'None'}\n"
    )


def get_software_version(software):
    """
    Return the version of the specified software package.

    Parameters
    ----------
    software : str
        Name of the software package.

    Returns
    -------
    str
        Version of the specified software package.
    """
    if software.lower() == "simtools":
        return __version__

    try:
        version_call = f"get_{software.lower()}_version"
        return globals()[version_call]()
    except KeyError as exc:
        raise ValueError(f"Unknown software: {software}") from exc


def get_database_version_or_name(db_config, version=True):
    """
    Get the version or name of the simulation model data base used.

    Parameters
    ----------
    db_config : dict
        Dictionary containing the database configuration.
    version : bool
        If True, return the version of the database. If False, return the name.

    Returns
    -------
    str
        Version or name of the simulation model data base used.

    """
    if version:
        return db_config and db_config.get("db_simulation_model_version")
    return db_config and db_config.get("db_simulation_model")


def get_sim_telarray_version(run_time=None):
    """
    Get the version of the sim_telarray package using 'sim_telarray --version'.

    Version strings for sim_telarray are of the form "2024.271.0" (year.day_of_year.patch).

    Parameters
    ----------
    run_time : list, optional
        Runtime environment command (e.g., Docker).

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

    if run_time is None:
        command = [str(sim_telarray_path), "--version"]
    else:
        command = [*run_time, str(sim_telarray_path), "--version"]

    _logger.debug(f"Running command: {command}")
    result = subprocess.run(command, capture_output=True, text=True, check=False)

    # expect stdout with e.g. a line 'Release: 2024.271.0 from 2024-09-27'
    match = re.search(r"^Release:\s+(.+)", result.stdout, re.MULTILINE)
    if match:
        return match.group(1).split()[0]

    _logger.debug(f"Command output stdout: {result.stdout} stderr: {result.stderr}")

    raise ValueError(f"sim_telarray release not found in {result.stdout}")


def get_corsika_version(run_time=None):
    """
    Get the version of the CORSIKA package.

    Version strings for CORSIKA are of the form "7.7550" (major.minor with 4-digit minor).

    Parameters
    ----------
    run_time : list, optional
        Runtime environment command (e.g., Docker).

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

    if run_time is None:
        command = [str(corsika_command)]
    else:
        command = [*run_time, str(corsika_command)]

    # Below I do not use the standard context manager because
    # it makes mocking in the tests significantly more difficult
    process = subprocess.Popen(  # pylint: disable=consider-using-with
        command,
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
        build_opts = get_build_options(run_time)
    except (FileNotFoundError, TypeError, ValueError):
        _logger.warning("Could not get CORSIKA version.")
        return None
    _logger.debug("Getting the CORSIKA version from the build options.")
    return build_opts.get("corsika_version")


def get_build_options(run_time=None):
    """
    Return CORSIKA / sim_telarray build options.

    Expects a build_opts.yml file in the sim_telarray directory.

    Parameters
    ----------
    run_time : list, optional
        Runtime environment command (e.g., Docker).

    Returns
    -------
    dict
        Build options from build_opts.yml file.
    """
    sim_telarray_path = os.getenv("SIMTOOLS_SIMTEL_PATH")
    if sim_telarray_path is None:
        raise ValueError("SIMTOOLS_SIMTEL_PATH not defined.")

    build_opts_path = Path(sim_telarray_path) / "build_opts.yml"

    if run_time is None:
        try:
            return ascii_handler.collect_data_from_file(build_opts_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError("No build_opts.yml file found.") from exc

    command = [*run_time, "cat", str(build_opts_path)]
    _logger.debug(f"Reading build_opts.yml with command: {command}")

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise FileNotFoundError(f"No build_opts.yml file found in container: {result.stderr}")

    try:
        return yaml.safe_load(result.stdout)
    except yaml.YAMLError as exc:
        raise ValueError(f"Error parsing build_opts.yml from container: {exc}") from exc
