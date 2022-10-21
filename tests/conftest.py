import logging
import os
from pathlib import Path
from unittest import mock

import pytest
import yaml

import simtools.io_handler
from simtools import db_handler
from simtools.configuration import Configurator

logger = logging.getLogger()


@pytest.fixture
def tmp_test_directory(tmpdir_factory):
    """
    Sets test directories.

    Some tests depend on this structure.

    """

    tmp_test_dir = tmpdir_factory.mktemp("test-data")
    tmp_sub_dirs = ["resources", "output", "simtel", "model", "application-plots"]
    for sub_dir in tmp_sub_dirs:
        tmp_sub_dir = tmp_test_dir / sub_dir
        tmp_sub_dir.mkdir()

    return tmp_test_dir


@pytest.fixture
def io_handler(tmp_test_directory):

    tmp_io_handler = simtools.io_handler.IOHandler()
    tmp_io_handler.setPaths(
        output_path=str(tmp_test_directory) + "/output",
        # TODO confirm that tests are always run from the gammasim-tools directory
        data_path="./data/",
        model_path=str(tmp_test_directory) + "/model",
    )
    return tmp_io_handler


@pytest.fixture
def mock_settings_env_vars(tmp_test_directory):
    """
    Removes all environment variable from the test system.
    Explicitely sets those needed.

    """
    with mock.patch.dict(
        os.environ, {"SIMTELPATH": str(tmp_test_directory) + "/simtel"}, clear=True
    ):
        yield


@pytest.fixture
def simtelpath(mock_settings_env_vars):
    simtelpath = Path(os.path.expandvars("$SIMTELPATH"))
    if simtelpath.exists():
        return simtelpath
    return ""


@pytest.fixture
def simtelpath_no_mock():
    simtelpath = Path(os.path.expandvars("$SIMTELPATH"))
    if simtelpath.exists():
        return simtelpath
    return ""


@pytest.fixture
def args_dict(tmp_test_directory, simtelpath):

    return Configurator().default_config(
        (
            "--output_path",
            str(tmp_test_directory),
            "--data_path",
            "./data/",
            "--simtelpath",
            str(simtelpath),
        )
    )


@pytest.fixture
def args_dict_site(tmp_test_directory, simtelpath):

    return Configurator().default_config(
        (
            "--output_path",
            str(tmp_test_directory),
            "--data_path",
            "./data/",
            "--simtelpath",
            str(simtelpath),
            "--site",
            "South",
            "--telescope",
            "MST-NectarCam-D",
        )
    )


@pytest.fixture
def configurator(tmp_test_directory, simtelpath):

    config = Configurator()
    config.default_config(
        ("--output_path", str(tmp_test_directory), "--simtelpath", str(simtelpath))
    )
    return config


def write_dummy_dbdetails_file(filename="dbDetails.yml", **kwargs):
    """
    Create a dummy dbDetails.yml file to be used in test enviroments only.

    Parameters
    ----------
    filename: str
        Name of the dummy dbDetails file (default=dbDetails.yml)
    **kwargs
        The default parameters can be overwritten using kwargs.
    """
    pars = {
        "dbPort": None,
        "mongodbServer": None,
        "userDB": None,
        "passDB": None,
        "authenticationDatabase": "admin",
    }

    if len(kwargs) > 0:
        for key, value in kwargs.items():
            pars[key] = int(value) if key == "dbPort" else str(value)

    with open(filename, "w") as outfile:
        yaml.dump(pars, outfile)


def write_configuration_test_file(config_file, config_dict):
    """
    Write a simtools configuration file

    Do not overwrite existing files
    """

    if not Path(config_file).exists():
        with open(config_file, "w") as output:
            yaml.safe_dump(config_dict, output, sort_keys=False)


@pytest.fixture
def configuration_parameters(tmp_test_directory):

    return {
        "modelFilesLocations": [
            str(tmp_test_directory / "resources/"),
            str(tmp_test_directory / "tests/resources"),
        ],
        "dataLocation": "./data/",
        "outputLocation": str(tmp_test_directory / "output"),
        "simtelPath": str(tmp_test_directory / "simtel"),
        "useMongoDB": False,
        "mongoDBConfigFile": None,
        "extraCommands": [""],
    }


@pytest.fixture
def db(db_connection):
    db = db_handler.DatabaseHandler(mongoDBConfigFile=str(db_connection))
    return db


@pytest.fixture
def db_connection(tmp_test_directory):
    # prefer local dbDetails.yml file for testing
    try:
        dbDetailsFile = "dbDetails.yml"
        with open(dbDetailsFile, "r") as stream:
            yaml.safe_load(stream)
        return dbDetailsFile
    # try if DB details are defined in environment
    # (e.g., as secrets in github actions)
    except FileNotFoundError:
        parsToDbDetails = {}
        environVariblesToCheck = {
            "mongodbServer": "DB_API_NAME",
            "userDB": "DB_API_USER",
            "passDB": "DB_API_PW",
            "dbPort": "DB_API_PORT",
        }
        found_env = True
        for par, env in environVariblesToCheck.items():
            if env in os.environ:
                parsToDbDetails[par] = os.environ[env]
            else:
                found_env = False
        if found_env:
            dbDetailsFileName = tmp_test_directory / "dbDetails.yml"
            write_dummy_dbdetails_file(filename=dbDetailsFileName, **parsToDbDetails)
            return dbDetailsFileName

        return ""
