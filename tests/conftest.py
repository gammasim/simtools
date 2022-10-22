import logging
import os
from pathlib import Path
from unittest import mock

import pytest

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
        os.environ,
        {
            "SIMTELPATH": str(tmp_test_directory) + "/simtel",
            "DB_API_USER": "db_user",
            "DB_API_PW": "12345",
            "DB_API_PORT": "42",
            "DB_API_NAME": "abc@def.de",
        },
        clear=True,
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
        ),
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


@pytest.fixture
def db_config():
    """
    Read DB configuration from tests from environmental variables

    """
    mongoDBConfig = {}
    _db_para = ("db_api_user", "db_api_pw", "db_api_port", "db_api_name")
    for _para in _db_para:
        mongoDBConfig[_para] = os.environ.get(_para.upper())
    if mongoDBConfig["db_api_port"] is not None:
        mongoDBConfig["db_api_port"] = int(mongoDBConfig["db_api_port"])
    return mongoDBConfig


@pytest.fixture
def db(db_config):
    db = db_handler.DatabaseHandler(mongoDBConfig=db_config)
    return db
