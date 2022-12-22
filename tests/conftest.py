import logging
import os
from pathlib import Path
from unittest import mock

import pytest

import simtools.io_handler
import simtools.util.general as gen
from simtools import db_handler
from simtools.configuration.configurator import Configurator
from simtools.layout.layout_array import LayoutArray
from simtools.model.telescope_model import TelescopeModel

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
    tmp_io_handler.set_paths(
        output_path=str(tmp_test_directory) + "/output",
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
            "SIMTEL_PATH": str(tmp_test_directory) + "/simtel",
            "DB_API_USER": "db_user",
            "DB_API_PW": "12345",
            "DB_API_PORT": "42",
            "DB_SERVER": "abc@def.de",
        },
        clear=True,
    ):
        yield


@pytest.fixture
def simtel_path(mock_settings_env_vars):
    simtel_path = Path(os.path.expandvars("$SIMTEL_PATH"))
    if simtel_path.exists():
        return simtel_path
    return ""


@pytest.fixture
def simtel_path_no_mock():
    simtel_path = Path(os.path.expandvars("$SIMTEL_PATH"))
    if simtel_path.exists():
        return simtel_path
    return ""


@pytest.fixture
def args_dict(tmp_test_directory, simtel_path):

    return Configurator().default_config(
        (
            "--output_path",
            str(tmp_test_directory),
            "--data_path",
            "./data/",
            "--simtel_path",
            str(simtel_path),
        ),
    )


@pytest.fixture
def args_dict_site(tmp_test_directory, simtel_path):

    return Configurator().default_config(
        (
            "--output_path",
            str(tmp_test_directory),
            "--data_path",
            "./data/",
            "--simtel_path",
            str(simtel_path),
            "--site",
            "South",
            "--telescope",
            "MST-NectarCam-D",
            "--label",
            "integration_test",
        )
    )


@pytest.fixture
def configurator(tmp_test_directory, simtel_path):

    config = Configurator()
    config.default_config(
        ("--output_path", str(tmp_test_directory), "--simtel_path", str(simtel_path))
    )
    return config


@pytest.fixture
def db_config():
    """
    Read DB configuration from tests from environmental variables
    """
    mongo_db_config = {}
    _db_para = ("db_api_user", "db_api_pw", "db_api_port", "db_server")
    for _para in _db_para:
        mongo_db_config[_para] = os.environ.get(_para.upper())
    if mongo_db_config["db_api_port"] is not None:
        mongo_db_config["db_api_port"] = int(mongo_db_config["db_api_port"])
    return mongo_db_config


@pytest.fixture
def db(db_config):
    db = db_handler.DatabaseHandler(mongo_db_config=db_config)
    return db


@pytest.fixture
def db_no_config_file():
    """
    Same as db above, but without DB variable defined,
    since we do not want to set the config file as well.
    Otherwise it creates a conflict between the config file
    set by set_db and the one set by set_simtools
    """
    db = db_handler.DatabaseHandler(mongo_db_config=None)
    return db


@pytest.fixture()
def db_cleanup_file_sandbox(db_no_config_file):
    yield
    # Cleanup
    logger.info("Dropping the temporary files in the sandbox")
    db_no_config_file.db_client["sandbox"]["fs.chunks"].drop()
    db_no_config_file.db_client["sandbox"]["fs.files"].drop()


@pytest.fixture
def telescope_model_lst(db, db_config, io_handler):
    telescope_model_LST = TelescopeModel(
        site="North",
        telescope_model_name="LST-1",
        model_version="Prod5",
        mongo_db_config=db_config,
        label="validate_camera_efficiency",
    )
    return telescope_model_LST


@pytest.fixture
def telescope_model_sst(db, db_config, io_handler):
    telescope_model_SST = TelescopeModel(
        site="South",
        telescope_model_name="SST-D",
        model_version="Prod5",
        mongo_db_config=db_config,
        label="test-telescope-model-sst",
    )
    return telescope_model_SST


@pytest.fixture
def telescope_test_file(db, args_dict, io_handler):
    test_file_name = "telescope_positions-North-TestLayout.ecsv"
    db.export_file_db(
        db_name="test-data",
        dest=io_handler.get_output_directory(dir_type="model", test=True),
        file_name=test_file_name,
    )

    cfg_file = gen.find_file(
        test_file_name,
        io_handler.get_output_directory(dir_type="model", test=True),
    )
    return cfg_file


@pytest.fixture
def layout_array_north_instance(io_handler, db_config):
    return LayoutArray(site="North", mongo_db_config=db_config, name="test_layout")


@pytest.fixture
def layout_array_south_instance(io_handler, db_config):
    return LayoutArray(site="South", mongo_db_config=db_config, name="test_layout")
