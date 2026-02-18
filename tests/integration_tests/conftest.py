"""Common fixtures for integration tests."""

import os

import pytest
from dotenv import load_dotenv

import simtools.io.io_handler
from simtools import settings


def pytest_addoption(parser):
    """Model version command line parameter."""
    parser.addoption("--model_version", action="store", default=None)


@pytest.fixture(autouse=True)
def simtools_settings(db_config):
    """Load simtools settings for the test session."""
    load_dotenv(".env")
    settings.config.load(db_config=db_config)


@pytest.fixture
def db_config():
    """DB configuration from .env file."""
    load_dotenv(".env")

    _db_para = (
        "db_api_user",
        "db_api_pw",
        "db_api_port",
        "db_api_authentication_database",
        "db_server",
        "db_simulation_model",
        "db_simulation_model_version",
    )
    db_config = {_para: os.environ.get(f"SIMTOOLS_{_para.upper()}") for _para in _db_para}
    if db_config["db_api_port"] is not None:
        db_config["db_api_port"] = int(db_config["db_api_port"])
    return db_config


@pytest.fixture
def tmp_test_directory(tmpdir_factory):
    """Sets temporary test directories. Some tests depend on this structure."""

    tmp_test_dir = tmpdir_factory.mktemp("test-data")
    tmp_sub_dirs = ["resources", "output", "sim_telarray", "model", "application-plots"]
    for sub_dir in tmp_sub_dirs:
        tmp_sub_dir = tmp_test_dir / sub_dir
        tmp_sub_dir.mkdir()

    return tmp_test_dir


@pytest.fixture(autouse=True)
def io_handler(tmp_test_directory):
    """Define io_handler fixture including output and model directories."""
    tmp_io_handler = simtools.io.io_handler.IOHandler()
    tmp_io_handler.set_paths(
        output_path=str(tmp_test_directory) + "/output",
        model_path=str(tmp_test_directory) + "/model",
    )
    return tmp_io_handler
