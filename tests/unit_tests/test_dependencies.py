#!/usr/bin/python3

import logging
import os
from unittest import mock

import pytest

from simtools.db.db_handler import DatabaseHandler
from simtools.dependencies import (
    get_corsika_version,
    get_database_version,
    get_sim_telarray_version,
)


def test_get_database_version_success():
    db_config = {"host": "localhost", "port": 27017}
    mock_db_handler = mock.MagicMock(spec=DatabaseHandler)
    mock_db_handler.mongo_db_config = {"db_simulation_model": "v1.0.0"}

    with mock.patch("simtools.dependencies.DatabaseHandler", return_value=mock_db_handler):
        assert get_database_version(db_config) == "v1.0.0"


def test_get_database_version_no_version():
    db_config = {"host": "localhost", "port": 27017}
    mock_db_handler = mock.MagicMock(spec=DatabaseHandler)
    mock_db_handler.mongo_db_config = {}

    with mock.patch("simtools.dependencies.DatabaseHandler", return_value=mock_db_handler):
        assert get_database_version(db_config) is None


def test_get_corsika_version(caplog):

    with caplog.at_level(logging.WARNING):
        assert get_corsika_version() == "7.7"

    assert "CORSIKA version not implemented yet." in caplog.text


def test_get_sim_telarray_version_success():
    os.environ["SIMTOOLS_SIMTEL_PATH"] = "/fake/path"
    expected_version = "2024.271.0"
    mock_result = mock.Mock()
    mock_result.stdout = "Release: 2024.271.0 from 2024-09-27"
    mock_result.stderr = ""

    with mock.patch("subprocess.run", return_value=mock_result):
        assert get_sim_telarray_version() == expected_version


def test_get_sim_telarray_version_no_env_var(caplog):
    if "SIMTOOLS_SIMTEL_PATH" in os.environ:
        del os.environ["SIMTOOLS_SIMTEL_PATH"]

    with caplog.at_level(logging.WARNING):
        assert get_sim_telarray_version() is None

    assert "Environment variable SIMTOOLS_SIMTEL_PATH is not set." in caplog.text


def test_get_sim_telarray_version_no_release():
    os.environ["SIMTOOLS_SIMTEL_PATH"] = "/fake/path"
    mock_result = mock.Mock()
    mock_result.stdout = "Some other output"
    mock_result.stderr = ""

    with mock.patch("subprocess.run", return_value=mock_result):
        with pytest.raises(ValueError, match="sim_telarray release not found in Some other output"):
            get_sim_telarray_version()
