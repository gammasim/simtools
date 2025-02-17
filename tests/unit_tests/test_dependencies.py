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


def test_get_sim_telarray_version_success():
    with mock.patch.dict(os.environ, {"SIMTOOLS_SIMTEL_PATH": "/fake/path"}):
        version_content = '#define BASE_RELEASE "1.2.3"'

        with mock.patch("builtins.open", mock.mock_open(read_data=version_content)):
            assert get_sim_telarray_version() == "1.2.3"


def test_get_sim_telarray_version_env_not_set():
    with mock.patch.dict(os.environ, {}, clear=True):
        assert get_sim_telarray_version() is None


def test_get_sim_telarray_version_file_not_found():
    with mock.patch.dict(os.environ, {"SIMTOOLS_SIMTEL_PATH": "/fake/path"}):
        with mock.patch("pathlib.Path.open", side_effect=FileNotFoundError):
            with pytest.raises(FileNotFoundError, match="sim_telarray version file not found."):
                get_sim_telarray_version()


def test_get_sim_telarray_version_base_release_not_found():
    with mock.patch.dict(os.environ, {"SIMTOOLS_SIMTEL_PATH": "/fake/path"}):
        version_content = '#define SOME_OTHER_DEFINE "1.2.3"'

        with mock.patch("builtins.open", mock.mock_open(read_data=version_content)):
            with pytest.raises(
                ValueError, match="sim_telarray BASE_RELEASE not found in the file."
            ):
                get_sim_telarray_version()


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
