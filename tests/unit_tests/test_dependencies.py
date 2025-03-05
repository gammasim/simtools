#!/usr/bin/python3

import logging
from unittest import mock

import pytest

from simtools import dependencies
from simtools.db.db_handler import DatabaseHandler


def test_get_database_version_success():
    db_config = {"host": "localhost", "port": 27017}
    mock_db_handler = mock.MagicMock(spec=DatabaseHandler)
    mock_db_handler.mongo_db_config = {"db_simulation_model": "v1.0.0"}

    with mock.patch("simtools.dependencies.DatabaseHandler", return_value=mock_db_handler):
        assert dependencies.get_database_version(db_config) == "v1.0.0"


def test_get_database_version_no_version():
    db_config = {"host": "localhost", "port": 27017}
    mock_db_handler = mock.MagicMock(spec=DatabaseHandler)
    mock_db_handler.mongo_db_config = {}

    with mock.patch("simtools.dependencies.DatabaseHandler", return_value=mock_db_handler):
        assert dependencies.get_database_version(db_config) is None


def test_get_corsika_version(caplog):
    # no build_opts.yml file
    with caplog.at_level(logging.WARNING):
        assert dependencies.get_corsika_version() is None
    assert "CORSIKA version not implemented yet." in caplog.text

    # mock get_build_options to return a dict
    with mock.patch(
        "simtools.dependencies.get_build_options", return_value={"corsika_version": "7.7"}
    ):
        assert dependencies.get_corsika_version() == "7.7"


def test_get_sim_telarray_version_success(monkeypatch):
    monkeypatch.setenv("SIMTOOLS_SIMTEL_PATH", "/fake/path")
    expected_version = "2024.271.0"
    mock_result = mock.Mock()
    mock_result.stdout = "Release: 2024.271.0 from 2024-09-27"
    mock_result.stderr = ""

    subprocess_mock = "subprocess.run"
    with mock.patch(subprocess_mock, return_value=mock_result):
        assert dependencies.get_sim_telarray_version() == expected_version

    with mock.patch(subprocess_mock, return_value=mock_result):
        version_string = dependencies.get_version_string()
        assert "Database version: None" in version_string
        assert "sim_telarray version:" in version_string


def test_get_sim_telarray_version_no_env_var(caplog, monkeypatch):
    monkeypatch.delenv("SIMTOOLS_SIMTEL_PATH", raising=False)

    with caplog.at_level(logging.WARNING):
        assert dependencies.get_sim_telarray_version() is None

    assert "Environment variable SIMTOOLS_SIMTEL_PATH is not set." in caplog.text


def test_get_sim_telarray_version_no_release(monkeypatch):
    monkeypatch.setenv("SIMTOOLS_SIMTEL_PATH", "/fake/path_simtel")
    mock_result = mock.Mock()
    mock_result.stdout = "Some other output"
    mock_result.stderr = ""

    with mock.patch("subprocess.run", return_value=mock_result):
        with pytest.raises(ValueError, match="sim_telarray release not found in Some other output"):
            dependencies.get_sim_telarray_version()


def test_build_options(monkeypatch):
    # no SIMTEL_PATH defined
    with pytest.raises(TypeError, match="SIMTEL_PATH not defined"):
        dependencies.get_build_options()
    # SIMTEL_PATH defined, but no build_opts.yml file
    monkeypatch.setenv("SIMTOOLS_SIMTEL_PATH", "/fake/path")
    with pytest.raises(FileNotFoundError, match="No build_opts.yml file found."):
        dependencies.get_build_options()

    # mock gen.collect_data_from_file to return a dict
    with mock.patch(
        "simtools.dependencies.gen.collect_data_from_file", return_value={"corsika_version": "7.7"}
    ):
        build_opts = dependencies.get_build_options()
        assert build_opts == {"corsika_version": "7.7"}
