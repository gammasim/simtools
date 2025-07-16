#!/usr/bin/python3

import logging
from unittest import mock

import pytest

from simtools import dependencies
from simtools.db.db_handler import DatabaseHandler


@pytest.fixture
def mongo_db_config():
    return {"db_simulation_model": "v1.0.0"}


@pytest.fixture
def db_config():
    return {"host": "localhost", "port": 27017}


@pytest.fixture
def db_handler_function():
    return "simtools.dependencies.DatabaseHandler"


@pytest.fixture
def fake_path():
    return "/fake/path"


@pytest.fixture
def corsika_version_string():
    return "NUMBER OF VERSION :  7.7550\n"


@pytest.fixture
def corsika_request_for_input():
    return "DATA CARDS FOR RUN STEERING ARE EXPECTED FROM STANDARD INPUT\n"


@pytest.fixture
def subprocess_run():
    return "subprocess.run"


@pytest.fixture
def subprocess_popen():
    return "subprocess.Popen"


@pytest.fixture
def env_not_set_error():
    return "Environment variable SIMTOOLS_SIMTEL_PATH is not set."


@pytest.fixture
def get_build_options_literal():
    return "simtools.dependencies.get_build_options"


def test_get_version_string_success(
    monkeypatch,
    mongo_db_config,
    db_config,
    db_handler_function,
    fake_path,
    corsika_version_string,
    corsika_request_for_input,
    subprocess_run,
    subprocess_popen,
    get_build_options_literal,
):
    mock_db_handler = mock.MagicMock(spec=DatabaseHandler)
    mock_db_handler.mongo_db_config = mongo_db_config

    with mock.patch(db_handler_function, return_value=mock_db_handler):
        monkeypatch.setenv("SIMTOOLS_SIMTEL_PATH", fake_path)

        mock_sim_telarray_result = mock.Mock()
        mock_sim_telarray_result.stdout = "Release: 2024.271.0 from 2024-09-27"
        mock_sim_telarray_result.stderr = ""

        mock_corsika_process = mock.Mock()
        mock_corsika_process.stdout.readline.side_effect = [
            corsika_version_string,
            corsika_request_for_input,
            "",
        ]

        with mock.patch(subprocess_run, return_value=mock_sim_telarray_result):
            with mock.patch(subprocess_popen, return_value=mock_corsika_process):
                with mock.patch(get_build_options_literal, return_value={"corsika_version": "7.7"}):
                    expected_output = (
                        "Database version: v1.0.0\n"
                        "sim_telarray version: 2024.271.0\n"
                        "CORSIKA version: 7.7550\n"
                        "Build options: {'corsika_version': '7.7'}\n"
                        "Runtime environment: None\n"
                    )
                    assert dependencies.get_version_string(db_config) == expected_output


def test_get_version_string_no_env_var(
    monkeypatch,
    mongo_db_config,
    db_config,
    db_handler_function,
    env_not_set_error,
    caplog,
    get_build_options_literal,
):
    mock_db_handler = mock.MagicMock(spec=DatabaseHandler)
    mock_db_handler.mongo_db_config = mongo_db_config

    with mock.patch(db_handler_function, return_value=mock_db_handler):
        monkeypatch.delenv("SIMTOOLS_SIMTEL_PATH", raising=False)

        with caplog.at_level(logging.WARNING):
            with mock.patch(get_build_options_literal, return_value=None):
                expected_output = (
                    "Database version: v1.0.0\n"
                    "sim_telarray version: None\n"
                    "CORSIKA version: None\n"
                    "Build options: None\n"
                    "Runtime environment: None\n"
                )
                assert dependencies.get_version_string(db_config) == expected_output

        assert env_not_set_error in caplog.text


def test_get_database_version_success(mongo_db_config, db_config, db_handler_function):
    mock_db_handler = mock.MagicMock(spec=DatabaseHandler)
    mock_db_handler.mongo_db_config = mongo_db_config

    with mock.patch(db_handler_function, return_value=mock_db_handler):
        assert dependencies.get_database_version(db_config) == "v1.0.0"

    assert dependencies.get_database_version(None) is None


def test_get_database_version_no_version(db_config, db_handler_function):
    mock_db_handler = mock.MagicMock(spec=DatabaseHandler)
    mock_db_handler.mongo_db_config = {}

    with mock.patch(db_handler_function, return_value=mock_db_handler):
        assert dependencies.get_database_version(db_config) is None


def test_get_sim_telarray_version_success(monkeypatch, fake_path, subprocess_run):
    monkeypatch.setenv("SIMTOOLS_SIMTEL_PATH", fake_path)
    expected_version = "2024.271.0"
    mock_result = mock.Mock()
    mock_result.stdout = "Release: 2024.271.0 from 2024-09-27"
    mock_result.stderr = ""

    with mock.patch(subprocess_run, return_value=mock_result):
        assert dependencies.get_sim_telarray_version(None) == expected_version

    with mock.patch(subprocess_run, return_value=mock_result):
        assert dependencies.get_sim_telarray_version("podman run") == expected_version


def test_get_sim_telarray_version_no_env_var(caplog, monkeypatch, env_not_set_error):
    monkeypatch.delenv("SIMTOOLS_SIMTEL_PATH", raising=False)

    with caplog.at_level(logging.WARNING):
        assert dependencies.get_sim_telarray_version(None) is None

    assert env_not_set_error in caplog.text


def test_get_sim_telarray_version_no_release(monkeypatch, subprocess_run):
    monkeypatch.setenv("SIMTOOLS_SIMTEL_PATH", "/fake/path_simtel")
    mock_result = mock.Mock()
    mock_result.stdout = "Some other output"
    mock_result.stderr = ""

    with mock.patch(subprocess_run, return_value=mock_result):
        with pytest.raises(ValueError, match="sim_telarray release not found in Some other output"):
            dependencies.get_sim_telarray_version(None)


def test_build_options(monkeypatch, fake_path):
    # no SIMTEL_PATH defined
    monkeypatch.delenv("SIMTOOLS_SIMTEL_PATH", raising=False)
    with pytest.raises(TypeError, match="SIMTOOLS_SIMTEL_PATH not defined"):
        dependencies.get_build_options()
    # SIMTEL_PATH defined, but no build_opts.yml file
    monkeypatch.setenv("SIMTOOLS_SIMTEL_PATH", fake_path)
    with pytest.raises(FileNotFoundError, match="No build_opts.yml file found."):
        dependencies.get_build_options()

    # mock gen.collect_data_from_file to return a dict
    with mock.patch(
        "simtools.dependencies.gen.collect_data_from_file", return_value={"corsika_version": "7.7"}
    ):
        build_opts = dependencies.get_build_options()
        assert build_opts == {"corsika_version": "7.7"}


def test_get_corsika_version_success(
    monkeypatch, fake_path, corsika_version_string, corsika_request_for_input, subprocess_popen
):
    monkeypatch.setenv("SIMTOOLS_SIMTEL_PATH", fake_path)
    mock_process = mock.Mock()
    mock_process.stdout.readline.side_effect = [
        corsika_version_string,
        corsika_request_for_input,
        "",
    ]

    with mock.patch(subprocess_popen, return_value=mock_process):
        assert dependencies.get_corsika_version() == "7.7550"


def test_get_corsika_version_no_env_var(caplog, monkeypatch, env_not_set_error):
    monkeypatch.delenv("SIMTOOLS_SIMTEL_PATH", raising=False)

    with caplog.at_level(logging.WARNING):
        assert dependencies.get_corsika_version() is None

    assert env_not_set_error in caplog.text


def test_get_corsika_version_no_version(
    monkeypatch,
    fake_path,
    corsika_request_for_input,
    subprocess_popen,
    get_build_options_literal,
    caplog,
):
    monkeypatch.setenv("SIMTOOLS_SIMTEL_PATH", fake_path)
    mock_process = mock.Mock()
    mock_process.stdout.readline.side_effect = [
        "Some other output\n",
        corsika_request_for_input,
        "",
    ]

    with mock.patch(subprocess_popen, return_value=mock_process):
        with mock.patch(get_build_options_literal, return_value={"corsika_version": "7.7"}):
            with caplog.at_level(logging.DEBUG):
                assert dependencies.get_corsika_version() == "7.7"

            assert "Getting the CORSIKA version from the build options." in caplog.text

            assert dependencies.get_corsika_version("podman run") == "7.7"


def test_get_corsika_version_no_build_opts(
    monkeypatch,
    fake_path,
    corsika_request_for_input,
    subprocess_popen,
    get_build_options_literal,
    caplog,
):
    monkeypatch.setenv("SIMTOOLS_SIMTEL_PATH", fake_path)
    mock_process = mock.Mock()
    mock_process.stdout.readline.side_effect = [
        "Some other output\n",
        corsika_request_for_input,
        "",
    ]

    with mock.patch(subprocess_popen, return_value=mock_process):
        with mock.patch(get_build_options_literal, side_effect=FileNotFoundError):
            with caplog.at_level(logging.WARNING):
                assert dependencies.get_corsika_version() is None

            assert "Could not get CORSIKA version." in caplog.text


def test_get_corsika_version_empty_line(
    monkeypatch, fake_path, corsika_version_string, subprocess_popen, get_build_options_literal
):
    # The empty line causes the function to break before reaching the version
    # When actually running CORSIKA, we won't get an empty line in the stdout
    # but rather "\n" at the end of the line. An empty line is a sign that the
    # process has finished.
    monkeypatch.setenv("SIMTOOLS_SIMTEL_PATH", fake_path)
    mock_process = mock.Mock()
    mock_process.stdout.readline.side_effect = [
        "",  # Empty line first
        corsika_version_string,  # This should not be reached due to break
    ]

    with mock.patch(subprocess_popen, return_value=mock_process):
        with mock.patch(get_build_options_literal, return_value={"corsika_version": "7.7"}):
            assert dependencies.get_corsika_version() == "7.7"

    # Verify readline was called only once before breaking
    assert mock_process.stdout.readline.call_count == 1


def test_get_build_options_no_env_var(monkeypatch):
    monkeypatch.delenv("SIMTOOLS_SIMTEL_PATH", raising=False)
    with pytest.raises(TypeError, match="SIMTOOLS_SIMTEL_PATH not defined."):
        dependencies.get_build_options()


def test_get_build_options_file_not_found(monkeypatch, fake_path):
    monkeypatch.setenv("SIMTOOLS_SIMTEL_PATH", fake_path)
    with pytest.raises(FileNotFoundError, match="No build_opts.yml file found."):
        dependencies.get_build_options()


def test_get_build_options_success(monkeypatch, fake_path):
    monkeypatch.setenv("SIMTOOLS_SIMTEL_PATH", fake_path)
    mock_build_opts = {"corsika_version": "7.7"}
    with mock.patch(
        "simtools.dependencies.gen.collect_data_from_file", return_value=mock_build_opts
    ):
        assert dependencies.get_build_options() == mock_build_opts


def test_get_build_options_container_file_not_found(monkeypatch, fake_path, subprocess_run):
    monkeypatch.setenv("SIMTOOLS_SIMTEL_PATH", fake_path)
    mock_result = mock.Mock()
    mock_result.returncode = 1
    mock_result.stderr = "File not found in container."
    with mock.patch(subprocess_run, return_value=mock_result):
        with pytest.raises(
            FileNotFoundError,
            match="No build_opts.yml file found in container: File not found in container.",
        ):
            dependencies.get_build_options(run_time=["docker", "exec", "container"])


def test_get_build_options_container_success(monkeypatch, fake_path, subprocess_run):
    monkeypatch.setenv("SIMTOOLS_SIMTEL_PATH", fake_path)
    mock_result = mock.Mock()
    mock_result.returncode = 0
    mock_result.stdout = "corsika_version: '7.7'"
    with mock.patch(subprocess_run, return_value=mock_result):
        assert dependencies.get_build_options(run_time=["docker", "exec", "container"]) == {
            "corsika_version": "7.7"
        }


def test_get_build_options_container_yaml_error(monkeypatch, fake_path, subprocess_run):
    monkeypatch.setenv("SIMTOOLS_SIMTEL_PATH", fake_path)
    mock_result = mock.Mock()
    mock_result.returncode = 0
    mock_result.stdout = "invalid_yaml: ["
    with mock.patch(subprocess_run, return_value=mock_result):
        with pytest.raises(ValueError, match="Error parsing build_opts.yml from container:"):
            dependencies.get_build_options(run_time=["docker", "exec", "container"])
