#!/usr/bin/python3

import subprocess
from pathlib import Path
from unittest import mock

import pytest
import yaml

from simtools.runners import simtools_runner

TEST_OUTPUT_PATH = Path("output/test")
DUMMY_CONFIG_FILE = "dummy.yml"


@pytest.fixture
def mock_logger():
    return mock.Mock()


@pytest.fixture
def mock_collect_data_from_file(monkeypatch):
    def _mocked_collect_data_from_file(applications):
        def inner(_):
            return {"applications": applications}

        return inner

    return _mocked_collect_data_from_file


@pytest.fixture
def mock_set_input_output_directories(monkeypatch):
    def _mocked_set_input_output_directories(_):
        return (Path("output/test_workflow"), "test_workflow")

    return _mocked_set_input_output_directories


@pytest.fixture
def mock_change_dict_keys_case(monkeypatch):
    def _mocked_change_dict_keys_case(d, _):
        return d

    return _mocked_change_dict_keys_case


def test_set_input_output_directories():
    path = "input/LSTN-01/pm_photoelectron_spectrum/20250304T073000/config.yml"
    output_path, setting_workflow = simtools_runner._set_input_output_directories(path)
    assert str(output_path) == "output/LSTN-01/pm_photoelectron_spectrum/20250304T073000"

    assert setting_workflow == "LSTN-01/pm_photoelectron_spectrum/20250304T073000"

    path = "output/LSTN-01/pm_photoelectron_spectrum/20250304T073000/config.yml"
    with pytest.raises(ValueError, match="^Could not find subdirectory under"):
        simtools_runner._set_input_output_directories(path)


def test_replace_placeholders_in_configuration_replaces_string():
    config = {"input_file": "__SETTING_WORKFLOW__/data.txt", "other_key": "no_placeholder"}
    output_path = TEST_OUTPUT_PATH
    setting_workflow = "LSTN-01/workflow"
    result = simtools_runner._replace_placeholders_in_configuration(
        config.copy(), output_path, setting_workflow
    )
    assert result["input_file"] == "LSTN-01/workflow/data.txt"
    assert result["other_key"] == "no_placeholder"
    assert result["use_plain_output_path"] is True
    assert result["output_path"] == str(output_path)


def test_replace_placeholders_in_configuration_replaces_in_list():
    config = {"files": ["__SETTING_WORKFLOW__/a.txt", "__SETTING_WORKFLOW__/b.txt", 42, None]}
    output_path = TEST_OUTPUT_PATH
    setting_workflow = "WF"
    result = simtools_runner._replace_placeholders_in_configuration(
        config.copy(), output_path, setting_workflow
    )
    assert result["files"][0] == "WF/a.txt"
    assert result["files"][1] == "WF/b.txt"
    assert result["files"][2] == 42
    assert result["files"][3] is None
    assert result["use_plain_output_path"] is True
    assert result["output_path"] == str(output_path)


def test_replace_placeholders_in_configuration_no_placeholder():
    config = {"key": "value", "list": ["item1", "item2"]}
    output_path = TEST_OUTPUT_PATH
    setting_workflow = "WF"
    result = simtools_runner._replace_placeholders_in_configuration(
        config.copy(), output_path, setting_workflow
    )
    assert result["key"] == "value"
    assert result["list"] == ["item1", "item2"]
    assert result["use_plain_output_path"] is True
    assert result["output_path"] == str(output_path)


def test_replace_placeholders_in_configuration_empty_config():
    config = {}
    output_path = TEST_OUTPUT_PATH
    setting_workflow = "WF"
    result = simtools_runner._replace_placeholders_in_configuration(
        config.copy(), output_path, setting_workflow
    )
    assert result["use_plain_output_path"] is True
    assert result["output_path"] == str(output_path)


def test_read_application_configuration_selected_steps(
    monkeypatch,
    mock_logger,
    mock_collect_data_from_file,
    mock_set_input_output_directories,
    mock_change_dict_keys_case,
):
    applications = [
        {"application": "app1", "configuration": {"key": "value"}},
        {"application": "app2", "configuration": {"key": "value2"}},
        {"application": "app3", "configuration": {"key": "value3"}},
    ]
    monkeypatch.setattr(
        "simtools.utils.general.collect_data_from_file", mock_collect_data_from_file(applications)
    )
    monkeypatch.setattr(
        "simtools.runners.simtools_runner._set_input_output_directories",
        mock_set_input_output_directories,
    )
    monkeypatch.setattr("simtools.utils.general.change_dict_keys_case", mock_change_dict_keys_case)
    monkeypatch.setattr(
        "simtools.runners.simtools_runner._replace_placeholders_in_configuration",
        lambda config, output_path, setting_workflow: {**config, "output_path": str(output_path)},
    )

    configs, log_file = simtools_runner._read_application_configuration(
        DUMMY_CONFIG_FILE, [2], mock_logger
    )
    assert configs[0]["run_application"] is False
    assert configs[1]["run_application"] is True
    assert configs[2]["run_application"] is False


def test_read_application_configuration_empty_applications(
    monkeypatch,
    mock_logger,
    mock_collect_data_from_file,
    mock_set_input_output_directories,
    mock_change_dict_keys_case,
):
    applications = []
    monkeypatch.setattr(
        "simtools.utils.general.collect_data_from_file", mock_collect_data_from_file(applications)
    )
    monkeypatch.setattr(
        "simtools.runners.simtools_runner._set_input_output_directories",
        mock_set_input_output_directories,
    )
    monkeypatch.setattr("simtools.utils.general.change_dict_keys_case", mock_change_dict_keys_case)
    monkeypatch.setattr(
        "simtools.runners.simtools_runner._replace_placeholders_in_configuration",
        lambda config, output_path, setting_workflow: config,
    )

    configs, log_file = simtools_runner._read_application_configuration(
        DUMMY_CONFIG_FILE, None, mock_logger
    )
    assert configs == []
    assert isinstance(log_file, Path)


def test_run_application_success(monkeypatch, mock_logger, tmp_path):
    mock_result = mock.Mock()
    mock_result.stdout = "stdout output"
    mock_result.stderr = "stderr output"

    # Mock yaml.dump to capture what was written
    mock_yaml_dump = mock.Mock()
    monkeypatch.setattr(yaml, "dump", mock_yaml_dump)
    monkeypatch.setattr(subprocess, "run", mock.Mock(return_value=mock_result))

    application = "dummy_app"
    configuration = {"key": "value"}
    stdout, stderr = simtools_runner.run_application(application, configuration, mock_logger)

    assert stdout == "stdout output"
    assert stderr == "stderr output"
    subprocess.run.assert_called_once()
    args, kwargs = subprocess.run.call_args
    assert args[0][0] == application
    assert args[0][1] == "--config"
    assert Path(args[0][2]).suffix == ".yml"
    assert kwargs["check"] is True
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True

    # Check that yaml.dump was called with the configuration
    mock_yaml_dump.assert_called_once()
    dump_args, dump_kwargs = mock_yaml_dump.call_args
    assert dump_args[0] == configuration
    assert dump_kwargs.get("default_flow_style") is False


def test_run_application_failure(monkeypatch, mock_logger):
    exc = subprocess.CalledProcessError(
        returncode=1, cmd=["dummy_app", "--config", DUMMY_CONFIG_FILE], stderr="error occurred"
    )
    monkeypatch.setattr(subprocess, "run", mock.Mock(side_effect=exc))

    application = "dummy_app"
    configuration = {"key": "value"}

    with pytest.raises(subprocess.CalledProcessError):
        simtools_runner.run_application(application, configuration, mock_logger)
    mock_logger.error.assert_called_once_with("Error running application dummy_app: error occurred")


def test_run_applications_runs_and_logs(monkeypatch, tmp_path):
    # Prepare mocks
    mock_logger = mock.Mock()
    mock_db_config = {"db": "config"}
    mock_args_dict = {"configuration_file": "dummy_config.yml", "steps": None}

    # Prepare configurations returned by _read_application_configuration
    mock_configurations = [
        {"application": "app1", "run_application": True, "configuration": {"key": "value1"}},
        {"application": "app2", "run_application": False, "configuration": {"key": "value2"}},
        {"application": "app3", "run_application": True, "configuration": {"key": "value3"}},
    ]
    log_file_path = tmp_path / "simtools.log"

    # Patch _read_application_configuration
    monkeypatch.setattr(
        "simtools.runners.simtools_runner._read_application_configuration",
        mock.Mock(return_value=(mock_configurations, log_file_path)),
    )

    # Patch dependencies.get_version_string
    monkeypatch.setattr(
        "simtools.dependencies.get_version_string",
        mock.Mock(return_value="simtools version: 1.2.3\n"),
    )

    # Patch run_application
    def mock_run_application(app, config, logger):
        return f"{app}_stdout", f"{app}_stderr"

    monkeypatch.setattr("simtools.runners.simtools_runner.run_application", mock_run_application)

    simtools_runner.run_applications(mock_args_dict, mock_db_config, mock_logger)

    # Check log file contents
    with log_file_path.open("r", encoding="utf-8") as f:
        content = f.read()
    assert "Running simtools applications" in content
    assert "simtools version: 1.2.3" in content
    assert "Application: app1" in content
    assert "STDOUT:\napp1_stdout" in content
    assert "STDERR:\napp1_stderr" in content
    assert "Application: app3" in content
    assert "STDOUT:\napp3_stdout" in content
    assert "STDERR:\napp3_stderr" in content
    assert "Application: app2" not in content  # skipped

    # Check logger calls
    mock_logger.info.assert_any_call("Running application: app1")
    mock_logger.info.assert_any_call("Skipping application: app2")
    mock_logger.info.assert_any_call("Running application: app3")


def test_run_applications_handles_run_application_exception(monkeypatch, tmp_path):
    mock_logger = mock.Mock()
    mock_db_config = {"db": "config"}
    mock_args_dict = {"configuration_file": "dummy_config.yml", "steps": None}

    mock_configurations = [
        {"application": "app1", "run_application": True, "configuration": {"key": "value1"}}
    ]
    log_file_path = tmp_path / "simtools.log"

    monkeypatch.setattr(
        "simtools.runners.simtools_runner._read_application_configuration",
        mock.Mock(return_value=(mock_configurations, log_file_path)),
    )
    monkeypatch.setattr(
        "simtools.dependencies.get_version_string",
        mock.Mock(return_value="simtools version: 1.2.3\n"),
    )

    def mock_run_application(app, config, logger):
        raise subprocess.CalledProcessError(returncode=1, cmd=app, stderr="fail")

    monkeypatch.setattr("simtools.runners.simtools_runner.run_application", mock_run_application)

    with pytest.raises(subprocess.CalledProcessError):
        simtools_runner.run_applications(mock_args_dict, mock_db_config, mock_logger)
