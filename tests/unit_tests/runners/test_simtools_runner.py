#!/usr/bin/python3

import shutil
from pathlib import Path
from unittest import mock

import pytest

from simtools.job_execution.job_manager import JobExecutionError
from simtools.runners import simtools_runner

TEST_OUTPUT_PATH = Path("output/test")
DUMMY_CONFIG_FILE = "dummy.yml"
TEST_WORKDIR = "/workdir/external/"
TEST_FILE1 = "file1.txt"
TEST_FILE2 = "file2.txt"


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
    with pytest.raises(ValueError, match=r"^Could not find subdirectory under"):
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
    assert result["output_path"] == str(output_path)


def test_replace_placeholders_in_configuration_empty_config():
    config = {}
    output_path = TEST_OUTPUT_PATH
    setting_workflow = "WF"
    result = simtools_runner._replace_placeholders_in_configuration(
        config.copy(), output_path, setting_workflow
    )
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
        "simtools.io.ascii_handler.collect_data_from_file",
        mock_collect_data_from_file(applications),
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

    configs, _, _ = simtools_runner._read_application_configuration(
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
        "simtools.io.ascii_handler.collect_data_from_file",
        mock_collect_data_from_file(applications),
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

    configs, _, log_file = simtools_runner._read_application_configuration(
        DUMMY_CONFIG_FILE, None, mock_logger
    )
    assert configs == []
    assert isinstance(log_file, Path)


def test_run_applications_runs_and_logs(monkeypatch, tmp_test_directory):
    # Prepare mocks
    mock_logger = mock.Mock()
    mock_args_dict = {
        "configuration_file": "dummy_config.yml",
        "steps": None,
        "ignore_runtime_environment": False,
    }

    # Prepare configurations returned by _read_application_configuration
    mock_configurations = [
        {"application": "app1", "run_application": True, "configuration": {"key": "value1"}},
        {"application": "app2", "run_application": False, "configuration": {"key": "value2"}},
        {"application": "app3", "run_application": True, "configuration": {"key": "value3"}},
    ]
    log_file_path = tmp_test_directory / "simtools.log"

    # Patch _read_application_configuration
    monkeypatch.setattr(
        "simtools.runners.simtools_runner._read_application_configuration",
        mock.Mock(return_value=(mock_configurations, None, log_file_path)),
    )

    # Patch dependencies.get_version_string
    monkeypatch.setattr(
        "simtools.dependencies.get_version_string",
        mock.Mock(return_value="simtools version: 1.2.3\n"),
    )

    # Patch job_manager.submit
    def mock_submit(app, out_file, err_file, configuration=None, runtime_environment=None):
        result_mock = mock.Mock()
        result_mock.stdout = f"{app}_stdout"
        result_mock.stderr = f"{app}_stderr"
        return result_mock

    monkeypatch.setattr("simtools.job_execution.job_manager.submit", mock_submit)

    simtools_runner.run_applications(mock_args_dict, mock_logger)

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


def test_run_applications_handles_job_execution_exception(monkeypatch, tmp_test_directory):
    mock_logger = mock.Mock()
    mock_args_dict = {
        "configuration_file": "dummy_config.yml",
        "steps": None,
        "ignore_runtime_environment": False,
    }

    mock_configurations = [
        {"application": "app1", "run_application": True, "configuration": {"key": "value1"}}
    ]
    log_file_path = tmp_test_directory / "simtools.log"

    monkeypatch.setattr(
        "simtools.runners.simtools_runner._read_application_configuration",
        mock.Mock(return_value=(mock_configurations, None, log_file_path)),
    )
    monkeypatch.setattr(
        "simtools.dependencies.get_version_string",
        mock.Mock(return_value="simtools version: 1.2.3\n"),
    )

    def mock_submit_failure(app, out_file, err_file, configuration=None, runtime_environment=None):
        raise JobExecutionError("Job failed")

    monkeypatch.setattr("simtools.job_execution.job_manager.submit", mock_submit_failure)

    with pytest.raises(JobExecutionError):
        simtools_runner.run_applications(mock_args_dict, mock_logger)


# Note: _convert_dict_to_args is now handled by job_manager module


def test_read_runtime_environment_with_full_options(monkeypatch):
    common_image = (
        "ghcr.io/gammasim/simtools-prod-sim-telarray-240927-corsika-77550-"
        "bernlohr-1.68-prod6-baseline-qgs2-no_opt:20250715-152108"
    )
    common_network = "simtools-mongo-network"
    common_env_file = "./.env"
    common_container_engine = "podman"
    common_options = ["--arch", "amd64"]

    runtime_environment = {
        "image": common_image,
        "network": common_network,
        "env_file": common_env_file,
        "container_engine": common_container_engine,
        "options": common_options,
    }
    workdir = TEST_WORKDIR
    expected_command = [
        common_container_engine,
        "run",
        "--rm",
        "-v",
        f"{Path.cwd()}:{workdir}",
        "-w",
        workdir,
        *common_options,
        "--env-file",
        common_env_file,
        "--network",
        common_network,
        common_image,
    ]
    monkeypatch.setattr(shutil, "which", mock.Mock(return_value=None))
    with pytest.raises(
        RuntimeError, match=f"Container engine '{common_container_engine}' not found."
    ):
        simtools_runner.read_runtime_environment(runtime_environment, workdir)

    monkeypatch.setattr(shutil, "which", mock.Mock(return_value="podman"))
    result = simtools_runner.read_runtime_environment(runtime_environment, workdir)

    assert result == expected_command


def test_read_runtime_environment_with_minimal_options(monkeypatch):
    runtime_environment = {
        "image": "ghcr.io/gammasim/simtools-prod-sim-telarray",
        "container_engine": "docker",
    }
    monkeypatch.setattr(shutil, "which", mock.Mock(return_value="docker"))
    workdir = TEST_WORKDIR
    expected_command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{Path.cwd()}:{workdir}",
        "-w",
        workdir,
        runtime_environment["image"],
    ]
    result = simtools_runner.read_runtime_environment(runtime_environment, workdir)
    assert result == expected_command


def test_read_runtime_environment_with_no_runtime_environment():
    runtime_environment = None
    workdir = TEST_WORKDIR
    result = simtools_runner.read_runtime_environment(runtime_environment, workdir)
    assert result == []


def test_read_runtime_environment_with_missing_options(monkeypatch):
    runtime_environment = {
        "image": "ghcr.io/gammasim/simtools-prod-sim-telarray",
        "network": "simtools-mongo-network",
        "container_engine": "docker",
    }
    monkeypatch.setattr(shutil, "which", mock.Mock(return_value="docker"))
    workdir = TEST_WORKDIR
    expected_command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{Path.cwd()}:{workdir}",
        "-w",
        workdir,
        "--network",
        runtime_environment["network"],
        runtime_environment["image"],
    ]
    result = simtools_runner.read_runtime_environment(runtime_environment, workdir)
    assert result == expected_command


# Note: run_application function has been replaced by job_manager.submit


def test_run_applications_with_runtime_environment_ignored(monkeypatch, tmp_test_directory):
    """Test that runtime environment is ignored when ignore_runtime_environment is True."""
    mock_logger = mock.Mock()
    mock_args_dict = {
        "configuration_file": "dummy_config.yml",
        "steps": [1],
        "ignore_runtime_environment": True,
    }

    mock_configurations = [
        {"application": "app1", "run_application": True, "configuration": {"key": "value1"}},
    ]
    runtime_environment = {"image": "test-image", "container_engine": "docker"}
    log_file_path = tmp_test_directory / "simtools.log"

    monkeypatch.setattr(
        "simtools.runners.simtools_runner._read_application_configuration",
        mock.Mock(return_value=(mock_configurations, runtime_environment, log_file_path)),
    )
    monkeypatch.setattr(
        "simtools.dependencies.get_version_string",
        mock.Mock(return_value="simtools version: 1.2.3\n"),
    )

    # Mock job_manager.submit to verify no runtime_environment is passed
    def mock_submit(app, out_file, err_file, configuration=None, runtime_environment=None):
        assert runtime_environment == []  # Should be empty list when ignored
        result_mock = mock.Mock()
        result_mock.stdout = f"{app}_stdout"
        result_mock.stderr = f"{app}_stderr"
        return result_mock

    monkeypatch.setattr("simtools.job_execution.job_manager.submit", mock_submit)

    simtools_runner.run_applications(mock_args_dict, mock_logger)


def test_read_runtime_environment_error_handling(monkeypatch):
    """Test error handling in read_runtime_environment."""
    # Test with container engine not found
    runtime_environment = {"image": "test-image", "container_engine": "nonexistent"}

    # Mock shutil.which to return None (engine not found)
    monkeypatch.setattr(shutil, "which", mock.Mock(return_value=None))

    with pytest.raises(RuntimeError, match="Container engine 'nonexistent' not found"):
        simtools_runner.read_runtime_environment(runtime_environment)


def test_read_runtime_environment_with_env_file_and_options(monkeypatch):
    """Test read_runtime_environment with env_file and various options."""
    runtime_environment = {
        "image": "test-image",
        "container_engine": "podman",
        "env_file": ".env",
        "options": ["--privileged", "--user", "root"],
    }

    monkeypatch.setattr(shutil, "which", mock.Mock(return_value="podman"))

    result = simtools_runner.read_runtime_environment(runtime_environment)

    expected = [
        "podman",
        "run",
        "--rm",
        "-v",
        f"{Path.cwd()}:/workdir/external/",
        "-w",
        "/workdir/external/",
        "--privileged",
        "--user",
        "root",
        "--env-file",
        ".env",
        "test-image",
    ]
    assert result == expected


def test_run_applications_with_empty_configuration_list(monkeypatch, tmp_test_directory):
    """Test run_applications with empty configuration list."""
    mock_logger = mock.Mock()
    mock_args_dict = {
        "configuration_file": "empty_config.yml",
        "steps": None,
        "ignore_runtime_environment": False,
    }

    log_file_path = tmp_test_directory / "simtools.log"

    monkeypatch.setattr(
        "simtools.runners.simtools_runner._read_application_configuration",
        mock.Mock(return_value=([], None, log_file_path)),
    )
    monkeypatch.setattr(
        "simtools.dependencies.get_version_string",
        mock.Mock(return_value="simtools version: 1.2.3\n"),
    )

    # Should not call job_manager.submit at all
    mock_submit = mock.Mock()
    monkeypatch.setattr("simtools.job_execution.job_manager.submit", mock_submit)

    simtools_runner.run_applications(mock_args_dict, mock_logger)

    # Check log file was created with version info
    with log_file_path.open("r", encoding="utf-8") as f:
        content = f.read()
    assert "Running simtools applications" in content
    assert "simtools version: 1.2.3" in content

    # Verify no applications were submitted
    mock_submit.assert_not_called()
