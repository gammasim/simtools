#!/usr/bin/python3

import logging
import shutil
from pathlib import Path
from unittest import mock

import pytest

import simtools.utils.general as gen
from simtools.job_execution.job_manager import JobExecutionError
from simtools.runners import simtools_runner
from simtools.runners.simtools_runner import _find_collection_files

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

    path = "tests/resources_generation/application_config/config.yml"
    output_path, setting_workflow = simtools_runner._set_input_output_directories(path)
    assert str(output_path) == "output/tests/resources_generation/application_config"
    assert setting_workflow == "tests/resources_generation/application_config"


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


def test_replace_placeholders_in_configuration_keeps_existing_output_path(tmp_test_directory):
    config = {
        "output_path": str(tmp_test_directory / "WF"),
        "input_file": "__SETTING_WORKFLOW__/data.txt",
    }
    output_path = TEST_OUTPUT_PATH
    setting_workflow = "LSTN-01/workflow"
    result = simtools_runner._replace_placeholders_in_configuration(
        config.copy(), output_path, setting_workflow
    )
    assert result["output_path"] == str(tmp_test_directory / "WF")
    assert result["input_file"] == "LSTN-01/workflow/data.txt"


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

    configs, _, _, workflow_activity_id = simtools_runner._read_application_configuration(
        DUMMY_CONFIG_FILE, [2], mock_logger
    )
    assert configs[0]["run_application"] is False
    assert configs[1]["run_application"] is True
    assert configs[2]["run_application"] is False
    assert workflow_activity_id is not None
    assert configs[0]["configuration"]["activity_id"] is not None
    assert configs[1]["configuration"]["activity_id"] is not None
    assert configs[2]["configuration"]["activity_id"] is not None


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

    configs, _, log_file, workflow_activity_id = simtools_runner._read_application_configuration(
        DUMMY_CONFIG_FILE, None, mock_logger
    )
    assert configs == []
    assert isinstance(log_file, Path)
    assert workflow_activity_id is not None


def test_read_application_configuration_uses_first_app_output_path_for_log(
    monkeypatch,
    mock_logger,
    mock_collect_data_from_file,
    mock_set_input_output_directories,
    mock_change_dict_keys_case,
):
    """When all apps already have output_path, log goes to the first app's dir."""
    applications = [
        {"application": "app1", "configuration": {"key": "value", "output_path": "my-output"}},
        {"application": "app2", "configuration": {"key": "value2", "output_path": "other-output"}},
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
        lambda config, output_path, setting_workflow: config,
    )

    _, _, log_file, _ = simtools_runner._read_application_configuration(
        DUMMY_CONFIG_FILE, None, mock_logger
    )
    assert log_file == Path("my-output") / "simtools.log"


def test_read_application_configuration_falls_back_to_derived_path_when_first_app_has_no_output(
    monkeypatch,
    mock_logger,
    mock_collect_data_from_file,
    mock_set_input_output_directories,
    mock_change_dict_keys_case,
):
    """When placeholder replacement removes output_path, fall back to derived path."""
    applications = [
        {"application": "app1", "configuration": {"key": "value", "output_path": "my-output"}},
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
    # placeholder replacement strips output_path from the config
    monkeypatch.setattr(
        "simtools.runners.simtools_runner._replace_placeholders_in_configuration",
        lambda config, output_path, setting_workflow: {
            k: v for k, v in config.items() if k != "output_path"
        },
    )

    _, _, log_file, _ = simtools_runner._read_application_configuration(
        DUMMY_CONFIG_FILE, None, mock_logger
    )
    assert log_file == Path("output/test_workflow") / "simtools.log"


def test_run_applications_runs_and_logs(monkeypatch, tmp_test_directory):
    # Prepare mocks
    mock_args_dict = {
        "config_file": "dummy_config.yml",
        "steps": None,
        "ignore_runtime_environment": False,
    }

    # Prepare configurations returned by _read_application_configuration
    mock_configurations = [
        {
            "application": "app1",
            "run_application": True,
            "configuration": {
                "key": "value1",
                "activity_id": "cfg-id-1",
                "output_path": str(tmp_test_directory),
            },
        },
        {
            "application": "app2",
            "run_application": False,
            "configuration": {
                "key": "value2",
                "activity_id": "cfg-id-2",
                "output_path": str(tmp_test_directory),
            },
        },
        {
            "application": "app3",
            "run_application": True,
            "configuration": {
                "key": "value3",
                "activity_id": "cfg-id-3",
                "output_path": str(tmp_test_directory),
            },
        },
    ]
    log_file_path = tmp_test_directory / "simtools.log"

    # Patch _read_application_configuration
    monkeypatch.setattr(
        "simtools.runners.simtools_runner._read_application_configuration",
        mock.Mock(return_value=(mock_configurations, None, log_file_path, "wf-activity-id")),
    )
    workflow_build_mock = mock.Mock(return_value={"id": "wf-activity-id"})
    workflow_update_mock = mock.Mock()
    monkeypatch.setattr(
        "simtools.runners.simtools_runner.workflow_metadata.build_workflow_activity_metadata",
        workflow_build_mock,
    )
    monkeypatch.setattr(
        "simtools.runners.simtools_runner.workflow_metadata.update_model_parameter_metadata_file",
        workflow_update_mock,
    )

    # Patch dependencies.get_version_string
    version_string_mock = mock.Mock(return_value="simtools version: 1.2.3\n")
    monkeypatch.setattr("simtools.dependencies.get_version_string", version_string_mock)

    # Patch job_manager.submit
    submit_calls = []

    def mock_submit(app, out_file, err_file, configuration=None, runtime_environment=None):
        submit_calls.append({"app": app, "configuration": configuration})
        result_mock = mock.Mock()
        result_mock.stdout = f"{app}_stdout"
        result_mock.stderr = f"{app}_stderr"
        return result_mock

    monkeypatch.setattr("simtools.job_execution.job_manager.submit", mock_submit)

    simtools_runner.run_applications(mock_args_dict)

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

    assert len(submit_calls) == 2
    assert submit_calls[0]["configuration"]["activity_id"] == "cfg-id-1"
    assert submit_calls[1]["configuration"]["activity_id"] == "cfg-id-3"
    assert submit_calls[0]["configuration"]["label"] == "app1"
    assert submit_calls[1]["configuration"]["label"] == "app3"
    assert submit_calls[0]["configuration"]["disable_log_file"] is True
    assert submit_calls[1]["configuration"]["disable_log_file"] is True
    assert submit_calls[0]["configuration"]["output_path"] == str(tmp_test_directory)
    assert submit_calls[1]["configuration"]["output_path"] == str(tmp_test_directory)

    version_string_mock.assert_called_once_with([], include_software_versions=False)
    workflow_build_mock.assert_not_called()
    workflow_update_mock.assert_not_called()


def test_run_applications_copies_collection_files(monkeypatch, tmp_test_directory):
    """Copy collection files from application output_path to collection output_path."""
    tmp_path = Path(str(tmp_test_directory))
    source_output = tmp_path / "app_output"
    source_output.mkdir(parents=True, exist_ok=True)
    source_file = source_output / "result.dat"
    source_file.write_text("test-data", encoding="utf-8")

    collection_output = tmp_path / "collection"

    mock_args_dict = {
        "config_file": "dummy_config.yml",
        "steps": None,
        "ignore_runtime_environment": False,
    }
    mock_configurations = [
        {
            "application": "app1",
            "run_application": True,
            "configuration": {
                "activity_id": "cfg-id-1",
                "output_path": str(source_output),
            },
        }
    ]
    log_file_path = tmp_path / "simtools.log"

    monkeypatch.setattr(
        "simtools.runners.simtools_runner._read_application_configuration",
        mock.Mock(return_value=(mock_configurations, None, log_file_path, "wf-activity-id")),
    )
    monkeypatch.setattr(
        "simtools.io.ascii_handler.collect_data_from_file",
        mock.Mock(
            return_value={
                "collection": {
                    "output_path": str(collection_output),
                    "files": ["result.dat"],
                }
            }
        ),
    )
    monkeypatch.setattr(
        "simtools.dependencies.get_version_string",
        mock.Mock(return_value="simtools version: 1.2.3\n"),
    )
    monkeypatch.setattr(
        "simtools.job_execution.job_manager.submit",
        mock.Mock(return_value=mock.Mock(stdout="ok", stderr="")),
    )
    monkeypatch.setattr(
        "simtools.runners.simtools_runner.workflow_metadata.build_workflow_activity_metadata",
        mock.Mock(return_value={"id": "wf-activity-id"}),
    )
    monkeypatch.setattr(
        "simtools.runners.simtools_runner.workflow_metadata.update_model_parameter_metadata_file",
        mock.Mock(),
    )

    simtools_runner.run_applications(mock_args_dict)

    copied_file = collection_output / "result.dat"
    assert copied_file.exists()
    assert copied_file.read_text(encoding="utf-8") == "test-data"


def test_run_applications_passes_workflow_instrument_context(monkeypatch, tmp_test_directory):
    mock_args_dict = {
        "config_file": "dummy_config.yml",
        "steps": None,
        "ignore_runtime_environment": False,
    }
    mock_configurations = [
        {
            "application": "simtools-submit-model-parameter-from-external",
            "run_application": True,
            "configuration": {
                "parameter": "pm_photoelectron_spectrum",
                "parameter_version": "2.0.1",
                "output_path": "output/test_workflow",
                "site": "North",
                "telescope": "LSTN-design",
                "activity_id": "cfg-id-1",
            },
        },
    ]
    log_file_path = tmp_test_directory / "simtools.log"

    monkeypatch.setattr(
        "simtools.runners.simtools_runner._read_application_configuration",
        mock.Mock(return_value=(mock_configurations, None, log_file_path, "wf-activity-id")),
    )
    monkeypatch.setattr(
        "simtools.dependencies.get_version_string",
        mock.Mock(return_value="simtools version: 1.2.3\n"),
    )
    monkeypatch.setattr(
        "simtools.job_execution.job_manager.submit",
        mock.Mock(return_value=mock.Mock(stdout="ok", stderr="")),
    )
    workflow_build_mock = mock.Mock(return_value={"id": "wf-activity-id"})
    workflow_update_mock = mock.Mock()
    monkeypatch.setattr(
        "simtools.runners.simtools_runner.workflow_metadata.build_workflow_activity_metadata",
        workflow_build_mock,
    )
    monkeypatch.setattr(
        "simtools.runners.simtools_runner.workflow_metadata.update_model_parameter_metadata_file",
        workflow_update_mock,
    )

    simtools_runner.run_applications(mock_args_dict)

    assert workflow_build_mock.call_args.kwargs["workflow_context"]["site"] == "North"
    assert workflow_build_mock.call_args.kwargs["workflow_context"]["instrument"] == "LSTN-design"
    workflow_update_mock.assert_called_once()


def test_run_applications_retries_metadata_file_detection_after_submit(
    monkeypatch, tmp_test_directory
):
    """Retry metadata detection after submit when pre-submit detection returns None."""
    mock_args_dict = {
        "config_file": "dummy_config.yml",
        "steps": None,
        "ignore_runtime_environment": False,
    }
    mock_configurations = [
        {
            "application": "simtools-submit-model-parameter-from-external",
            "run_application": True,
            "configuration": {
                "output_path": str(tmp_test_directory),
                "activity_id": "cfg-id-1",
            },
        },
    ]
    log_file_path = tmp_test_directory / "simtools.log"

    monkeypatch.setattr(
        "simtools.runners.simtools_runner._read_application_configuration",
        mock.Mock(return_value=(mock_configurations, None, log_file_path, "wf-activity-id")),
    )
    monkeypatch.setattr(
        "simtools.dependencies.get_version_string",
        mock.Mock(return_value="simtools version: 1.2.3\n"),
    )
    monkeypatch.setattr(
        "simtools.job_execution.job_manager.submit",
        mock.Mock(return_value=mock.Mock(stdout="ok", stderr="")),
    )

    discovered_metadata_file = Path(str(tmp_test_directory)) / "p" / "p-1.0.0.meta.yml"
    get_metadata_mock = mock.Mock(side_effect=[None, discovered_metadata_file])
    monkeypatch.setattr(
        "simtools.runners.simtools_runner._get_model_parameter_metadata_file",
        get_metadata_mock,
    )

    workflow_build_mock = mock.Mock(return_value={"id": "wf-activity-id"})
    workflow_update_mock = mock.Mock()
    monkeypatch.setattr(
        "simtools.runners.simtools_runner.workflow_metadata.build_workflow_activity_metadata",
        workflow_build_mock,
    )
    monkeypatch.setattr(
        "simtools.runners.simtools_runner.workflow_metadata.update_model_parameter_metadata_file",
        workflow_update_mock,
    )

    simtools_runner.run_applications(mock_args_dict)

    assert get_metadata_mock.call_count == 2
    workflow_build_mock.assert_called_once()
    workflow_update_mock.assert_called_once()
    assert workflow_update_mock.call_args.kwargs["metadata_file"] == discovered_metadata_file


def test_run_applications_handles_job_execution_exception(monkeypatch, tmp_test_directory):
    mock_args_dict = {
        "config_file": "dummy_config.yml",
        "steps": None,
        "ignore_runtime_environment": False,
    }

    mock_configurations = [
        {"application": "app1", "run_application": True, "configuration": {"key": "value1"}}
    ]
    log_file_path = tmp_test_directory / "simtools.log"

    monkeypatch.setattr(
        "simtools.runners.simtools_runner._read_application_configuration",
        mock.Mock(return_value=(mock_configurations, None, log_file_path, "wf-activity-id")),
    )
    monkeypatch.setattr(
        "simtools.dependencies.get_version_string",
        mock.Mock(return_value="simtools version: 1.2.3\n"),
    )

    def mock_submit_failure(app, out_file, err_file, configuration=None, runtime_environment=None):
        raise JobExecutionError("Job failed")

    monkeypatch.setattr("simtools.job_execution.job_manager.submit", mock_submit_failure)
    monkeypatch.setattr(
        "simtools.runners.simtools_runner.workflow_metadata.build_workflow_activity_metadata",
        mock.Mock(return_value={"id": "wf-activity-id"}),
    )
    monkeypatch.setattr(
        "simtools.runners.simtools_runner.workflow_metadata.update_model_parameter_metadata_file",
        mock.Mock(),
    )

    with pytest.raises(JobExecutionError):
        simtools_runner.run_applications(mock_args_dict)


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
    monkeypatch.setattr("simtools.runners.simtools_runner._pull_image", mock.Mock())
    result = simtools_runner.read_runtime_environment(runtime_environment, workdir)

    assert result == expected_command


def test_read_runtime_environment_with_minimal_options(monkeypatch):
    runtime_environment = {
        "image": "ghcr.io/gammasim/simtools-prod-sim-telarray",
        "container_engine": "docker",
    }
    monkeypatch.setattr(shutil, "which", mock.Mock(return_value="docker"))
    monkeypatch.setattr("simtools.runners.simtools_runner._pull_image", mock.Mock())
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
    monkeypatch.setattr("simtools.runners.simtools_runner._pull_image", mock.Mock())
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
    mock_args_dict = {
        "config_file": "dummy_config.yml",
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
        mock.Mock(
            return_value=(mock_configurations, runtime_environment, log_file_path, "wf-activity-id")
        ),
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
    monkeypatch.setattr(
        "simtools.runners.simtools_runner.workflow_metadata.build_workflow_activity_metadata",
        mock.Mock(return_value={"id": "wf-activity-id"}),
    )
    monkeypatch.setattr(
        "simtools.runners.simtools_runner.workflow_metadata.update_model_parameter_metadata_file",
        mock.Mock(),
    )

    simtools_runner.run_applications(mock_args_dict)


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
    monkeypatch.setattr("simtools.runners.simtools_runner._pull_image", mock.Mock())

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
    mock_args_dict = {
        "config_file": "empty_config.yml",
        "steps": None,
        "ignore_runtime_environment": False,
    }

    log_file_path = tmp_test_directory / "simtools.log"

    monkeypatch.setattr(
        "simtools.runners.simtools_runner._read_application_configuration",
        mock.Mock(return_value=([], None, log_file_path, "wf-activity-id")),
    )
    monkeypatch.setattr(
        "simtools.dependencies.get_version_string",
        mock.Mock(return_value="simtools version: 1.2.3\n"),
    )

    # Should not call job_manager.submit at all
    mock_submit = mock.Mock()
    monkeypatch.setattr("simtools.job_execution.job_manager.submit", mock_submit)
    monkeypatch.setattr(
        "simtools.runners.simtools_runner.workflow_metadata.build_workflow_activity_metadata",
        mock.Mock(return_value={"id": "wf-activity-id"}),
    )
    monkeypatch.setattr(
        "simtools.runners.simtools_runner.workflow_metadata.update_model_parameter_metadata_file",
        mock.Mock(),
    )

    simtools_runner.run_applications(mock_args_dict)

    # Check log file was created with version info
    with log_file_path.open("r", encoding="utf-8") as f:
        content = f.read()
    assert "Running simtools applications" in content
    assert "simtools version: 1.2.3" in content

    # Verify no applications were submitted
    mock_submit.assert_not_called()


def test_pull_image_skips_pull_if_image_exists(monkeypatch):
    image = "ghcr.io/gammasim/simtools-prod:test"
    inspect_result = mock.Mock(returncode=0)
    submit_mock = mock.Mock(return_value=inspect_result)
    monkeypatch.setattr("simtools.job_execution.job_manager.submit", submit_mock)

    simtools_runner._pull_image("podman", image)

    submit_mock.assert_called_once_with(["podman", "image", "inspect", image], check=False)


def test_pull_image_pulls_if_image_missing(monkeypatch):
    image = "ghcr.io/gammasim/simtools-prod:test"
    inspect_result = mock.Mock(returncode=125)
    pull_result = mock.Mock(returncode=0)
    submit_mock = mock.Mock(side_effect=[inspect_result, pull_result])
    monkeypatch.setattr("simtools.job_execution.job_manager.submit", submit_mock)

    simtools_runner._pull_image("podman", image)

    assert submit_mock.call_count == 2
    submit_mock.assert_any_call(["podman", "image", "inspect", image], check=False)
    submit_mock.assert_any_call(["podman", "pull", image], capture_output=False)


def test_pull_image_raises_if_pull_fails(monkeypatch):
    image = "ghcr.io/gammasim/simtools-prod:test"
    inspect_result = mock.Mock(returncode=125)
    submit_mock = mock.Mock(side_effect=[inspect_result, JobExecutionError("pull failed")])
    monkeypatch.setattr("simtools.job_execution.job_manager.submit", submit_mock)

    with pytest.raises(RuntimeError, match="Failed to pull image"):
        simtools_runner._pull_image("podman", image)


def test_get_application_log_file_no_existing_log_file(tmp_test_directory):
    app_configuration = {"output_path": str(tmp_test_directory)}
    result = simtools_runner._get_application_log_file("simtools-derive-psf", app_configuration, 3)
    assert result == tmp_test_directory / "simtools-derive-psf-03.log"


def test_get_application_log_file_returns_existing_log_file(tmp_test_directory):
    existing = tmp_test_directory / "my_custom.log"
    app_configuration = {"output_path": str(tmp_test_directory), "log_file": existing}
    result = simtools_runner._get_application_log_file("simtools-derive-psf", app_configuration, 1)
    assert result == existing


def test_get_application_log_file_returns_none_without_output_path():
    result = simtools_runner._get_application_log_file("simtools-derive-psf", {}, 1)
    assert result is None


def test_get_model_parameter_metadata_file():
    config = {
        "output_path": "output/test",
        "parameter": "pm_photoelectron_spectrum",
        "parameter_version": "2.0.1",
    }
    metadata_file = simtools_runner._get_model_parameter_metadata_file(config)
    assert metadata_file == Path(
        "output/test/pm_photoelectron_spectrum/pm_photoelectron_spectrum-2.0.1.meta.yml"
    )


def test_get_model_parameter_metadata_file_autodetects_single_metadata_file(tmp_test_directory):
    tmp_path = Path(str(tmp_test_directory))
    metadata_file = tmp_path / "array_layouts" / "array_layouts-3.0.0.meta.yml"
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    metadata_file.write_text("dummy: value\n", encoding="utf-8")

    config = {
        "output_path": str(tmp_path),
        "updated_parameter_version": "3.0.0",
    }
    resolved_file = simtools_runner._get_model_parameter_metadata_file(config)
    assert resolved_file == metadata_file


def test_get_model_parameter_metadata_file_returns_none_for_ambiguous_metadata_files(
    tmp_test_directory,
):
    tmp_path = Path(str(tmp_test_directory))
    file_a = tmp_path / "p1" / "p1-3.0.0.meta.yml"
    file_b = tmp_path / "p2" / "p2-3.0.0.meta.yml"
    file_a.parent.mkdir(parents=True, exist_ok=True)
    file_b.parent.mkdir(parents=True, exist_ok=True)
    file_a.write_text("a: 1\n", encoding="utf-8")
    file_b.write_text("b: 2\n", encoding="utf-8")

    config = {
        "output_path": str(tmp_path),
        "updated_parameter_version": "3.0.0",
    }
    resolved_file = simtools_runner._get_model_parameter_metadata_file(config)
    assert resolved_file is None


def test_get_workflow_configuration_value():
    configurations = [
        {"configuration": {"site": None}},
        {"configuration": {"site": "North"}},
    ]
    assert simtools_runner._get_workflow_configuration_value(configurations, "site") == "North"
    assert simtools_runner._get_workflow_configuration_value(configurations, "instrument") is None


def test_extract_uuid7_from_configuration_path():
    config_file = (
        "input/LSTN-design/pm_photoelectron_spectrum/"
        "019d776b-e24c-741d-bc05-e3f6f7ec77c7/config.yml"
    )
    extracted = gen.extract_uuid7_from_path(config_file)
    assert extracted == "019d776b-e24c-741d-bc05-e3f6f7ec77c7"


def test_read_application_configuration_prefers_path_uuid7(
    monkeypatch,
    mock_logger,
    mock_set_input_output_directories,
    mock_change_dict_keys_case,
):
    path_uuid = "019d776b-e24c-741d-bc05-e3f6f7ec77c7"
    configuration_file = f"input/test/workflow/{path_uuid}/config.yml"

    monkeypatch.setattr(
        "simtools.io.ascii_handler.collect_data_from_file",
        mock.Mock(return_value={"applications": [{"application": "app1", "configuration": {}}]}),
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

    _, _, _, workflow_activity_id = simtools_runner._read_application_configuration(
        configuration_file,
        steps=None,
        workflow_activity_id="generated-by-run-application",
    )

    assert workflow_activity_id == path_uuid


def test_read_application_configuration_ignores_top_level_activity_id(
    monkeypatch,
    mock_set_input_output_directories,
    mock_change_dict_keys_case,
):
    path_uuid = "019d776b-e24c-741d-bc05-e3f6f7ec77c7"
    configuration_file = f"input/test/workflow/{path_uuid}/config.yml"

    monkeypatch.setattr(
        "simtools.io.ascii_handler.collect_data_from_file",
        mock.Mock(
            return_value={
                "activity_id": "workflow-yaml-activity-id",
                "applications": [{"application": "app1", "configuration": {}}],
            }
        ),
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

    _, _, _, workflow_activity_id = simtools_runner._read_application_configuration(
        configuration_file,
        steps=None,
        workflow_activity_id="generated-by-run-application",
    )

    assert workflow_activity_id == path_uuid


def test_find_collection_files_exact_match(tmp_test_directory):
    """Exact filename (no glob) returns the first match and preserves existing semantics."""
    src = Path(str(tmp_test_directory)) / "src"
    src.mkdir()
    (src / "result.ecsv").write_text("data", encoding="utf-8")

    matched = _find_collection_files("result.ecsv", [src])
    assert len(matched) == 1
    assert matched[0] == src / "result.ecsv"


def test_find_collection_files_exact_not_found(tmp_test_directory):
    """Exact filename raises FileNotFoundError when absent."""
    src = Path(str(tmp_test_directory)) / "src_missing"
    src.mkdir()

    with pytest.raises(FileNotFoundError, match=r"result\.ecsv"):
        _find_collection_files("result.ecsv", [src])


def test_find_collection_files_glob_matches_multiple(tmp_test_directory):
    """Glob pattern collects all matching files recursively."""
    src = Path(str(tmp_test_directory)) / "src_glob"
    src.mkdir()
    (src / "energy_MyArray_z20_az0_nsb0.png").write_text("p1", encoding="utf-8")
    (src / "energy_MyArray_z52_az0_nsb0.png").write_text("p2", encoding="utf-8")

    matched = _find_collection_files("energy_*.png", [src])
    assert len(matched) == 2
    names = {f.name for f in matched}
    assert "energy_MyArray_z20_az0_nsb0.png" in names
    assert "energy_MyArray_z52_az0_nsb0.png" in names


def test_find_collection_files_glob_recursive(tmp_test_directory):
    """Glob pattern searches subdirectories recursively."""
    src = Path(str(tmp_test_directory)) / "src_recursive"
    subdir = src / "sub"
    subdir.mkdir(parents=True)
    (subdir / "energy_z20.png").write_text("p", encoding="utf-8")

    matched = _find_collection_files("energy_*.png", [src])
    assert len(matched) == 1
    assert matched[0].name == "energy_z20.png"


def test_find_collection_files_glob_no_match_warns(tmp_test_directory, caplog):
    """Glob pattern with no matches emits a warning rather than raising."""
    src = Path(str(tmp_test_directory)) / "src_empty"
    src.mkdir()

    with caplog.at_level(logging.WARNING, logger="simtools.runners.simtools_runner"):
        matched = _find_collection_files("nonexistent_*.png", [src])
    assert matched == []
    assert "nonexistent_*.png" in caplog.text


def test_run_applications_copies_collection_files_glob(monkeypatch, tmp_test_directory):
    """Glob patterns in collection.files copy all matching files."""
    tmp_path = Path(str(tmp_test_directory))
    source_output = tmp_path / "app_output_glob"
    source_output.mkdir(parents=True, exist_ok=True)
    (source_output / "energy_MyArray_z20_az0_nsb0.png").write_text("img1", encoding="utf-8")
    (source_output / "energy_MyArray_z52_az0_nsb0.png").write_text("img2", encoding="utf-8")

    collection_output = tmp_path / "collection_glob"

    mock_configurations = [
        {
            "application": "app1",
            "run_application": True,
            "configuration": {
                "activity_id": "cfg-id-1",
                "output_path": str(source_output),
            },
        }
    ]
    log_file_path = tmp_path / "simtools.log"

    monkeypatch.setattr(
        "simtools.runners.simtools_runner._read_application_configuration",
        mock.Mock(return_value=(mock_configurations, None, log_file_path, "wf-activity-id")),
    )
    monkeypatch.setattr(
        "simtools.io.ascii_handler.collect_data_from_file",
        mock.Mock(
            return_value={
                "collection": {
                    "output_path": str(collection_output),
                    "files": ["energy_*.png"],
                }
            }
        ),
    )
    monkeypatch.setattr(
        "simtools.dependencies.get_version_string",
        mock.Mock(return_value="simtools version: 1.2.3\n"),
    )
    monkeypatch.setattr(
        "simtools.job_execution.job_manager.submit",
        mock.Mock(return_value=mock.Mock(stdout="ok", stderr="")),
    )
    monkeypatch.setattr(
        "simtools.runners.simtools_runner.workflow_metadata.build_workflow_activity_metadata",
        mock.Mock(return_value={"id": "wf-activity-id"}),
    )
    monkeypatch.setattr(
        "simtools.runners.simtools_runner.workflow_metadata.update_model_parameter_metadata_file",
        mock.Mock(),
    )

    simtools_runner.run_applications(
        {
            "config_file": "dummy_config.yml",
            "steps": None,
            "ignore_runtime_environment": False,
        }
    )

    assert (collection_output / "energy_MyArray_z20_az0_nsb0.png").exists()
    assert (collection_output / "energy_MyArray_z52_az0_nsb0.png").exists()


def test_copy_collection_files_raises_on_name_collision(tmp_test_directory):
    """Raise FileExistsError when two different sources produce the same basename."""
    tmp_path = Path(str(tmp_test_directory))
    src_a = tmp_path / "src_a"
    src_b = tmp_path / "src_b"
    src_a.mkdir()
    src_b.mkdir()
    (src_a / "energy_z20.png").write_text("a", encoding="utf-8")
    (src_b / "energy_z20.png").write_text("b", encoding="utf-8")

    collection_output = tmp_path / "coll_collision"
    collection_config = {
        "output_path": str(collection_output),
        "files": ["energy_*.png"],
    }
    configurations = [
        {"configuration": {"output_path": str(src_a)}},
        {"configuration": {"output_path": str(src_b)}},
    ]
    with pytest.raises(FileExistsError, match=r"energy_z20\.png"):
        simtools_runner._copy_collection_files(configurations, collection_config)


def test_copy_collection_files_list_format(tmp_test_directory):
    """List collection config writes to separate output directories."""
    tmp_path = Path(str(tmp_test_directory))
    src = tmp_path / "app_out"
    src.mkdir()
    (src / "result.ecsv").write_text("data", encoding="utf-8")
    (src / "plot_MyArray.png").write_text("img", encoding="utf-8")

    out_data = tmp_path / "data"
    out_plots = tmp_path / "plots"
    configurations = [{"configuration": {"output_path": str(src)}}]
    collection_config = [
        {"output_path": str(out_data), "files": ["result.ecsv"]},
        {"output_path": str(out_plots), "files": ["plot_*.png"]},
    ]
    simtools_runner._copy_collection_files(configurations, collection_config)

    assert (out_data / "result.ecsv").exists()
    assert (out_plots / "plot_MyArray.png").exists()
    assert not (out_data / "plot_MyArray.png").exists()


def test_copy_collection_files_list_format_skips_empty_entry(tmp_test_directory):
    """List entries with no output_path or files are silently skipped."""
    tmp_path = Path(str(tmp_test_directory))
    src = tmp_path / "app_out2"
    src.mkdir()
    (src / "result.ecsv").write_text("data", encoding="utf-8")

    out = tmp_path / "out_skip"
    configurations = [{"configuration": {"output_path": str(src)}}]
    collection_config = [
        {"output_path": None, "files": ["result.ecsv"]},
        {"output_path": str(out), "files": []},
    ]
    simtools_runner._copy_collection_files(configurations, collection_config)
    assert not out.exists()
