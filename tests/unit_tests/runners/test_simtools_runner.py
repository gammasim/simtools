#!/usr/bin/python3

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from unittest import mock

import jsonschema
import pytest
import yaml

from simtools.job_execution.execution import JobExecutionError
from simtools.runners import simtools_runner
from simtools.runners.simtools_runner import _find_collection_files

TEST_OUTPUT_PATH = Path("output/test")
SIMULATION_MODELS_PATH = "simulation_models_path"
SIMULATION_MODELS_GIT_PATH = "simulation_models_git_path"
SIMULATION_MODELS_GIT_REVISION = "simulation_models_git_revision"
BASELINE_MODELS_PATH = "/baseline/simulation-models"
TELESCOPE_THROUGHPUT_APPLICATION = "simtools-validate-telescope-throughput"


def _runner_args(config_file="dummy_config.yml", steps=None, ignore_runtime_environment=False):
    return {
        "config_file": config_file,
        "steps": steps,
        "ignore_runtime_environment": ignore_runtime_environment,
    }


def _patch_run_dependencies(
    monkeypatch, mock_configurations, log_file_path, collection_config=None, submit_mock=None
):
    monkeypatch.setattr(
        "simtools.runners.simtools_runner._read_application_configuration",
        mock.Mock(
            return_value=(
                mock_configurations,
                None,
                log_file_path,
                "wf-activity-id",
                collection_config,
            )
        ),
    )
    if collection_config is not None:
        monkeypatch.setattr(
            "simtools.io.ascii_handler.collect_data_from_file",
            mock.Mock(return_value={"collection": collection_config}),
        )
    monkeypatch.setattr(
        "simtools.dependencies.get_version_string",
        mock.Mock(return_value="simtools version: 1.2.3\n"),
    )
    monkeypatch.setattr(
        "simtools.job_execution.execution.run_command",
        submit_mock or mock.Mock(return_value=mock.Mock(stdout="ok", stderr="")),
    )
    monkeypatch.setattr(
        "simtools.runners.simtools_runner.workflow_metadata.build_workflow_activity_metadata",
        mock.Mock(return_value={"id": "wf-activity-id"}),
    )
    monkeypatch.setattr(
        "simtools.runners.simtools_runner.workflow_metadata.update_model_parameter_metadata_file",
        mock.Mock(),
    )


def test_collection_preserves_pair_paths_and_checksums(tmp_test_directory):
    source = Path(tmp_test_directory) / "source"
    destination = Path(tmp_test_directory) / "collected"
    for pair in ("pair-a", "pair-b"):
        path = source / pair / "statistics.json"
        path.parent.mkdir(parents=True)
        path.write_text(pair, encoding="utf-8")
    simtools_runner._copy_collection_files(
        [{"configuration": {"output_path": str(source)}}],
        {
            "output_path": str(destination),
            "files": ["**/statistics.json"],
            "preserve_relative_paths": True,
            "write_inventory": True,
        },
    )
    inventory = json.loads((destination / "inventory.json").read_text())
    assert {item["destination"] for item in inventory} == {
        "pair-a/statistics.json",
        "pair-b/statistics.json",
    }
    for item in inventory:
        content = (destination / item["destination"]).read_bytes()
        assert item["sha256"] == hashlib.sha256(content).hexdigest()
        assert not Path(item["source"]).is_absolute()


def test_prepare_workflow_preserves_empty_runtime_override(mocker):
    mocker.patch.object(
        simtools_runner,
        "_read_application_configuration",
        return_value=([], {"image": "container-image"}, Path("workflow.log"), "activity", None),
    )
    _, runtime, *_ = simtools_runner.prepare_workflow(
        {"config_file": "workflow.yml", "runtime_environment": {}}
    )
    assert runtime == {}


def test_prepare_workflow_resolves_nested_profile_without_writes(tmp_test_directory):
    root = Path(tmp_test_directory)
    workflow = root / "workflow.yml"
    profile = root / "backend.yml"
    profile.write_text("container_image: __CANDIDATE__/image.sif\n", encoding="utf-8")
    workflow.write_text(
        yaml.safe_dump(
            {
                "applications": [
                    {
                        "application": "simtools-simulate-prod",
                        "configuration": {
                            "output_path": "__CANDIDATE__/output",
                            "backend_config": "backend.yml",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    candidate = root / "candidate"
    configurations, *_ = simtools_runner.prepare_workflow(
        {"config_file": str(workflow)}, replacements={"__CANDIDATE__": str(candidate)}
    )
    assert configurations[0]["configuration"]["backend_config"] == {
        "container_image": str(candidate / "image.sif"),
    }
    assert not candidate.exists()
    with pytest.raises(ValueError, match="Unresolved workflow placeholder"):
        simtools_runner.prepare_workflow({"config_file": str(workflow)})


def test_set_input_output_directories():
    path = "input/LSTN-01/pm_photoelectron_spectrum/20250304T073000/config.yml"
    output_path, setting_workflow = simtools_runner._set_input_output_directories(path)
    assert str(output_path) == "simtools-output/LSTN-01/pm_photoelectron_spectrum/20250304T073000"
    assert setting_workflow == "LSTN-01/pm_photoelectron_spectrum/20250304T073000"

    path = "tests/config_files/application_config/config.yml"
    output_path, setting_workflow = simtools_runner._set_input_output_directories(path)
    assert str(output_path) == "simtools-output/tests/config_files/application_config"
    assert setting_workflow == "tests/config_files/application_config"


def test_replace_placeholders_in_configuration():
    config = {
        "input_file": "__SETTING_WORKFLOW__/data.txt",
        "files": ["__SETTING_WORKFLOW__/a.txt", 42, None],
        "other_key": "no_placeholder",
    }
    result = simtools_runner._replace_placeholders_in_configuration(
        config.copy(), TEST_OUTPUT_PATH, "LSTN-01/workflow"
    )
    assert result["input_file"] == "LSTN-01/workflow/data.txt"
    assert result["files"] == ["LSTN-01/workflow/a.txt", 42, None]
    assert result["other_key"] == "no_placeholder"
    assert result["output_path"] == str(TEST_OUTPUT_PATH)


def test_prepare_application_configuration_resolves_by_version(tmp_test_directory):
    config = {
        "model_version": "6.0.2",
        "array_layout_name": {"by_version": {"<7.0.0": "alpha", ">=7.0.0": "CTAO-North-Alpha"}},
        "output_path": "existing-output",
    }
    result = simtools_runner._prepare_application_configuration(
        config.copy(),
        tmp_test_directory / "workflow",
        "tests/config_files/application_config",
    )
    assert result["array_layout_name"] == "alpha"
    assert result["output_path"] == "existing-output"


def test_read_application_configuration_applies_replacements(monkeypatch, tmp_test_directory):
    monkeypatch.setattr(
        simtools_runner.ascii_handler,
        "collect_data_from_file",
        lambda _: {
            "applications": [
                {"application": "app", "configuration": {"output_path": "__TEST_DIRECTORY__/gen"}}
            ],
            "collection": {
                "output_path": "output/__SETTING_WORKFLOW__/collection/",
                "files": ["result.ecsv"],
            },
        },
    )
    configurations, _, _, _, collection = simtools_runner._read_application_configuration(
        "input/LSTN-01/fadc_pulse_shape/019d7d43-84d0-70a3-873e-8d6ab0636895/config.yml",
        steps=None,
        replacements={"__TEST_DIRECTORY__": str(tmp_test_directory)},
    )
    assert configurations[0]["configuration"]["output_path"] == str(tmp_test_directory / "gen")
    assert collection["output_path"] == (
        "output/LSTN-01/fadc_pulse_shape/019d7d43-84d0-70a3-873e-8d6ab0636895/collection/"
    )


def test_read_application_configuration_selected_steps(monkeypatch):
    applications = [
        {"application": f"app{i}", "configuration": {"key": f"value{i}"}} for i in (1, 2, 3)
    ]
    monkeypatch.setattr(
        simtools_runner.ascii_handler,
        "collect_data_from_file",
        lambda _: {"applications": applications},
    )
    configs, _, _, workflow_activity_id, _ = simtools_runner._read_application_configuration(
        "dummy.yml", [2], logging.getLogger()
    )
    assert [c["run_application"] for c in configs] == [False, True, False]
    assert workflow_activity_id is not None
    assert all(c["configuration"]["activity_id"] is not None for c in configs)


def test_read_application_configuration_prefers_path_uuid7(monkeypatch):
    path_uuid = "019d776b-e24c-741d-bc05-e3f6f7ec77c7"
    monkeypatch.setattr(
        simtools_runner.ascii_handler,
        "collect_data_from_file",
        lambda _: {"applications": [{"application": "app1", "configuration": {}}]},
    )
    _, _, _, workflow_activity_id, _ = simtools_runner._read_application_configuration(
        f"input/test/workflow/{path_uuid}/config.yml",
        steps=None,
        workflow_activity_id="generated-by-run-application",
    )
    assert workflow_activity_id == path_uuid


def test_model_source_options_are_applied_per_application():
    source_options = {
        SIMULATION_MODELS_PATH: "/models",
        SIMULATION_MODELS_GIT_PATH: "/models.git",
        SIMULATION_MODELS_GIT_REVISION: "main",
    }
    derive_configuration = {}
    simtools_runner._apply_model_source_options(
        "simtools-derive-photon-electron-spectrum", derive_configuration, source_options
    )
    assert derive_configuration == {}

    submit_configuration = {}
    simtools_runner._apply_model_source_options(
        "simtools-submit-model-parameter-from-external", submit_configuration, source_options
    )
    assert submit_configuration == source_options

    # explicit filesystem source is not overridden by inherited git options
    configuration = {SIMULATION_MODELS_PATH: BASELINE_MODELS_PATH}
    simtools_runner._apply_model_source_options(
        TELESCOPE_THROUGHPUT_APPLICATION,
        configuration,
        {SIMULATION_MODELS_GIT_PATH: "/candidate", SIMULATION_MODELS_GIT_REVISION: "main"},
    )
    assert configuration == {SIMULATION_MODELS_PATH: BASELINE_MODELS_PATH}

    # explicit git source excludes inherited filesystem source
    configuration = {SIMULATION_MODELS_GIT_PATH: "selected-models"}
    simtools_runner._apply_model_source_options(
        TELESCOPE_THROUGHPUT_APPLICATION,
        configuration,
        {SIMULATION_MODELS_PATH: "inherited-models"},
    )
    assert configuration == {SIMULATION_MODELS_GIT_PATH: "selected-models"}


def test_run_applications_runs_and_logs(monkeypatch, tmp_test_directory):
    mock_configurations = [
        {
            "application": f"app{i}",
            "run_application": run,
            "configuration": {
                "key": f"value{i}",
                "activity_id": f"cfg-id-{i}",
                "output_path": str(tmp_test_directory),
            },
        }
        for i, run in ((1, True), (2, False), (3, True))
    ]
    log_file_path = tmp_test_directory / "simtools.log"
    _patch_run_dependencies(monkeypatch, mock_configurations, log_file_path)

    submit_calls = []

    def mock_submit(app, stdout, stderr, configuration=None, runtime_environment=None):
        submit_calls.append({"app": app, "configuration": configuration})
        result_mock = mock.Mock()
        result_mock.output = f"{app}_stdout"
        result_mock.error_output = f"{app}_stderr"
        return result_mock

    monkeypatch.setattr("simtools.job_execution.execution.run_command", mock_submit)
    workflow_build_mock = mock.Mock(return_value={"id": "wf-activity-id"})
    monkeypatch.setattr(
        "simtools.runners.simtools_runner.workflow_metadata.build_workflow_activity_metadata",
        workflow_build_mock,
    )
    simtools_runner.run_applications(_runner_args())

    content = log_file_path.read_text(encoding="utf-8")
    assert "Running simtools applications" in content
    assert "simtools version: 1.2.3" in content
    assert "STDOUT:\napp1_stdout" in content
    assert "STDERR:\napp3_stderr" in content
    assert "Application: app2" not in content  # skipped

    assert [c["app"] for c in submit_calls] == ["app1", "app3"]
    for call, activity in zip(submit_calls, ("cfg-id-1", "cfg-id-3"), strict=True):
        configuration = call["configuration"]
        assert configuration["activity_id"] == activity
        assert configuration["disable_log_file"] is True
        assert configuration["output_path"] == str(tmp_test_directory)
    workflow_build_mock.assert_not_called()


def test_run_applications_propagates_options_to_submitted_configuration(
    monkeypatch, tmp_test_directory
):
    env_file = tmp_test_directory / "production.env"
    submit_mock = mock.Mock(return_value=mock.Mock(stdout="", stderr=""))
    _patch_run_dependencies(
        monkeypatch,
        [
            {
                "application": "app1",
                "run_application": True,
                "configuration": {"env_file": ".env"},
            }
        ],
        tmp_test_directory / "simtools.log",
        submit_mock=submit_mock,
    )
    monkeypatch.setattr("simtools.dependencies.get_version_string", mock.Mock(return_value=""))
    simtools_runner.run_applications(
        {
            "config_file": "workflow.config.yml",
            "env_file": str(env_file),
            "_metadata_configuration_sources": {"env_file": "cli"},
            "steps": None,
            "ignore_runtime_environment": True,
            "ignore_existing_parameter_version": True,
        }
    )
    submitted_configuration = submit_mock.call_args.kwargs["configuration"]
    assert submitted_configuration["env_file"] == str(env_file)
    assert submitted_configuration["ignore_existing_parameter_version"] is True


def test_run_applications_uses_log_file_override(monkeypatch, tmp_test_directory):
    tmp_test_directory = Path(tmp_test_directory)
    default_log = tmp_test_directory / "tmp_application_output" / "simtools.log"
    requested_log = tmp_test_directory / "log_files" / "workflow.log"
    monkeypatch.setattr(
        simtools_runner,
        "_read_application_configuration",
        mock.Mock(return_value=([], None, default_log, "wf-activity-id", None)),
    )
    monkeypatch.setattr(
        simtools_runner.dependencies, "get_version_string", mock.Mock(return_value="")
    )
    simtools_runner.run_applications(
        {
            "config_file": "workflow.config.yml",
            "log_file": str(requested_log),
            "steps": None,
            "ignore_runtime_environment": True,
        }
    )
    assert requested_log.is_file()
    assert not default_log.exists()


def test_run_applications_copies_collection_file_to_destination_name(
    monkeypatch, tmp_test_directory
):
    tmp_path = Path(str(tmp_test_directory))
    source_output = tmp_path / "app_output"
    source_output.mkdir(parents=True, exist_ok=True)
    source_file = source_output / "input-name.ecsv"
    source_file.write_text("test-data", encoding="utf-8")
    source_mtime = 1_000_000_000
    os.utime(source_file, (source_mtime, source_mtime))

    collection_output = tmp_path / "collection"
    _patch_run_dependencies(
        monkeypatch,
        [
            {
                "application": "app1",
                "run_application": True,
                "configuration": {"activity_id": "cfg-id-1", "output_path": str(source_output)},
            }
        ],
        tmp_path / "simtools.log",
        {
            "output_path": str(collection_output),
            "files": [{"source": "input-name.ecsv", "destination": "model.ecsv"}],
        },
    )
    simtools_runner.run_applications(_runner_args())

    copied_file = collection_output / "model.ecsv"
    assert copied_file.read_text(encoding="utf-8") == "test-data"
    assert copied_file.stat().st_mtime > source_mtime


@pytest.mark.parametrize("destination", ["../outside.ecsv", None, "nested/out.ecsv"])
def test_copy_collection_files_rejects_destination_paths(tmp_test_directory, destination):
    source_output = Path(str(tmp_test_directory)) / "app_output"
    source_output.mkdir()
    (source_output / "input.ecsv").write_text("test-data", encoding="utf-8")
    collection_output = Path(str(tmp_test_directory)) / "collection"
    if destination is None:
        destination = str(Path(str(tmp_test_directory)) / "outside.ecsv")

    with pytest.raises(ValueError, match="non-empty filename"):
        simtools_runner._copy_collection_files(
            [{"configuration": {"output_path": str(source_output)}}],
            {
                "output_path": str(collection_output),
                "files": [{"source": "input.ecsv", "destination": destination}],
            },
        )


def test_copy_collection_files_uses_source_directory_list(tmp_test_directory):
    tmp_path = Path(str(tmp_test_directory))
    source_output = tmp_path / "app_output_multi"
    source_output.mkdir()
    grid_output = tmp_path / "grid_output_multi"
    grid_output.mkdir()
    (source_output / "result.dat").write_text("output-data", encoding="utf-8")
    (grid_output / "result.simtel.zst").write_text("grid-data", encoding="utf-8")

    collection_output = tmp_path / "collection_multi"
    configurations = [
        {
            "configuration": {
                "output_path": str(source_output),
                "grid_output_path": str(grid_output),
            }
        }
    ]
    simtools_runner._copy_collection_files(
        configurations,
        {
            "output_path": str(collection_output),
            "source_directory": ["output_path", "grid_output_path"],
            "files": ["result.dat", "result.simtel.zst"],
        },
    )
    assert (collection_output / "result.dat").read_text(encoding="utf-8") == "output-data"
    assert (collection_output / "result.simtel.zst").read_text(encoding="utf-8") == "grid-data"


def test_run_applications_passes_workflow_instrument_context(monkeypatch, tmp_test_directory):
    mock_configurations = [
        {
            "application": "simtools-submit-model-parameter-from-external",
            "run_application": True,
            "configuration": {
                "parameter": "pm_photoelectron_spectrum",
                "parameter_version": "2.0.1",
                "output_path": str(tmp_test_directory / "test_workflow"),
                "site": "North",
                "telescope": "LSTN-design",
                "activity_id": "cfg-id-1",
            },
        },
    ]
    _patch_run_dependencies(monkeypatch, mock_configurations, tmp_test_directory / "simtools.log")
    workflow_build_mock = mock.Mock(return_value={"id": "wf-activity-id"})
    monkeypatch.setattr(
        "simtools.runners.simtools_runner.workflow_metadata.build_workflow_activity_metadata",
        workflow_build_mock,
    )
    workflow_update_mock = mock.Mock()
    monkeypatch.setattr(
        "simtools.runners.simtools_runner.workflow_metadata.update_model_parameter_metadata_file",
        workflow_update_mock,
    )
    simtools_runner.run_applications(_runner_args())

    assert workflow_build_mock.call_args.kwargs["workflow_context"]["site"] == "North"
    assert workflow_build_mock.call_args.kwargs["workflow_context"]["instrument"] == "LSTN-design"
    workflow_update_mock.assert_called_once()


def test_run_applications_propagates_job_execution_error(monkeypatch, tmp_test_directory):
    def mock_submit_failure(app, stdout, stderr, configuration=None, runtime_environment=None):
        raise JobExecutionError("Job failed")

    _patch_run_dependencies(
        monkeypatch,
        [{"application": "app1", "run_application": True, "configuration": {}}],
        tmp_test_directory / "simtools.log",
        submit_mock=mock.Mock(side_effect=mock_submit_failure),
    )
    with pytest.raises(JobExecutionError):
        simtools_runner.run_applications(_runner_args())


def test_read_runtime_environment(monkeypatch):
    common_image = (
        "ghcr.io/gammasim/simtools-prod-sim-telarray-240927-corsika-77550-"
        "bernlohr-1.68-prod6-baseline-qgs2-no_opt:20250715-152108"
    )
    runtime_environment = {
        "image": common_image,
        "network": "simtools-test-network",
        "environment_file": "./.env",
        "container_engine": "podman",
        "options": ["--arch", "amd64"],
    }
    expected_command = [
        "podman",
        "run",
        "--rm",
        "--arch",
        "amd64",
        "--env-file",
        "./.env",
        "--network",
        "simtools-test-network",
        common_image,
    ]
    monkeypatch.setattr(shutil, "which", mock.Mock(return_value=None))
    with pytest.raises(RuntimeError, match=r"Container engine 'podman' not found\."):
        simtools_runner.read_runtime_environment(runtime_environment)

    monkeypatch.setattr(shutil, "which", mock.Mock(return_value="podman"))
    monkeypatch.setattr("simtools.runners.simtools_runner._pull_image", mock.Mock())
    assert simtools_runner.read_runtime_environment(runtime_environment) == expected_command

    minimal = simtools_runner.read_runtime_environment(
        {"image": "ghcr.io/gammasim/simtools-prod-sim-telarray", "container_engine": "docker"}
    )
    assert minimal == ["docker", "run", "--rm", "ghcr.io/gammasim/simtools-prod-sim-telarray"]


def test_read_runtime_environment_apptainer(monkeypatch):
    monkeypatch.setattr(shutil, "which", mock.Mock(return_value="apptainer"))
    result = simtools_runner.read_runtime_environment(
        {
            "image": "simtools.sif",
            "container_engine": "apptainer",
            "environment_file": "runtime.env",
            "options": ["--bind /models:/models", "--containall"],
        }
    )
    assert result == [
        "apptainer",
        "exec",
        "--bind",
        "/models:/models",
        "--containall",
        "--env-file",
        "runtime.env",
        "simtools.sif",
    ]
    with pytest.raises(ValueError, match="do not support the network option"):
        simtools_runner.read_runtime_environment(
            {"image": "simtools.sif", "container_engine": "apptainer", "network": "host"}
        )


def test_prepare_runtime_environment(tmp_test_directory, monkeypatch):
    runtime_file = Path(tmp_test_directory) / "runtime.yml"
    runtime_file.write_text(
        "\n".join(
            [
                "runtime_environment:",
                "  container_engine: podman",
                "  image: test-image",
                "  network: simtools-test-network",
                "  environment_file: .env",
                "  options:",
                '    - "--arch amd64"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        simtools_runner, "read_runtime_environment", lambda config: ["runtime", config["image"]]
    )
    runtime_environment, run_time = simtools_runner.prepare_runtime_environment(runtime_file)
    assert runtime_environment == {
        "container_engine": "podman",
        "image": "test-image",
        "network": "simtools-test-network",
        "environment_file": ".env",
        "options": ["--arch amd64"],
    }
    assert run_time == ["runtime", "test-image"]


@pytest.mark.parametrize(
    ("content", "error"),
    [
        ("- invalid\n", ValueError),
        ("other: value\n", ValueError),
        ("runtime_environment:\n  container_engine: podman\n", jsonschema.ValidationError),
        (
            "runtime_environment:\n  image: test-image\n  options: --arch amd64\n",
            jsonschema.ValidationError,
        ),
        (
            "runtime_environment:\n  image: test-image\n  unknown: value\n",
            jsonschema.ValidationError,
        ),
    ],
)
def test_prepare_runtime_environment_invalid(tmp_test_directory, content, error):
    runtime_file = Path(tmp_test_directory) / "runtime.yml"
    runtime_file.write_text(content, encoding="utf-8")
    with pytest.raises(error):
        simtools_runner.prepare_runtime_environment(runtime_file)


def test_pull_image(monkeypatch):
    image = "ghcr.io/gammasim/simtools-prod:test"
    submit_mock = mock.Mock(return_value=mock.Mock(return_code=0))
    monkeypatch.setattr("simtools.job_execution.execution.run_command", submit_mock)
    simtools_runner._pull_image("podman", image)
    submit_mock.assert_called_once_with(["podman", "image", "inspect", image], check=False)

    submit_mock = mock.Mock(side_effect=[mock.Mock(return_code=125), mock.Mock(return_code=0)])
    monkeypatch.setattr("simtools.job_execution.execution.run_command", submit_mock)
    simtools_runner._pull_image("podman", image)
    submit_mock.assert_any_call(["podman", "pull", image], capture_output=False)

    submit_mock = mock.Mock(
        side_effect=[mock.Mock(return_code=125), JobExecutionError("pull failed")]
    )
    monkeypatch.setattr("simtools.job_execution.execution.run_command", submit_mock)
    with pytest.raises(RuntimeError, match="Failed to pull image"):
        simtools_runner._pull_image("podman", image)


def test_get_model_parameter_metadata_file(tmp_test_directory):
    config = {
        "output_path": "output/test",
        "parameter": "pm_photoelectron_spectrum",
        "parameter_version": "2.0.1",
    }
    assert simtools_runner._get_model_parameter_metadata_file(config) == Path(
        "output/test/pm_photoelectron_spectrum/pm_photoelectron_spectrum-2.0.1.meta.yml"
    )

    tmp_path = Path(str(tmp_test_directory))
    metadata_file = tmp_path / "array_layouts" / "array_layouts-3.0.0.meta.yml"
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    metadata_file.write_text("dummy: value\n", encoding="utf-8")
    config = {"output_path": str(tmp_path), "updated_parameter_version": "3.0.0"}
    assert simtools_runner._get_model_parameter_metadata_file(config) == metadata_file


def test_find_collection_files(tmp_test_directory):
    tmp_path = Path(str(tmp_test_directory))
    with pytest.raises(FileNotFoundError, match=r"result\.ecsv"):
        _find_collection_files("result.ecsv", [tmp_path])

    nested = tmp_path / "model" / "6.0.2"
    nested.mkdir(parents=True)
    (nested / "CTA-alpha-North_test.cfg").write_text("cfg", encoding="utf-8")
    (nested / "energy_MyArray_z20.png").write_text("p1", encoding="utf-8")
    (nested / "energy_MyArray_z52.png").write_text("p2", encoding="utf-8")
    assert _find_collection_files("CTA-alpha-North_test.cfg", [tmp_path]) == [
        nested / "CTA-alpha-North_test.cfg"
    ]
    assert len(_find_collection_files("energy_*.png", [tmp_path])) == 2


def test_find_collection_files_glob_no_match_warns(tmp_test_directory, caplog):
    src = Path(str(tmp_test_directory)) / "src_empty"
    src.mkdir()
    with caplog.at_level(logging.WARNING, logger="simtools.runners.simtools_runner"):
        assert _find_collection_files("nonexistent_*.png", [src]) == []
    assert "nonexistent_*.png" in caplog.text


@pytest.mark.parametrize("overwrite", [False, True])
def test_copy_collection_files_name_collision(tmp_test_directory, overwrite):
    tmp_path = Path(str(tmp_test_directory))
    src_a = tmp_path / "src_a"
    src_b = tmp_path / "src_b"
    src_a.mkdir()
    src_b.mkdir()
    (src_a / "energy_z20.png").write_text("a", encoding="utf-8")
    (src_b / "energy_z20.png").write_text("b", encoding="utf-8")
    collection_output = tmp_path / "coll_collision"
    collection_config = {"output_path": str(collection_output), "files": ["energy_*.png"]}
    configurations = [
        {"configuration": {"output_path": str(src_a)}},
        {"configuration": {"output_path": str(src_b)}},
    ]
    if overwrite:
        simtools_runner._copy_collection_files(
            configurations, collection_config, overwrite_files=True
        )
        assert (collection_output / "energy_z20.png").read_text(encoding="utf-8") == "b"
    else:
        with pytest.raises(FileExistsError, match=r"energy_z20\.png"):
            simtools_runner._copy_collection_files(configurations, collection_config)


def test_copy_collection_files_list_format(tmp_test_directory):
    tmp_path = Path(str(tmp_test_directory))
    src = tmp_path / "app_out"
    src.mkdir()
    (src / "result.ecsv").write_text("data", encoding="utf-8")
    (src / "plot_MyArray.png").write_text("img", encoding="utf-8")
    out_data = tmp_path / "data"
    out_plots = tmp_path / "plots"
    out_skip = tmp_path / "out_skip"
    simtools_runner._copy_collection_files(
        [{"configuration": {"output_path": str(src)}}],
        [
            {"output_path": str(out_data), "files": ["result.ecsv"]},
            {"output_path": str(out_plots), "files": ["plot_*.png"]},
            {"output_path": None, "files": ["result.ecsv"]},
            {"output_path": str(out_skip), "files": []},
        ],
    )
    assert (out_data / "result.ecsv").exists()
    assert (out_plots / "plot_MyArray.png").exists()
    assert not (out_data / "plot_MyArray.png").exists()
    assert not out_skip.exists()


def test_effective_overrides_are_recorded_before_launch(mocker, tmp_test_directory):
    workflow = Path(tmp_test_directory) / "workflow.yml"
    workflow.write_text(
        "schema_name: application_workflow.metaschema\nschema_version: 0.7.0\n"
        "applications:\n- application: simtools-write-trigger-histograms\n"
        '  configuration: {output_path: "' + str(tmp_test_directory) + '"}\n',
        encoding="utf-8",
    )
    provenance = Path(tmp_test_directory) / "effective.yml"
    mocker.patch.object(
        simtools_runner.dependencies, "get_version_string", return_value="versions\n"
    )
    mocker.patch.object(simtools_runner.dependencies, "get_dependency_manifest", return_value={})
    mocker.patch.object(simtools_runner, "_update_workflow_metadata_files")
    mocker.patch.object(
        simtools_runner,
        "_submit_application_and_collect_metadata",
        return_value=mocker.Mock(stdout="", stderr=""),
    )
    simtools_runner.run_applications(
        {
            "config_file": str(workflow),
            "ignore_runtime_environment": True,
            "runtime_environment": {"image": "unused"},
            "provenance_path": str(provenance),
            "log_file": str(Path(tmp_test_directory) / "execution.log"),
            SIMULATION_MODELS_GIT_REVISION: "v1.2.3",
            "env_file": "explicit.env",
            "_metadata_configuration_sources": {"env_file": "cli"},
        }
    )
    payload = yaml.safe_load(provenance.read_text())
    assert payload["runtime_environment"] is None
    assert payload["applications"][0]["configuration"]["env_file"] == "explicit.env"
    assert payload["applications"][0]["configuration"][SIMULATION_MODELS_GIT_REVISION] == "v1.2.3"


def test_resolved_workflow_written_only_with_provenance_path(tmp_test_directory):
    output = Path(tmp_test_directory) / "setting-output"
    simtools_runner._write_resolved_workflow(
        "config.yml",
        [{"configuration": {"output_path": output}}],
        None,
        None,
        provenance_path=False,
    )
    assert not output.exists()

    repository = Path(tmp_test_directory) / "simulation-models.git"
    destination = Path(tmp_test_directory) / "resolved.yml"
    simtools_runner._write_resolved_workflow(
        "workflow.yml",
        [
            {
                "configuration": {
                    "output_path": Path(tmp_test_directory),
                    SIMULATION_MODELS_GIT_PATH: repository,
                }
            }
        ],
        None,
        None,
        provenance_path=destination,
    )
    resolved = yaml.safe_load(destination.read_text(encoding="utf-8"))
    assert resolved["applications"][0]["configuration"][SIMULATION_MODELS_GIT_PATH] == str(
        repository
    )


def test_run_applications_without_host_simulation_executables(mocker, tmp_test_directory):
    log_file = Path(tmp_test_directory) / "execution.log"
    mocker.patch.object(
        simtools_runner, "prepare_workflow", return_value=([], None, log_file, "workflow", None)
    )
    mocker.patch.object(simtools_runner, "_write_resolved_workflow")
    mocker.patch.object(simtools_runner, "_update_workflow_metadata_files")
    mocker.patch.object(
        simtools_runner.dependencies,
        "get_version_string",
        side_effect=[FileNotFoundError("No host simulation software"), "simtools version\n"],
    )
    mocker.patch.object(
        simtools_runner.dependencies, "get_dependency_manifest", return_value={"simtools": "test"}
    )
    simtools_runner.run_applications(_runner_args(ignore_runtime_environment=True))
    content = log_file.read_text(encoding="utf-8")
    assert "simtools version" in content
    assert "Runtime simtools: test" in content


@pytest.mark.parametrize(
    ("application", "accepted"),
    [
        ("simtools-submit-model-parameter-from-external", True),
        ("simtools-plot-camera", False),
        ("external-tool", True),
        ("simtools-not-an-application", True),
    ],
)
def test_ignore_existing_parameter_version_is_passed_only_to_accepting_applications(
    application, accepted
):
    option = simtools_runner._IGNORE_EXISTING_PARAMETER_VERSION
    configurations = [{"application": application, "configuration": {}}]
    simtools_runner.apply_execution_overrides(configurations, {option: True}, {}, None)
    assert (option in configurations[0]["configuration"]) is accepted
