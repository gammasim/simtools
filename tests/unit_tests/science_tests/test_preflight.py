import subprocess
from pathlib import Path

import pytest

from simtools.science_tests import preflight

_CONTAINER_IMAGE_CONTEXT_KEY = "__SCIENCE_CONTAINER_IMAGE_PATH__"
_ANALYSIS_IMAGE_CONTEXT_KEY = "__SCIENCE_ANALYSIS_IMAGE_PATH__"
_WORKFLOW_FILE_NAME = "workflow.yml"
_CONTAINER_IMAGE_PATH = "/image.sif"
_BACKEND_FILE_NAME = "backend.yml"
_CONTAINER_IMAGE_KEY = "container_image"
_ENVIRONMENT_FILE_KEY = "environment_file"
_ENCODING = "utf-8"


def test_validate_context_paths_accepts_nonempty_directories(tmp_test_directory):
    root = Path(tmp_test_directory)
    table_path = root / "tables"
    table_path.mkdir()
    (table_path / "table.dat").write_text("table", encoding=_ENCODING)

    preflight.validate_context_paths(
        {
            "__CORSIKA_INTERACTION_TABLE_PATH__": str(table_path),
            _CONTAINER_IMAGE_CONTEXT_KEY: str(table_path / "table.dat"),
        }
    )


def test_validate_context_paths_reports_missing_paths(tmp_test_directory):
    with pytest.raises(ValueError, match="CORSIKA_INTERACTION_TABLE_PATH"):
        preflight.validate_context_paths(
            {"__CORSIKA_INTERACTION_TABLE_PATH__": str(Path(tmp_test_directory) / "missing")}
        )


def test_validate_context_paths_reports_empty_directories_and_analysis_image(tmp_test_directory):
    root = Path(tmp_test_directory)
    empty = root / "empty"
    empty.mkdir()
    image = root / "empty.sif"
    image.touch()

    with pytest.raises(
        ValueError, match="simulation_models_path directory contains no non-empty files"
    ) as error:
        preflight.validate_context_paths(
            {
                "simulation_models_path": str(empty),
                _ANALYSIS_IMAGE_CONTEXT_KEY: str(image),
            }
        )

    message = str(error.value)
    assert "simulation_models_path directory contains no non-empty files" in message
    assert f"{_ANALYSIS_IMAGE_CONTEXT_KEY} is not a non-empty file" in message


def test_canary_config_ignores_null_context_values(tmp_test_directory):
    root = Path(tmp_test_directory)
    workflow = root / _WORKFLOW_FILE_NAME
    workflow.write_text(
        """
applications:
  - application: simtools-simulate-prod
    configuration:
      backend_config:
        container_image: __SCIENCE_CONTAINER_IMAGE_PATH__
        environment_file: __CONFIG_DIRECTORY__/environment
""",
        encoding=_ENCODING,
    )

    config = preflight._canary_config(
        root,
        _WORKFLOW_FILE_NAME,
        {
            _CONTAINER_IMAGE_CONTEXT_KEY: _CONTAINER_IMAGE_PATH,
            "optional_path": None,
        },
    )

    assert config[_CONTAINER_IMAGE_KEY] == _CONTAINER_IMAGE_PATH


@pytest.mark.parametrize("reference", [_BACKEND_FILE_NAME, "__CONFIG_DIRECTORY__/backend.yml"])
def test_canary_config_resolves_shared_backend_profile(tmp_test_directory, reference):
    root = Path(tmp_test_directory)
    profile = root / _BACKEND_FILE_NAME
    profile.write_text(
        "container_image: __SCIENCE_CONTAINER_IMAGE_PATH__\n"
        "environment_file: __CONFIG_DIRECTORY__/environment\n",
        encoding=_ENCODING,
    )
    workflow = root / _WORKFLOW_FILE_NAME
    workflow.write_text(
        f"applications:\n  - configuration:\n      backend_config: {reference}\n",
        encoding=_ENCODING,
    )
    config = preflight._canary_config(
        root, workflow.name, {_CONTAINER_IMAGE_CONTEXT_KEY: _CONTAINER_IMAGE_PATH}
    )
    assert config == {
        _CONTAINER_IMAGE_KEY: _CONTAINER_IMAGE_PATH,
        _ENVIRONMENT_FILE_KEY: str(root / "environment"),
    }
    profile.write_text("- invalid\n", encoding=_ENCODING)
    with pytest.raises(ValueError, match="Backend configuration must contain a mapping"):
        preflight._canary_config(root, workflow.name, {})
    profile.unlink()
    with pytest.raises(FileNotFoundError, match="Backend configuration not found"):
        preflight._canary_config(root, workflow.name, {})


def test_canary_config_requires_backend_config(tmp_test_directory):
    root = Path(tmp_test_directory)
    workflow = root / _WORKFLOW_FILE_NAME
    workflow.write_text("applications:\n  - configuration: {}\n", encoding=_ENCODING)
    with pytest.raises(ValueError, match="requires backend_config"):
        preflight._canary_config(root, _WORKFLOW_FILE_NAME, {})


def test_worker_canary_uses_high_priority(tmp_test_directory, monkeypatch):
    root = Path(tmp_test_directory)
    workflow = root / _WORKFLOW_FILE_NAME
    workflow.write_text(
        """
applications:
  - application: simtools-simulate-prod
    configuration:
      backend_config:
        container_image: __SCIENCE_CONTAINER_IMAGE_PATH__
""",
        encoding=_ENCODING,
    )
    captured = {}
    monkeypatch.setattr(
        preflight,
        "submit_jobs",
        lambda jobs, options: captured.update(jobs=jobs, options=options) or object(),
    )
    monkeypatch.setattr(preflight, "wait_for_submission", lambda *args, **kwargs: None)

    preflight.run_worker_canary(
        root,
        _WORKFLOW_FILE_NAME,
        {_CONTAINER_IMAGE_CONTEXT_KEY: _CONTAINER_IMAGE_PATH},
        root / "work",
    )

    assert captured["options"].backend_config["priority"] == preflight.PREFLIGHT_PRIORITY
    assert "PREFLIGHT_FAILED" in captured["jobs"][0].command[2]
    subprocess.run(["sh", "-n"], input=captured["jobs"][0].command[2], text=True, check=True)


def test_worker_canary_requires_container_image(tmp_test_directory):
    root = Path(tmp_test_directory)
    (root / _WORKFLOW_FILE_NAME).write_text(
        "applications:\n  - configuration:\n      backend_config: {}\n",
        encoding=_ENCODING,
    )

    with pytest.raises(ValueError, match="has no container image"):
        preflight.run_worker_canary(root, _WORKFLOW_FILE_NAME, {}, root / "work")


def test_worker_canary_checks_context_paths_and_waits_for_submission(
    tmp_test_directory, monkeypatch
):
    root = Path(tmp_test_directory)
    (root / _WORKFLOW_FILE_NAME).write_text(
        "applications:\n  - configuration:\n      backend_config:\n"
        "        container_image: /image.sif\n",
        encoding=_ENCODING,
    )
    captured = {}
    submission = object()
    monkeypatch.setattr(
        preflight,
        "submit_jobs",
        lambda jobs, options: captured.update(jobs=jobs, options=options) or submission,
    )
    monkeypatch.setattr(
        preflight,
        "wait_for_submission",
        lambda received, options: captured.update(submission=received, waited_options=options),
    )

    preflight.run_worker_canary(
        root,
        _WORKFLOW_FILE_NAME,
        {"__SCIENCE_CANDIDATE_PATH__": "/candidate dir"},
        root / "work",
    )

    command = captured["jobs"][0].command[2]
    assert "check_path __SCIENCE_CANDIDATE_PATH__ '/candidate dir'" in command
    assert captured["submission"] is submission
    assert captured["waited_options"] is captured["options"]
