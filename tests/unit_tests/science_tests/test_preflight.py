import subprocess
from pathlib import Path

import pytest

from simtools.science_tests import preflight

_CONTAINER_IMAGE_CONTEXT_KEY = "__SCIENCE_CONTAINER_IMAGE_PATH__"


def test_validate_context_paths_accepts_nonempty_directories(tmp_test_directory):
    root = Path(tmp_test_directory)
    table_path = root / "tables"
    table_path.mkdir()
    (table_path / "table.dat").write_text("table", encoding="utf-8")

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


def test_validate_context_paths_reports_empty_directories_and_image(tmp_test_directory):
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
                _CONTAINER_IMAGE_CONTEXT_KEY: str(image),
            }
        )

    message = str(error.value)
    assert "simulation_models_path directory contains no non-empty files" in message
    assert f"{_CONTAINER_IMAGE_CONTEXT_KEY} is not a non-empty file" in message


def test_canary_config_ignores_null_context_values(tmp_test_directory):
    root = Path(tmp_test_directory)
    workflow = root / "workflow.yml"
    workflow.write_text(
        """
applications:
  - application: simtools-simulate-prod
    configuration:
      backend_config:
        container_image: __SCIENCE_CONTAINER_IMAGE_PATH__
        environment_file: __CONFIG_DIRECTORY__/environment
""",
        encoding="utf-8",
    )

    config = preflight._canary_config(
        root,
        "workflow.yml",
        {
            _CONTAINER_IMAGE_CONTEXT_KEY: "/image.sif",
            "optional_path": None,
        },
    )

    assert config["container_image"] == "/image.sif"


def test_canary_config_requires_inline_backend_config(tmp_test_directory):
    root = Path(tmp_test_directory)
    workflow = root / "workflow.yml"
    workflow.write_text("applications:\n  - configuration: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="requires inline backend_config"):
        preflight._canary_config(root, "workflow.yml", {})


def test_worker_canary_uses_high_priority(tmp_test_directory, monkeypatch):
    root = Path(tmp_test_directory)
    workflow = root / "workflow.yml"
    workflow.write_text(
        """
applications:
  - application: simtools-simulate-prod
    configuration:
      backend_config:
        container_image: __SCIENCE_CONTAINER_IMAGE_PATH__
""",
        encoding="utf-8",
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
        "workflow.yml",
        {_CONTAINER_IMAGE_CONTEXT_KEY: "/image.sif"},
        root / "work",
    )

    assert captured["options"].backend_config["priority"] == preflight.PREFLIGHT_PRIORITY
    assert "PREFLIGHT_FAILED" in captured["jobs"][0].command[2]
    subprocess.run(["sh", "-n"], input=captured["jobs"][0].command[2], text=True, check=True)


def test_worker_canary_requires_container_image(tmp_test_directory):
    root = Path(tmp_test_directory)
    (root / "workflow.yml").write_text(
        "applications:\n  - configuration:\n      backend_config: {}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="has no container image"):
        preflight.run_worker_canary(root, "workflow.yml", {}, root / "work")


def test_worker_canary_checks_context_paths_and_waits_for_submission(
    tmp_test_directory, monkeypatch
):
    root = Path(tmp_test_directory)
    (root / "workflow.yml").write_text(
        "applications:\n  - configuration:\n      backend_config:\n"
        "        container_image: /image.sif\n",
        encoding="utf-8",
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
        "workflow.yml",
        {"__SCIENCE_CANDIDATE_PATH__": "/candidate dir"},
        root / "work",
    )

    command = captured["jobs"][0].command[2]
    assert "check_path __SCIENCE_CANDIDATE_PATH__ '/candidate dir'" in command
    assert captured["submission"] is submission
    assert captured["waited_options"] is captured["options"]
