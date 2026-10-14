import subprocess
from pathlib import Path

import pytest

from simtools.science_tests import preflight


def test_validate_context_paths_accepts_nonempty_directories(tmp_test_directory):
    root = Path(tmp_test_directory)
    table_path = root / "tables"
    table_path.mkdir()
    (table_path / "table.dat").write_text("table", encoding="utf-8")

    preflight.validate_context_paths(
        {
            "__CORSIKA_INTERACTION_TABLE_PATH__": str(table_path),
            "__SCIENCE_CONTAINER_IMAGE_PATH__": str(table_path / "table.dat"),
        }
    )


def test_validate_context_paths_reports_missing_paths(tmp_test_directory):
    with pytest.raises(ValueError, match="CORSIKA_INTERACTION_TABLE_PATH"):
        preflight.validate_context_paths(
            {"__CORSIKA_INTERACTION_TABLE_PATH__": str(Path(tmp_test_directory) / "missing")}
        )


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
            "__SCIENCE_CONTAINER_IMAGE_PATH__": "/image.sif",
            "optional_path": None,
        },
    )

    assert config["container_image"] == "/image.sif"


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
        {"__SCIENCE_CONTAINER_IMAGE_PATH__": "/image.sif"},
        root / "work",
    )

    assert captured["options"].backend_config["priority"] == preflight.PREFLIGHT_PRIORITY
    assert "PREFLIGHT_FAILED" in captured["jobs"][0].command[2]
    subprocess.run(["sh", "-n"], input=captured["jobs"][0].command[2], text=True, check=True)
