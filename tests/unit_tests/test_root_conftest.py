"""Tests for the repository-level pytest configuration."""

import importlib.util
from pathlib import Path

ROOT_CONFTEST = Path(__file__).parents[1] / "conftest.py"
SPEC = importlib.util.spec_from_file_location("simtools_root_conftest", ROOT_CONFTEST)
ROOT_CONFIGURATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ROOT_CONFIGURATION)


def test_missing_dotenv_path_ignores_runtime_created_output_path(tmp_path, monkeypatch):
    output_path = tmp_path / "not-created-yet"
    monkeypatch.delenv("SIMTOOLS_OUTPUT_PATH", raising=False)

    assert ROOT_CONFIGURATION._missing_dotenv_path("SIMTOOLS_OUTPUT_PATH", str(output_path)) is None


def test_missing_dotenv_path_ignores_unrecognized_path_setting(tmp_path):
    cache_path = tmp_path / "not-created-yet"

    assert ROOT_CONFIGURATION._missing_dotenv_path("SIMTOOLS_CACHE_PATH", str(cache_path)) is None


def test_missing_dotenv_path_reports_interaction_tables_path(tmp_path):
    input_path = tmp_path / "missing-input"

    assert (
        ROOT_CONFIGURATION._missing_dotenv_path(
            "SIMTOOLS_CORSIKA_INTERACTION_TABLE_PATH", str(input_path)
        )
        == f"  SIMTOOLS_CORSIKA_INTERACTION_TABLE_PATH: {input_path}"
    )


def test_missing_dotenv_path_reports_simtools_tests_path(tmp_path):
    input_path = tmp_path / "missing-input"

    assert ROOT_CONFIGURATION._missing_dotenv_path("SIMTOOLS_TESTS_PATH", str(input_path)) == (
        f"  SIMTOOLS_TESTS_PATH: {input_path}"
    )


def test_missing_dotenv_path_ignores_other_dependency_path(tmp_path):
    input_path = tmp_path / "not-validated"

    assert (
        ROOT_CONFIGURATION._missing_dotenv_path("SIMTOOLS_SIM_TELARRAY_PATH", str(input_path))
        is None
    )
