"""Production validation uses the target image's catalog, not the host's."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from simtools.job_execution.job import ExecutionOptions, JobSpec
from simtools.model_repository import compatibility as models
from simtools.production_configuration import compatibility as production

TARGET_VERSION = "0.37.0"
DEFAULT_VERSION = "v0.17.1"
VALIDATE = "_validate"
INCOMPATIBLE = "incompatible models"
PYTHON = "python"

CATALOG = {models.LEGACY_MODEL_KEY: {"default-tag": DEFAULT_VERSION}}
TARGET = {production.TARGET_KEY: TARGET_VERSION, production.CATALOG_KEY: CATALOG}
MODELS_PATH = "simulation_models_path"
IMAGE = "old.sif"


def test_image_contract(monkeypatch, tmp_test_directory):
    run = Mock(return_value=SimpleNamespace(stdout=json.dumps(TARGET)))
    monkeypatch.setattr(production.subprocess, "run", run)
    validate = Mock(return_value={models.STATUS_KEY: "compatible"})
    monkeypatch.setattr(production, VALIDATE, validate)
    args = {MODELS_PATH: str(tmp_test_directory)}
    assert production._preview(args, IMAGE)[production.TARGET_KEY] == TARGET_VERSION
    validate.assert_called_once_with(str(tmp_test_directory), CATALOG, None, False)
    assert run.call_args.args[0][:4] == ["apptainer", "exec", "--cleanenv", IMAGE]


def test_local_git_source_uses_catalog_default(monkeypatch, tmp_test_directory):
    catalog = Path(tmp_test_directory) / "dependency_versions.yml"
    catalog.write_text("unused")
    monkeypatch.setattr(production, "_bundled_catalog_path", lambda: catalog)
    monkeypatch.setattr(
        production.dependency_versions, "load_dependency_catalog", lambda _: CATALOG
    )
    validate = Mock(return_value={})
    monkeypatch.setattr(production, VALIDATE, validate)
    production._preview({production.GIT_PATH_KEY: str(tmp_test_directory)})
    validate.assert_called_once_with(str(tmp_test_directory), CATALOG, DEFAULT_VERSION, False)
    with pytest.raises(ValueError, match="explicit simulation-model source"):
        production._preview({})


def test_submission_checks_before_returning_jobs(monkeypatch):
    preview = Mock()
    monkeypatch.setattr(production, "_preview", preview)
    jobs = [JobSpec("job", 0, command=(PYTHON,))]
    assert production._prepare_jobs(jobs, {}, ExecutionOptions()) is jobs
    preview.assert_called_once_with({}, None)
    preview.side_effect = ValueError(INCOMPATIBLE)
    with pytest.raises(ValueError, match=INCOMPATIBLE):
        production._prepare_jobs(jobs, {}, ExecutionOptions())
    assert production._prepare_jobs([], {}, ExecutionOptions()) == []


def test_mixed_images_rejected():
    jobs = [JobSpec("job", 0, command=(PYTHON,), resources={"container_image": IMAGE})]
    with pytest.raises(ValueError, match="different target images"):
        production._prepare_jobs(jobs, {}, ExecutionOptions())


@pytest.mark.parametrize("installed", [False, True])
def test_bundled_catalog_ignores_working_directory(monkeypatch, tmp_test_directory, installed):
    root = Path(tmp_test_directory)
    module = root / "source" / "src" / "simtools" / "dependency_versions.py"
    monkeypatch.setattr(production.dependency_versions, "__file__", str(module))
    monkeypatch.setattr(production.sys, "prefix", str(root / "env"))
    expected = root / "source" / production.dependency_versions.DEPENDENCY_VERSIONS_FILENAME
    if installed:
        expected = root / "env" / "simtools" / expected.name
    expected.parent.mkdir(parents=True)
    expected.write_text("catalog")
    assert production._bundled_catalog_path() == expected
    expected.unlink()
    with pytest.raises(RuntimeError, match="no bundled dependency catalog"):
        production._bundled_catalog_path()
