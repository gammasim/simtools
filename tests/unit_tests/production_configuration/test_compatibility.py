"""Production compatibility gates run before any simulation job is submitted."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from simtools.job_execution.job import ExecutionOptions, JobSpec
from simtools.model_repository import compatibility as models
from simtools.production_configuration import compatibility as production

MODELS_NAME = "models"
CHANGED = "changed"
PREVIEW_NAME = "_preview"
EXECUTE_NAME = "execute_jobs"
OUTPUT_KEY = "output_path"
IDENTITY_NAME = "_identity"
LEGACY_VERSION = "0.37.0"
CATALOG_ENV = "SIMTOOLS_DEPENDENCY_VERSIONS"
ENCODING = "utf-8"
FAILURE_MESSAGE = "missing model parameter"
OLD_IMAGE = "old.sif"
CATALOG_KEY = "catalog"

GIT_SELECTION = "explicit Git source"
PACKAGE_NAME = "simtools"

SOURCE = models.SOURCE_KEY
DIGEST = models.DIGEST_KEY
SELECTION = "selection"
FILESYSTEM = "explicit filesystem path"


@pytest.fixture
def proof(tmp_test_directory):
    root = Path(tmp_test_directory) / MODELS_NAME
    data = root / models.MODELS_DIRECTORY
    data.mkdir(parents=True)
    (data / "value.json").write_text("1", encoding=ENCODING)
    return {
        SOURCE: str(root),
        DIGEST: models._tree_digest(root),
        SELECTION: FILESYSTEM,
        production.IMAGE_DIGEST_KEY: None,
        production.INITIALIZATION_KEY: "not checked",
    }


def job(tmp_test_directory, index=0, zenith="20 deg"):
    return JobSpec(
        f"job-{index}",
        index,
        command=(
            sys.executable,
            "-m",
            "simtools.applications.simulate_prod",
            "--output_path",
            str(Path(tmp_test_directory) / f"job-{index}"),
            "--run_number",
            str(index),
            "--zenith_angle",
            zenith,
        ),
    )


def test_worker_seal_detects_model_drift(tmp_test_directory, proof):
    original = JobSpec("test", 0, command=(sys.executable, "-c", "pass"))
    sealed = production._sealed_job(original, proof)
    # Test the transported stdlib guard itself, without requiring simtools on the worker.
    subprocess.run([sys.executable, "-c", production._guard_code(proof)], check=True)
    assert sealed.metadata[production.PROOF_KEY] == proof
    (Path(proof[SOURCE]) / models.MODELS_DIRECTORY / "new").write_text(CHANGED, encoding=ENCODING)
    result = subprocess.run(
        [sys.executable, "-c", production._guard_code(proof)], capture_output=True
    )
    assert result.returncode != 0
    assert b"changed since" in result.stderr


def test_representatives_include_configuration_modes_but_not_runs(tmp_test_directory):
    jobs = [
        job(tmp_test_directory),
        job(tmp_test_directory, 1),
        job(tmp_test_directory, 2, "67.5 deg"),
    ]
    assert list(production._representatives(jobs)) == [jobs[0], jobs[2]]


def test_initialization_failure_does_not_return_production_jobs(
    tmp_test_directory, proof, monkeypatch
):
    monkeypatch.setattr(production, PREVIEW_NAME, lambda *args: proof)

    def fail(*_args):
        raise RuntimeError(FAILURE_MESSAGE)

    monkeypatch.setattr(production, EXECUTE_NAME, fail)
    with pytest.raises(RuntimeError, match="Dependency initialization failed"):
        production._prepare_jobs(
            [job(tmp_test_directory)], {OUTPUT_KEY: str(tmp_test_directory)}, ExecutionOptions()
        )
    record = json.loads((Path(tmp_test_directory) / production.RECORD_NAME).read_text())
    assert record[production.INITIALIZATION_KEY] == "failed"


def test_successful_initialization_seals_jobs(tmp_test_directory, proof, monkeypatch):
    monkeypatch.setattr(production, PREVIEW_NAME, lambda *args: proof)
    captured = []
    monkeypatch.setattr(production, EXECUTE_NAME, lambda jobs, options: captured.extend(jobs))
    args = {OUTPUT_KEY: str(tmp_test_directory), production.PATH_KEY: proof[SOURCE]}
    result = production._prepare_jobs([job(tmp_test_directory)], args, ExecutionOptions())
    assert len(captured) == 1
    assert "Simulator" in captured[0].command[2]
    assert "simulate()" not in captured[0].command[2]
    assert result[0].metadata[production.PROOF_KEY][production.INITIALIZATION_KEY] == "passed"


def test_preview_uses_target_catalog_and_reports_legacy_policy(
    tmp_test_directory, monkeypatch, caplog
):
    catalog = {models.LEGACY_MODEL_KEY: {"default-tag": "v0.17.1"}}
    monkeypatch.setattr(
        production,
        "_target",
        lambda image: {
            production.TARGET_KEY: LEGACY_VERSION,
            CATALOG_KEY: catalog,
            production.IMAGE_DIGEST_KEY: "image",
        },
    )
    monkeypatch.setattr(
        models,
        IDENTITY_NAME,
        lambda *args: {
            SOURCE: str(tmp_test_directory),
            DIGEST: MODELS_NAME,
            models.VERSION_KEY: "0.17.2",
        },
    )
    with pytest.raises(ValueError, match=r"==0\.17\.1"):
        production._preview({production.PATH_KEY: str(tmp_test_directory)}, OLD_IMAGE)
    monkeypatch.setattr(
        models,
        IDENTITY_NAME,
        lambda *args: {
            SOURCE: str(tmp_test_directory),
            DIGEST: MODELS_NAME,
            models.VERSION_KEY: "0.17.1",
        },
    )
    with caplog.at_level("INFO"):
        result = production._preview({production.PATH_KEY: str(tmp_test_directory)}, OLD_IMAGE)
    assert result[production.TARGET_KEY] == LEGACY_VERSION
    assert "initialization not checked" in caplog.text
    assert models.LEGACY_POLICY in caplog.text


def test_target_inspection_uses_bundled_catalog(tmp_test_directory, monkeypatch):
    image = Path(tmp_test_directory) / "image.sif"
    image.write_bytes(b"image")
    captured = []

    def run(command, **kwargs):
        captured.append((command, kwargs))
        return subprocess.CompletedProcess(
            command, 0, json.dumps({production.TARGET_KEY: LEGACY_VERSION, CATALOG_KEY: {}})
        )

    monkeypatch.setattr(production.subprocess, "run", run)
    monkeypatch.setenv(CATALOG_ENV, "host.yml")
    target = production._target(image)
    assert target[production.IMAGE_DIGEST_KEY] == production._image_digest(image)
    assert CATALOG_ENV not in captured[0][1]["env"]
    assert "validate=False" in captured[0][0][-1]


def test_local_target_and_missing_source(monkeypatch):
    catalog = {models.MODEL_KEY: {"default-ref": "v0.18.0"}}
    monkeypatch.setattr(
        production.dependency_versions, "load_dependency_catalog", lambda *args: catalog
    )
    assert production._target()[CATALOG_KEY] == catalog
    with pytest.raises(ValueError, match="source path"):
        production._preview({})
    assert production._prepare_jobs([], {}, ExecutionOptions()) == []


def test_initialization_keeps_git_source_unmodified(tmp_test_directory):
    original = job(tmp_test_directory)
    proof = {SELECTION: GIT_SELECTION}
    initialized = production._initialization_job(original, 0, Path(tmp_test_directory), proof)
    sealed = production._sealed_job(original, proof)
    assert sealed.command == original.command
    assert initialized.command[1] == "-c"


def test_reject_mixed_images_before_validation(tmp_test_directory):
    original = job(tmp_test_directory)
    from dataclasses import replace

    with pytest.raises(ValueError, match="different target images"):
        production._prepare_jobs(
            [replace(original, resources={"container_image": "other.sif"})], {}, ExecutionOptions()
        )


def test_input_drift_during_canary_blocks_production(tmp_test_directory, proof, monkeypatch):
    monkeypatch.setattr(production, PREVIEW_NAME, lambda *args: proof)

    def mutate(*args):
        (Path(proof[SOURCE]) / models.MODELS_DIRECTORY / CHANGED).write_text(
            "new", encoding=ENCODING
        )

    monkeypatch.setattr(production, EXECUTE_NAME, mutate)
    with pytest.raises(ValueError, match="inputs changed"):
        production._prepare_jobs(
            [job(tmp_test_directory)],
            {OUTPUT_KEY: str(tmp_test_directory), production.PATH_KEY: proof[SOURCE]},
            ExecutionOptions(),
        )


def test_bundled_catalog_selection_ignores_working_directory(tmp_test_directory, monkeypatch):
    prefix = Path(tmp_test_directory) / "installed"
    bundled = prefix / PACKAGE_NAME / "dependency_versions.yml"
    bundled.parent.mkdir(parents=True)
    bundled.write_text("bundled", encoding=ENCODING)
    monkeypatch.setattr(sys, "prefix", str(prefix))
    monkeypatch.setattr(
        production.dependency_versions,
        "__file__",
        str(prefix / "lib" / "site-packages" / PACKAGE_NAME / "dependency_versions.py"),
    )
    monkeypatch.setenv(CATALOG_ENV, "host-catalog.yml")
    namespace = {}
    exec(production.BUNDLED_CATALOG_CODE, namespace)
    import os

    assert os.environ[CATALOG_ENV] == str(bundled)


def test_git_command_is_pinned_to_observed_commit(tmp_test_directory):
    proof = {SELECTION: GIT_SELECTION, models.COMMIT_KEY: "a" * 40}
    original = job(tmp_test_directory)
    pinned = production._sealed_job(original, proof)
    assert pinned.command[-2:] == ("--simulation_models_git_revision", "a" * 40)
    assert production._pin_git_command(pinned.command, proof) == pinned.command


def test_static_rejection_does_not_submit_canaries(tmp_test_directory, monkeypatch):
    called = []

    def reject(*args):
        raise ValueError("unsupported dependency release")

    monkeypatch.setattr(production, PREVIEW_NAME, reject)
    monkeypatch.setattr(production, EXECUTE_NAME, lambda *args: called.append(args))
    with pytest.raises(ValueError, match="unsupported dependency"):
        production._prepare_jobs([job(tmp_test_directory)], {}, ExecutionOptions())
    assert called == []


def test_canary_manifests_are_outside_production_collection(tmp_test_directory, proof, monkeypatch):
    monkeypatch.setattr(production, PREVIEW_NAME, lambda *args: proof)
    captured = []
    monkeypatch.setattr(production, EXECUTE_NAME, lambda jobs, options: captured.append(options))
    output = Path(tmp_test_directory) / "production-output"
    production._prepare_jobs(
        [job(tmp_test_directory)], {OUTPUT_KEY: str(output)}, ExecutionOptions()
    )
    assert not captured[0].work_dir.is_relative_to(output)


def test_image_change_during_inspection_is_rejected(tmp_test_directory, monkeypatch):
    image = Path(tmp_test_directory) / "changing.sif"
    image.write_bytes(b"original")
    digests = iter(["original-image", "replacement-image"])
    monkeypatch.setattr(production, "_image_digest", lambda path: next(digests))
    monkeypatch.setattr(
        production.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, "{}"),
    )
    with pytest.raises(ValueError, match="image changed during inspection"):
        production._target(image)
