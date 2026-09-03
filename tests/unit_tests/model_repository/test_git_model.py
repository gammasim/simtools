"""Tests for the Git-blob simulation-model source."""

import io
import json
import subprocess
from pathlib import Path

import pytest

from simtools.model_repository.git_backend import GitObjectStore
from simtools.model_repository.git_model import GitModelSource
from simtools.model_repository.reader import SimulationModelReader


class MemoryObjectStore(GitObjectStore):
    """Tiny object store used to test source behavior without Git or a network."""

    def __init__(self, objects):
        self.objects = objects
        self.reads = []
        self.list_calls = []

    def resolve_revision(self, revision):
        assert revision == "v1"
        return "a" * 40

    def iter_files(self, commit, prefix):
        self.list_calls.append(prefix)
        prefix = prefix.rstrip("/") + "/"
        return sorted(path for path in self.objects if path.startswith(prefix))

    def read_blob(self, commit, path):
        self.reads.append(path)
        return self.objects[path]

    def open_blob(self, commit, path):
        self.reads.append(path)
        return io.BytesIO(self.objects[path])


def _json(value):
    """Encode a test document as a Git blob."""
    return json.dumps(value).encode()


def _parameter(instrument, value):
    """Return a model parameter document."""
    return _json(
        {
            "file": False,
            "instrument": instrument,
            "parameter": "camera_body_diameter",
            "parameter_version": "1.0.0",
            "site": "North",
            "type": "float64",
            "unit": None,
            "value": value,
        }
    )


def test_git_source_preloads_tables_and_parameters_once(tmp_test_directory):
    """A model version warm-up reads all production and referenced blobs once."""
    objects = {
        "simulation-models/productions/1.0.0/LSTN-design.json": _json(
            {"parameters": {"LSTN-design": {"camera_body_diameter": "1.0.0"}}}
        ),
        "simulation-models/productions/1.0.0/LSTN-01.json": _json(
            {
                "parameters": {"LSTN-01": {"camera_body_diameter": "1.0.0"}},
                "design_model": {"LSTN-01": "LSTN-design"},
            }
        ),
        "simulation-models/model_parameters/LSTN-design/camera_body_diameter/"
        "camera_body_diameter-1.0.0.json": _parameter("LSTN-design", 348.0),
        "simulation-models/model_parameters/LSTN-01/camera_body_diameter/"
        "camera_body_diameter-1.0.0.json": _parameter("LSTN-01", 350.0),
    }
    store = MemoryObjectStore(objects)
    repository = Path(str(tmp_test_directory)) / "models.git"
    source = GitModelSource(repository, "v1", object_store=store)
    reader = SimulationModelReader(source)

    first = reader.get_model_parameters("North", "LSTN-01", "telescopes", "1.0.0")
    reads_after_first = list(store.reads)
    second = reader.get_model_parameters("North", "LSTN-01", "telescopes", "1.0.0")

    assert first["camera_body_diameter"]["value"] == pytest.approx(350.0)
    assert second == first
    assert store.reads == reads_after_first
    assert sorted(path for path in store.reads if path.endswith(".json")) == sorted(objects)
    assert source.source_config == {
        "type": "git",
        "repository": str(repository.resolve()),
        "commit": "a" * 40,
    }


def test_pygit2_source_reads_normal_and_bare_repositories(tmp_test_directory):
    """The pygit2 adapter reads the same immutable data from both repository forms."""
    pytest.importorskip("pygit2")
    repository = Path(str(tmp_test_directory)) / "models"
    objects = {
        "simulation-models/productions/1.0.0/LSTN-design.json": _json(
            {"parameters": {"LSTN-design": {"camera_body_diameter": "1.0.0"}}}
        ),
        "simulation-models/model_parameters/LSTN-design/camera_body_diameter/"
        "camera_body_diameter-1.0.0.json": _parameter("LSTN-design", 348.0),
    }
    for relative_path, data in objects.items():
        path = repository / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    _run_git("init", repository)
    _run_git("-C", repository, "config", "user.email", "tests@example.invalid")
    _run_git("-C", repository, "config", "user.name", "simtools tests")
    _run_git("-C", repository, "add", ".")
    _run_git("-C", repository, "commit", "-m", "model")
    _run_git("-C", repository, "tag", "v1")
    bare_repository = Path(str(tmp_test_directory)) / "models.git"
    _run_git("clone", "--bare", repository, bare_repository)

    for source_path in (repository, bare_repository):
        reader = SimulationModelReader.from_git(source_path, "v1")
        assert reader.get_model_versions() == ["1.0.0"]
        assert reader.get_model_parameters("North", "LSTN-design", "telescopes", "1.0.0")[
            "camera_body_diameter"
        ]["value"] == pytest.approx(348.0)


def _run_git(*args):
    """Run a local Git command used to construct a disposable repository."""
    subprocess.run(["git", *map(str, args)], check=True, capture_output=True, text=True)
