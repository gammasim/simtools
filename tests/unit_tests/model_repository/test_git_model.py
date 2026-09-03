"""Tests for the Git-blob simulation-model source."""

import io
import json
import subprocess
from pathlib import Path

import pytest
from astropy.table import Table

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


def test_git_source_exposes_metadata_and_reports_missing_tables(tmp_test_directory):
    """The Git source exposes its immutable identity and table errors."""
    objects = {
        "simulation-models/productions/1.0.0/LSTN-design.json": _json(
            {"parameters": {"LSTN-design": {}}}
        )
    }
    repository = Path(str(tmp_test_directory)) / "models.git"
    source = GitModelSource(repository, "v1", object_store=MemoryObjectStore(objects))

    assert source.source_name == f"{repository.resolve()}@{'a' * 40}"
    assert source.is_configured() is True
    assert source.get_model_versions("sites") == ["1.0.0"]
    with pytest.raises(ValueError, match="No production table for sites"):
        source.read_production_table("sites", "1.0.0")


def test_git_source_reads_patch_history_and_ignores_non_json_files(tmp_test_directory):
    """Patch production documents are combined in version order."""
    objects = {
        "simulation-models/productions/1.0.0/LSTN-design.json": _json(
            {"parameters": {"LSTN-design": {}}}
        ),
        "simulation-models/productions/1.1.0/info.yml": (
            b"model_update: patch_update\nmodel_version_history:\n  - 1.0.0\n"
        ),
        "simulation-models/productions/1.1.0/LSTN-01.json": _json(
            {"parameters": {"LSTN-01": {}}, "design_model": {"LSTN-01": "LSTN-design"}}
        ),
        "simulation-models/productions/1.1.0/README.txt": b"not a production table",
    }
    source = GitModelSource(
        Path(str(tmp_test_directory)) / "models.git", "v1", object_store=MemoryObjectStore(objects)
    )

    documents = source._production_documents("1.1.0")

    assert [(model, path) for model, path, _ in documents] == [
        ("1.0.0", "simulation-models/productions/1.0.0/LSTN-design.json"),
        ("1.1.0", "simulation-models/productions/1.1.0/LSTN-01.json"),
    ]

    empty_source = GitModelSource(
        Path(str(tmp_test_directory)) / "empty.git", "v1", object_store=MemoryObjectStore({})
    )
    with pytest.raises(ValueError, match=r"Model version 2\.0\.0 not found"):
        empty_source._production_documents("2.0.0")


def test_git_source_collects_parameter_paths_and_handles_missing_values():
    """Parameter warm-up indexes only dictionary string references."""
    tables = {
        "telescopes": {
            "parameters": {
                "LSTN-01": {"camera_body_diameter": "1.0.0", "ignored": 1},
                "invalid": "not a mapping",
            }
        }
    }

    paths = GitModelSource._parameter_paths(tables)

    expected_path = GitModelSource._parameter_path(
        "telescopes", "LSTN-01", "camera_body_diameter", "1.0.0"
    )
    assert paths == {expected_path: ("camera_body_diameter", "1.0.0")}


def test_git_source_reads_and_filters_parameters(tmp_test_directory, mocker):
    """Parameter reads normalize data, cache blobs, and filter metadata."""
    path = GitModelSource._parameter_path("telescopes", "LSTN-01", "camera_body_diameter", "1.0.0")
    store = MemoryObjectStore({path: _parameter("LSTN-01", 350.0)})
    source = GitModelSource(Path(str(tmp_test_directory)) / "models.git", "v1", object_store=store)

    parameters = source.read_parameters(
        {"camera_body_diameter": "1.0.0"},
        "telescopes",
        instrument="LSTN-01",
        site="North",
    )
    assert parameters[0]["value"] == pytest.approx(350.0)
    source.read_parameters(
        {"camera_body_diameter": "1.0.0"},
        "telescopes",
        instrument="LSTN-01",
        site="North",
    )
    assert store.reads.count(path) == 1

    source._parameters[path]["site"] = "South"
    with pytest.raises(ValueError, match="No parameters found"):
        source.read_parameters(
            {"camera_body_diameter": "1.0.0"},
            "telescopes",
            instrument="LSTN-01",
            site="North",
        )

    missing = GitModelSource(
        Path(str(tmp_test_directory)) / "missing.git", "v1", object_store=MemoryObjectStore({})
    )
    mocker.patch.object(missing._object_store, "read_blob", side_effect=FileNotFoundError)
    with pytest.raises(ValueError, match="No parameters found"):
        missing.read_parameters(
            {"camera_body_diameter": "1.0.0"},
            "telescopes",
            instrument="LSTN-01",
            site="North",
        )


@pytest.mark.parametrize(
    ("query", "collection", "expected"),
    [
        ({"instrument": "LSTN-01"}, "telescopes", "LSTN-01"),
        ({"site": "North"}, "sites", "OBS-North"),
        ({}, "configuration_corsika", "global"),
        ({}, "configuration_sim_telarray", "global"),
    ],
)
def test_git_source_resolves_parameter_instruments(query, collection, expected):
    """Git parameter lookups resolve element, site, and global scopes."""
    assert GitModelSource._get_parameter_instrument(query, collection) == expected


def test_git_source_rejects_ambiguous_parameter_instrument():
    """Non-global collections require an explicit element scope."""
    with pytest.raises(ValueError, match="requires an array element name"):
        GitModelSource._get_parameter_instrument({}, "telescopes")


@pytest.mark.parametrize(
    ("data", "instrument", "site", "expected"),
    [
        ({"instrument": "LSTN-01", "site": "North"}, "LSTN-01", "North", True),
        ({"instrument": "MSTN-01", "site": "North"}, "LSTN-01", "North", False),
        ({"instrument": "LSTN-01", "site": ["North", "South"]}, "LSTN-01", "South", True),
        ({"instrument": "LSTN-01", "site": "North"}, "LSTN-01", "South", False),
        ({"instrument": None, "site": None}, "global", "North", True),
    ],
)
def test_git_source_matches_parameter_filters(data, instrument, site, expected):
    """Git parameter filters handle scalar, list, and global metadata."""
    assert GitModelSource._matches_filters(data, instrument, site) is expected


def test_git_source_exports_files_lazily_and_safely(tmp_test_directory, mocker):
    """Git files are streamed on demand and cannot escape the Files directory."""
    objects = {
        "simulation-models/model_parameters/Files/nested/model.dat": b"model",
        "simulation-models/model_parameters/Files/other.dat": b"other",
    }
    store = MemoryObjectStore(objects)
    source = GitModelSource(Path(str(tmp_test_directory)) / "models.git", "v1", object_store=store)
    destination = Path(str(tmp_test_directory)) / "exported"

    assert source.export_model_files(file_names="nested/model.dat", dest=destination) == {
        "nested/model.dat": "copied from Git"
    }
    assert (destination / "nested/model.dat").read_bytes() == b"model"
    assert source.export_model_files(file_names="nested/model.dat", dest=destination) == {
        "nested/model.dat": "file exists"
    }
    assert source.export_model_files(
        parameters={"selected": {"file": True, "value": "other.dat"}}, dest=destination
    ) == {"other.dat": "copied from Git"}
    with pytest.raises(ValueError, match="escapes model Files"):
        source.export_model_files(file_names="../model.dat", dest=destination)
    with pytest.raises(ValueError, match="Destination path is required"):
        source.export_model_files(file_names="other.dat")

    missing = mocker.patch.object(store, "open_blob", side_effect=FileNotFoundError)
    with pytest.raises(FileNotFoundError, match="Model file not found at commit"):
        source.export_model_files(file_names="missing.dat", dest=destination)
    missing.assert_called_once()


def test_git_source_reads_ecsv_and_rejects_invalid_file_paths(tmp_test_directory, mocker):
    """ECSV blobs are read as tables and missing blobs get a useful error."""
    buffer = io.StringIO()
    Table({"value": [1, 2]}).write(buffer, format="ascii.ecsv")
    file_name = "values.ecsv"
    path = f"simulation-models/model_parameters/Files/{file_name}"
    store = MemoryObjectStore({path: buffer.getvalue().encode()})
    source = GitModelSource(Path(str(tmp_test_directory)) / "models.git", "v1", object_store=store)

    assert source.get_ecsv_file_as_astropy_table(file_name)["value"].tolist() == [1, 2]
    with pytest.raises(ValueError, match="escapes model Files"):
        source.get_ecsv_file_as_astropy_table("../values.ecsv")
    mocker.patch.object(store, "open_blob", side_effect=FileNotFoundError)
    with pytest.raises(FileNotFoundError, match="Model file not found at commit"):
        source.get_ecsv_file_as_astropy_table(file_name)


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
