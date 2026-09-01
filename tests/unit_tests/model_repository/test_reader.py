"""Tests for source-neutral simulation-model reading."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from simtools.application.model_reader import create_model_reader
from simtools.db.model_source import MongoDBModelSource
from simtools.model_repository import files as repository_files
from simtools.model_repository import reader as reader_module
from simtools.model_repository.reader import FileSystemModelSource, SimulationModelReader


def _write_json(path, data):
    """Write a JSON file and create its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _parameter(instrument, name, version, value):
    """Return a minimal model parameter document."""
    return {
        "file": False,
        "instrument": instrument,
        "parameter": name,
        "parameter_version": version,
        "site": "North",
        "type": "float64",
        "unit": None,
        "value": value,
    }


@pytest.fixture
def model_repository(tmp_test_directory):
    """Create a compact filesystem model repository."""
    root = Path(tmp_test_directory) / "models"
    production = root / "simulation-models" / "productions" / "1.0.0"
    _write_json(
        production / "LSTN-design.json",
        {
            "parameters": {"LSTN-design": {"camera_body_diameter": "1.0.0"}},
            "model_version": "1.0.0",
        },
    )
    _write_json(
        production / "LSTN-01.json",
        {
            "parameters": {"LSTN-01": {"camera_body_diameter": "2.0.0"}},
            "design_model": {"LSTN-01": "LSTN-design"},
            "model_version": "1.0.0",
        },
    )
    parameters = root / "simulation-models" / "model_parameters"
    _write_json(
        parameters / "LSTN-design/camera_body_diameter/camera_body_diameter-1.0.0.json",
        _parameter("LSTN-design", "camera_body_diameter", "1.0.0", 348.0),
    )
    _write_json(
        parameters / "LSTN-01/camera_body_diameter/camera_body_diameter-2.0.0.json",
        _parameter("LSTN-01", "camera_body_diameter", "2.0.0", 350.0),
    )
    return root


def test_reader_reads_resolved_parameters_and_design(model_repository):
    """The facade resolves production versions and design inheritance."""
    reader = SimulationModelReader.from_files(model_repository)

    assert reader.get_model_versions() == ["1.0.0"]
    assert reader.get_array_elements("1.0.0") == ["LSTN-01"]
    assert reader.get_design_model("1.0.0", "LSTN-01") == "LSTN-design"

    parameters = reader.get_model_parameters("North", "LSTN-01", "telescopes", "1.0.0")
    assert parameters["camera_body_diameter"]["value"] == pytest.approx(350.0)


def test_reader_factory_selects_filesystem_without_database(model_repository, mocker):
    """A configured repository path takes precedence over database setup."""
    database_handler = mocker.patch("simtools.db.db_handler.DatabaseHandler")

    reader = create_model_reader(simulation_models_path=model_repository)

    assert reader.source_name == str(model_repository.resolve())
    database_handler.assert_not_called()


def test_reader_factory_adapts_database_handler():
    """An explicitly supplied database handler can back the same reader API."""
    database_handler = Mock(model_source_name="simulation-model-db")

    reader = create_model_reader(database_handler=database_handler)

    assert reader.source_name == "simulation-model-db"


def test_mongodb_source_adapts_parameter_mapping():
    """The DB adapter presents parameter documents in the source protocol shape."""
    database_handler = Mock(model_source_name="simulation-model-db")
    database_handler._read_db.return_value = {  # pylint: disable=protected-access
        "camera_body_diameter": {"parameter": "camera_body_diameter", "value": 350.0}
    }
    source = MongoDBModelSource(database_handler)

    assert source.query_model_parameters({}, "telescopes") == [
        {"parameter": "camera_body_diameter", "value": 350.0}
    ]


def test_production_file_index_includes_patch_history(model_repository):
    """Patch productions include files from each version in their history."""
    patch_path = model_repository / "simulation-models/productions/1.1.0"
    patch_path.mkdir()
    (patch_path / "info.yml").write_text(
        "model_update: patch_update\nmodel_version_history:\n  - 1.0.0\n", encoding="utf-8"
    )
    _write_json(patch_path / "LSTN-01.json", {"parameters": {"LSTN-01": {}}})

    production_files = repository_files.get_production_table_files(patch_path)

    assert {version for version, _ in production_files} == {"1.0.0", "1.1.0"}


def test_filesystem_source_caches_reads_per_instance(model_repository, mocker):
    """Repeated reads are cached without sharing state between roots."""
    source = FileSystemModelSource(model_repository)
    read_spy = mocker.spy(reader_module.ascii_handler, "collect_data_from_file")
    parameter_path = source._parameter_path(
        "telescopes", "LSTN-01", "camera_body_diameter", "2.0.0"
    )
    initial_calls = read_spy.call_count
    source._read_parameter_file(parameter_path)
    source._read_parameter_file(parameter_path)
    assert read_spy.call_count - initial_calls == 1

    other_source = FileSystemModelSource(model_repository)
    assert other_source._parameters == {}


def test_filesystem_source_rejects_missing_repository(tmp_test_directory):
    """A missing repository fails with a useful path error."""
    with pytest.raises(FileNotFoundError, match="Simulation models path does not exist"):
        FileSystemModelSource(Path(tmp_test_directory) / "missing")
