"""Tests for source-neutral simulation-model reading."""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from simtools.application.model_reader import create_model_reader
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


def test_model_repository_import_does_not_load_database_modules():
    """Filesystem reading has no database or MongoDB import dependency."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import simtools.model_repository.reader; "
                "raise SystemExit(any(name == 'simtools.db' or name.startswith('simtools.db.') "
                "or name.startswith('pymongo') or name.startswith('gridfs') for name in sys.modules))"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_reader_facade_routes_source_operations_and_branches():
    """The facade resolves parameters and delegates source-specific operations."""
    source = Mock(source_name="mock")
    source.get_model_versions.return_value = ["1.0.0"]
    source.read_production_table.return_value = {
        "model_version": "1.0.0",
        "parameters": {"LSTN-design": {"camera_body_diameter": "1.0.0"}, "LSTN-01": {}},
        "design_model": {"LSTN-01": "LSTN-design"},
    }
    source.read_parameters.side_effect = lambda versions, *_: [
        {"parameter": parameter, "value": 1} for parameter in versions
    ]
    source.export_model_files.return_value = {"model.dat": "copied"}
    source.get_ecsv_file_as_astropy_table.return_value = "table"
    source.is_configured.return_value = True
    reader = SimulationModelReader(source)

    assert reader.get_model_parameter(
        "camera_body_diameter", "North", "LSTN-01", model_version="1.0.0"
    ) == {"camera_body_diameter": {"parameter": "camera_body_diameter", "value": 1}}
    assert reader.get_array_elements_of_type("LST", "1.0.0", "telescopes") == ["LSTN-01"]
    assert reader._get_array_element_list(None, "North", {}, "sites") == ["OBS-North"]
    assert reader._get_array_element_list("LSTN-design", None, {}, "telescopes") == ["LSTN-design"]
    assert reader.get_simulation_configuration_parameters("corsika", None, None, "1.0.0") == {}
    assert reader.get_simulation_configuration_parameters(
        "sim_telarray", "North", "LSTN-01", "1.0.0"
    ) == {"camera_body_diameter": {"parameter": "camera_body_diameter", "value": 1}}
    assert reader.get_simulation_configuration_parameters("sim_telarray", None, None, "1.0.0") == {}
    with pytest.raises(ValueError, match="Unknown simulation software"):
        reader.get_simulation_configuration_parameters("other", None, None, "1.0.0")
    with pytest.raises(ValueError, match="not a list"):
        reader.get_model_parameter(
            "camera_body_diameter", "North", "LSTN-01", model_version=["1.0.0"]
        )
    assert reader.export_model_files(file_names="model.dat", dest="output") == {
        "model.dat": "copied"
    }
    assert reader.get_ecsv_file_as_astropy_table("model.ecsv") == "table"
    assert reader.is_configured() is True


def test_reader_facade_covers_all_version_and_export_paths(mocker):
    """Cover the source-independent convenience methods."""
    source = Mock(source_name="mock")
    source.is_configured.return_value = True
    source.get_model_versions.return_value = ["1.0.0", "2.0.0"]
    reader = SimulationModelReader(source)

    reader.get_model_parameters = Mock(side_effect=[{"p": 1}, KeyError("missing")])
    assert reader.get_model_parameters_for_all_model_versions("North", "LSTN-01", "telescopes") == {
        "1.0.0": {"p": 1}
    }

    reader.get_model_parameter = Mock(return_value={"p": {"type": "dict", "value": {"x": [1]}}})
    row_table = mocker.patch(
        "simtools.model_repository.reader.simtel_table_reader.row_data_to_astropy_table",
        return_value="table",
    )
    assert reader.export_model_file("p", "North", "LSTN-01", export_file_as_table=True) == "table"
    row_table.assert_called_once_with({"x": [1]})
    assert reader.export_model_file("p", "North", "LSTN-01") is None

    reader.get_model_parameter.return_value = {"p": {"value": "p.dat"}}
    with pytest.raises(ValueError, match="Destination path is required"):
        reader.export_model_file("p", "North", "LSTN-01")
    reader.export_model_files = Mock()
    assert reader.export_model_file("p", "North", "LSTN-01", dest="output") is None
    read_table = mocker.patch(
        "simtools.model_repository.reader.simtel_table_reader.read_simtel_table",
        return_value="file-table",
    )
    assert (
        reader.export_model_file("p", "North", "LSTN-01", export_file_as_table=True, dest="output")
        == "file-table"
    )
    assert reader.export_model_files.call_count == 2
    read_table.assert_called_once_with("p", Path("output") / "p.dat")
