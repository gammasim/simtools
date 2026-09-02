"""Tests for source-neutral simulation-model reading."""

import ast
import json
import subprocess
import sys
import textwrap
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

    assert reader.source_config == {
        "type": "filesystem",
        "path": str(model_repository.resolve()),
    }
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


def test_path_first_startup_does_not_import_mongodb(tmp_test_directory):
    """A filesystem reader can start when MongoDB dependencies are unavailable."""
    repository = Path(tmp_test_directory)
    (repository / "simulation-models/productions").mkdir(parents=True)
    (repository / "simulation-models/model_parameters").mkdir(parents=True)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import builtins
                import sys

                real_import = builtins.__import__
                blocked = ("pymongo", "gridfs", "bson", "simtools.db")

                def guarded(name, *args, **kwargs):
                    if any(name == item or name.startswith(item + ".") for item in blocked):
                        raise ModuleNotFoundError(name=name)
                    return real_import(name, *args, **kwargs)

                builtins.__import__ = guarded
                from simtools.application.model_reader import create_model_reader

                create_model_reader(sys.argv[1])
                """
            ),
            str(repository),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_normal_runtime_modules_do_not_construct_database_handlers(simtools_root_path):
    """Database construction remains confined to the source-selection and DB packages."""
    allowed_source_selection = {
        simtools_root_path / "src/simtools/application/model_reader.py",
        simtools_root_path / "src/simtools/db/model_source.py",
        simtools_root_path / "src/simtools/db/mongo_db.py",
    }
    violations = [
        str(path)
        for path in _normal_runtime_files(simtools_root_path)
        if path.name != "model_reader.py"
        for violation in _boundary_violations(path, allowed_source_selection)
    ]
    assert violations == []


def _normal_runtime_files(simtools_root_path):
    """Yield Python files in modules that must remain database-independent."""
    roots = (
        "model",
        "simulator.py",
        "layout",
        "reporting",
        "visualization",
        "data_model",
        "configuration",
        "application",
        "testing",
        "simtel",
        "corsika",
    )
    for root_name in roots:
        root = simtools_root_path / "src/simtools" / root_name
        yield from root.rglob("*.py") if root.is_dir() else (root,)


def _boundary_violations(path, allowed_source_selection):
    """Return source-boundary violations found in one runtime module."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = list(ast.walk(tree))
    checks = (
        (_contains_database_handler_call(nodes), "DatabaseHandler construction"),
        (_contains_database_import(nodes), "database import"),
        (
            path not in allowed_source_selection and _contains_mongodb_literal(nodes),
            "MongoDB source literal",
        ),
        (
            path not in allowed_source_selection and _contains_mongodb_adapter(nodes),
            "MongoDB source adapter",
        ),
    )
    return [message for present, message in checks if present]


def _contains_database_handler_call(nodes):
    """Return whether AST nodes construct a database handler."""
    return any(
        isinstance(node, ast.Call) and _called_name(node) == "DatabaseHandler" for node in nodes
    )


def _contains_database_import(nodes):
    """Return whether AST nodes import the database package."""
    return any(
        isinstance(node, (ast.Import, ast.ImportFrom)) and "simtools.db" in ast.unparse(node)
        for node in nodes
    )


def _contains_mongodb_literal(nodes):
    """Return whether AST nodes contain the MongoDB source selector."""
    return any(isinstance(node, ast.Constant) and node.value == "mongodb" for node in nodes)


def _contains_mongodb_adapter(nodes):
    """Return whether AST nodes reference the MongoDB source adapter."""
    return any(isinstance(node, ast.Name) and node.id == "MongoDBModelSource" for node in nodes)


def _called_name(node):
    """Return the simple name of a called function."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


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
