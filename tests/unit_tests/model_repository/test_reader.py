"""Tests for source-neutral simulation-model reading."""

import json
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import Mock

import pytest
from astropy.table import Table

from simtools.application.model_reader import create_model_reader
from simtools.model_repository import reader as reader_module
from simtools.model_repository.reader import SimulationModelReader


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
    _write_json(
        parameters / "global/iobuf_maximum/iobuf_maximum-1.0.0.json",
        {
            **_parameter(None, "iobuf_maximum", "1.0.0", 1000.0),
            "site": None,
        },
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

    selected = reader.get_model_parameters(
        "North", "LSTN-01", "telescopes", "1.0.0", parameter_names={"camera_body_diameter"}
    )
    assert list(selected) == ["camera_body_diameter"]


def test_reader_preserves_generator_parameter_names(model_repository):
    """Filtering accepts one-shot iterables without dropping later names."""
    production_file = model_repository / "simulation-models/productions/1.0.0/LSTN-design.json"
    production = json.loads(production_file.read_text(encoding="utf-8"))
    production["parameters"]["LSTN-design"]["camera_body_diameter_2"] = "1.0.0"
    production_file.write_text(json.dumps(production), encoding="utf-8")
    _write_json(
        model_repository
        / "simulation-models/model_parameters/LSTN-design/camera_body_diameter_2/"
        / "camera_body_diameter_2-1.0.0.json",
        _parameter("LSTN-design", "camera_body_diameter_2", "1.0.0", 349.0),
    )

    reader = SimulationModelReader.from_files(model_repository)
    parameter_names = (name for name in ("camera_body_diameter_2", "camera_body_diameter"))

    selected = reader.get_model_parameters(
        "North", "LSTN-design", "telescopes", "1.0.0", parameter_names=parameter_names
    )

    assert list(selected) == ["camera_body_diameter", "camera_body_diameter_2"]


def test_reader_reads_global_parameter_by_version(model_repository):
    """A versioned global parameter does not require a global caller scope."""
    reader = SimulationModelReader.from_files(model_repository)

    parameter = reader.get_model_parameter(
        "iobuf_maximum", "North", "LSTN-01", parameter_version="1.0.0"
    )

    assert parameter["iobuf_maximum"]["value"] == pytest.approx(1000.0)


def test_reader_factory_selects_filesystem(model_repository):
    """A configured repository path selects the filesystem source."""
    reader = create_model_reader(simulation_models_path=model_repository)

    assert reader.source_name == str(model_repository.resolve())


def test_reader_factory_selects_memory_storage():
    """The in-memory constructor provides a source for generated model files."""
    production_path = "simulation-models/productions/1.0.0/LSTN-01.json"
    files = {
        production_path: json.dumps({"parameters": {"LSTN-01": {}}}).encode(),
    }

    reader = SimulationModelReader.from_memory(files)

    assert reader.source_name == "memory"
    assert reader.source_config["type"] == "memory"
    assert reader.get_model_versions() == ["1.0.0"]


def test_model_repository_import_has_no_external_source_dependency():
    """Filesystem reading has no external source dependency."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import simtools.model_repository.reader; "
                "raise SystemExit(any(name.startswith('simtools.model_source') for name in sys.modules))"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_path_first_startup_has_no_optional_source_dependency(tmp_test_directory):
    """A filesystem reader starts without optional source dependencies."""
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
                blocked = ("simtools.model_source",)

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
    source.get_parameter_table.return_value = "table"
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
    assert reader.is_configured() is True


def test_reader_caches_exported_ecsv_table_access():
    """Repeated table exports reuse the source-neutral table cache."""
    source = Mock(source_name="mock")
    source.export_model_files.return_value = {"table-LSTN-design.ecsv": "copied"}
    source.get_parameter_table.return_value = Table({"wavelength": [300.0]})
    reader = SimulationModelReader(source)
    parameter = {
        "parameter": "camera_filter",
        "parameter_version": "1.0.0",
        "instrument": "LSTN-design",
        "site": "North",
        "value": "table.ecsv",
    }

    first = reader._read_exported_parameter_table("camera_filter", parameter, Path("ignored.ecsv"))
    second = reader._read_exported_parameter_table("camera_filter", parameter, Path("ignored.ecsv"))

    assert first == second
    source.get_parameter_table.assert_called_once_with(parameter)


def test_reader_returns_defensive_table_record_copies():
    """Mutating table records does not corrupt the reader cache."""
    source = Mock(source_name="mock")
    source.get_parameter_table.return_value = Table({"wavelength": [300.0]})
    reader = SimulationModelReader(source)
    parameter = {
        "parameter": "camera_filter",
        "parameter_version": "1.0.0",
        "instrument": "LSTN-design",
        "site": "North",
        "value": "table.ecsv",
    }

    records = reader.get_parameter_table_records(parameter)
    records[0]["wavelength"] = 999.0

    assert reader.get_parameter_table_records(parameter) == [{"wavelength": 300.0}]
    source.get_parameter_table.assert_called_once_with(parameter)


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
    with pytest.raises(ValueError, match="not ECSV tables"):
        reader.export_model_file("p", "North", "LSTN-01", export_file_as_table=True)
    assert reader.export_model_file("p", "North", "LSTN-01") is None

    reader.get_model_parameter.return_value = {"p": {"value": "p.ecsv"}}
    reader.export_model_files = Mock()
    source.get_parameter_table.return_value = "ecsv-table"
    assert (
        reader.export_model_file("p", "North", "LSTN-01", export_file_as_table=True, dest="output")
        == "ecsv-table"
    )

    reader.get_model_parameter.return_value = {"p": {"value": "p.dat"}}
    with pytest.raises(ValueError, match="Destination path is required"):
        reader.export_model_file("p", "North", "LSTN-01")
    assert reader.export_model_file("p", "North", "LSTN-01", dest="output") is None
    with pytest.raises(ValueError, match="not an ECSV model table"):
        reader.export_model_file("p", "North", "LSTN-01", export_file_as_table=True, dest="output")
    assert reader.export_model_files.call_count == 3


def test_reader_caches_table_records(mocker):
    """Repeated camera-table record access does not rebuild row dictionaries."""
    source = Mock()
    source.get_parameter_table.return_value = Table({"pixel_id": [1, 2]})
    reader = SimulationModelReader(source)
    parameter = {
        "parameter": "camera_pixel_layout",
        "parameter_version": "1.0.0",
        "instrument": "LSTN-design",
        "site": "North",
        "value": "layout.ecsv",
    }

    first = reader.get_parameter_table_records(parameter)
    second = reader.get_parameter_table_records(parameter)

    assert first == second
    assert first is not second
    source.get_parameter_table.assert_called_once_with(parameter)


def test_reader_caches_parameter_tables():
    """Repeated table access uses the parsed table from the reader cache."""
    source = Mock()
    source.get_parameter_table.return_value = Table({"pixel_id": [1, 2]})
    reader = SimulationModelReader(source)
    parameter = {
        "parameter": "camera_pixel_layout",
        "parameter_version": "1.0.0",
        "instrument": "LSTN-design",
        "site": "North",
        "value": "layout.ecsv",
    }

    first = reader.get_parameter_table(parameter)
    second = reader.get_parameter_table(parameter)

    assert first["pixel_id"].tolist() == second["pixel_id"].tolist()
    assert first is not second
    source.get_parameter_table.assert_called_once_with(parameter)


def test_reader_facade_delegates_git_source_and_optional_source_config(mocker):
    """Git construction and source metadata are exposed by the facade."""
    git_storage = Mock(name="storage", name_attr="models@commit")
    git_storage.name = "models@commit"
    git_storage.config = {"type": "git"}
    git_storage_class = mocker.patch(
        "simtools.model_repository.reader.GitStorage", return_value=git_storage
    )

    reader = SimulationModelReader.from_git("models.git", "abc", object_store="store")

    git_storage_class.assert_called_once_with("models.git", "abc", object_store="store")
    assert reader.source_name == "models@commit"
    assert reader.source_config == {"type": "git"}

    source_without_config = Mock(spec=["source_name", "is_configured"])
    assert SimulationModelReader(source_without_config).source_config is None


def test_reader_facade_handles_value_errors_across_model_versions():
    """Unavailable model versions are skipped during aggregate reads."""
    source = Mock()
    source.get_model_versions.return_value = ["1.0.0"]
    reader = SimulationModelReader(source)
    reader.get_model_parameters = Mock(side_effect=ValueError("not available"))

    assert (
        reader.get_model_parameters_for_all_model_versions("North", "LSTN-01", "telescopes") == {}
    )


def test_reader_facade_selects_all_array_element_scope_branches(mocker):
    """Scope selection handles global, design, and concrete requests."""
    source = Mock()
    source.read_production_table.return_value = {
        "model_version": "1.0.0",
        "design_model": {"LSTN-01": "LSTN-design"},
    }
    reader = SimulationModelReader(source)
    simtel_scopes = mocker.patch.object(
        reader, "_get_sim_telarray_array_element_list", return_value=["global"]
    )

    assert reader._get_array_element_list(None, None, {}, "configuration_corsika") == ["global"]
    assert reader._get_array_element_list("LSTN-01", "North", {}, "configuration_sim_telarray") == [
        "global"
    ]
    simtel_scopes.assert_called_once()
    assert reader._get_array_element_list("LSTN-design", None, {}, "telescopes") == ["LSTN-design"]
    assert reader._get_array_element_list(
        "LSTN-01", None, source.read_production_table.return_value, "telescopes"
    ) == [
        "LSTN-design",
        "LSTN-01",
    ]


def test_reader_simtel_scope_lookup_supports_fallback_and_wraps_errors(mocker, run_context):
    """Concrete sim_telarray scopes use design data or configured fallbacks."""
    source = Mock()
    source.read_production_table.return_value = {"design_model": {"LSTN-01": "LSTN-design"}}
    reader = SimulationModelReader(source)
    context = run_context.replace(args={"ignore_missing_design_model": False})

    assert reader._get_sim_telarray_array_element_list(
        "LSTN-01", {"model_version": "1.0.0"}, context
    ) == [
        "global",
        "LSTN-design",
        "LSTN-01",
    ]
    assert reader._get_sim_telarray_array_element_list("global", {}, context) == ["global"]
    assert reader._get_sim_telarray_array_element_list("LSTN-design", {}, context) == [
        "global",
        "LSTN-design",
    ]

    source.read_production_table.return_value = {"design_model": {}}
    context = run_context.replace(args={"ignore_missing_design_model": True})
    assert reader._get_sim_telarray_array_element_list(
        "LSTN-01", {"model_version": "1.0.0"}, context
    ) == [
        "global",
        "LSTN-01",
        "LSTN-01",
        "LSTN-design",
    ]

    scopes = mocker.patch.object(
        reader_module.catalog,
        "get_sim_telarray_parameter_scopes",
        side_effect=KeyError("missing"),
    )
    with pytest.raises(KeyError, match="Failed to generate array element list"):
        reader._get_sim_telarray_array_element_list(None, {})
    scopes.assert_called_once()


def test_from_files_requires_model_directories(tmp_test_directory):
    """A missing checkout or missing model directories raise before any read."""
    root = Path(tmp_test_directory)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        SimulationModelReader.from_files(root / "missing")
    (root / "partial" / "simulation-models" / "productions").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="model_parameters"):
        SimulationModelReader.from_files(root / "partial")
