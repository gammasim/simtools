"""Tests for reading simulation models from files."""

import json
from pathlib import Path

import pytest
from astropy.table import Table

from simtools.db import db_handler, file_system_model

pytestmark = pytest.mark.db_unit_test


def _write_json(path, data):
    """Write JSON test data, creating its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _parameter(instrument, site, name, version, value, file=False, parameter_type=None, unit=None):
    """Return minimal filesystem model-parameter data."""
    return {
        "file": file,
        "instrument": instrument,
        "meta_parameter": False,
        "model_parameter_schema_version": "0.1.0",
        "parameter": name,
        "parameter_version": version,
        "schema_version": "0.3.0",
        "site": site,
        "type": parameter_type or ("string" if file else "float64"),
        "unique_id": None,
        "unit": unit,
        "value": value,
    }


@pytest.fixture
def simulation_models_path(tmp_test_directory):
    """Create a small but representative simulation-model directory."""
    model_root = Path(tmp_test_directory) / "model-files"
    productions = model_root / "simulation-models" / "productions" / "1.0.0"
    parameters = model_root / "simulation-models" / "model_parameters"

    _write_json(
        productions / "LSTN-design.json",
        {
            "model_version": "1.0.0",
            "production_table_name": "LSTN-design",
            "parameters": {"LSTN-design": {"camera_body_diameter": "1.0.0"}},
        },
    )
    _write_json(
        productions / "LSTN-01.json",
        {
            "design_model": {"LSTN-01": "LSTN-design"},
            "model_version": "1.0.0",
            "production_table_name": "LSTN-01",
            "parameters": {"LSTN-01": {"camera_body_diameter": "2.0.0"}},
        },
    )
    _write_json(
        productions / "OBS-North.json",
        {
            "model_version": "1.0.0",
            "production_table_name": "OBS-North",
            "parameters": {"OBS-North": {"array_layouts": "1.0.0"}},
        },
    )
    _write_json(
        productions / "configuration_corsika.json",
        {
            "model_version": "1.0.0",
            "production_table_name": "configuration_corsika",
            "parameters": {
                "xSTx-design": {
                    "corsika_cherenkov_photon_bunch_size": "1.0.0",
                    "corsika_particle_kinetic_energy_cutoff": "1.0.0",
                }
            },
        },
    )
    _write_json(
        productions / "configuration_sim_telarray.json",
        {
            "model_version": "1.0.0",
            "production_table_name": "configuration_sim_telarray",
            "parameters": {"LSTN-design": {"min_photons": "1.0.0"}},
        },
    )

    _write_json(
        parameters / "LSTN-design/camera_body_diameter/camera_body_diameter-1.0.0.json",
        _parameter("LSTN-design", "North", "camera_body_diameter", "1.0.0", 348.0, unit="cm"),
    )
    _write_json(
        parameters / "LSTN-design/dsum_prescale/dsum_prescale-1.0.0.json",
        _parameter(
            "LSTN-design",
            "North",
            "dsum_prescale",
            "1.0.0",
            [42.0, 256.0],
            parameter_type="int64",
        ),
    )
    _write_json(
        parameters / "LSTN-01/camera_body_diameter/camera_body_diameter-2.0.0.json",
        _parameter("LSTN-01", "North", "camera_body_diameter", "2.0.0", 350.0, unit="cm"),
    )
    _write_json(
        parameters / "OBS-North/array_layouts/array_layouts-1.0.0.json",
        _parameter(
            "OBS-North",
            "North",
            "array_layouts",
            "1.0.0",
            [{"name": "test", "elements": ["LSTN-01"]}],
        ),
    )
    _write_json(
        parameters
        / (
            "configuration_corsika/corsika_cherenkov_photon_bunch_size/"
            "corsika_cherenkov_photon_bunch_size-1.0.0.json"
        ),
        _parameter(
            None,
            None,
            "corsika_cherenkov_photon_bunch_size",
            "1.0.0",
            5.0,
        ),
    )
    _write_json(
        parameters
        / (
            "configuration_corsika/corsika_particle_kinetic_energy_cutoff/"
            "corsika_particle_kinetic_energy_cutoff-1.0.0.json"
        ),
        _parameter(
            None,
            None,
            "corsika_particle_kinetic_energy_cutoff",
            "1.0.0",
            [0.3, 0.1, 0.02, 0.02],
            unit="GeV",
        ),
    )
    _write_json(
        parameters / "configuration_sim_telarray/LSTN-design/min_photons/min_photons-1.0.0.json",
        _parameter("LSTN-design", "North", "min_photons", "1.0.0", 2.0),
    )
    files = parameters / "Files"
    files.mkdir()
    (files / "model.dat").write_text("model data\n", encoding="utf-8")
    Table({"value": [1.0]}).write(files / "model.ecsv", format="ascii.ecsv")
    return model_root


@pytest.fixture(autouse=True)
def clear_file_system_caches():
    """Prevent filesystem cache state from leaking between tests."""
    file_system_model.FileSystemModelHandler.clear_caches()
    db_handler.DatabaseHandler.model_parameters_cached.clear()
    yield
    file_system_model.FileSystemModelHandler.clear_caches()
    db_handler.DatabaseHandler.model_parameters_cached.clear()


def test_file_system_handler_reads_production_and_parameters(simulation_models_path):
    handler = file_system_model.FileSystemModelHandler(simulation_models_path)

    production = handler.read_production_table("telescopes", "1.0.0")
    parameters = handler.query_model_parameters(
        {
            "$or": [{"parameter": "camera_body_diameter", "parameter_version": "2.0.0"}],
            "instrument": "LSTN-01",
            "site": "North",
        },
        "telescopes",
    )

    assert production["design_model"] == {"LSTN-01": "LSTN-design"}
    assert parameters[0]["value"] == pytest.approx(350.0)
    assert handler.get_model_versions() == ["1.0.0"]

    integer_parameter = handler.query_model_parameters(
        {
            "instrument": "LSTN-design",
            "parameter": "dsum_prescale",
            "parameter_version": "1.0.0",
            "site": "North",
        },
        "telescopes",
    )
    assert integer_parameter[0]["value"] == [42, 256]
    assert all(isinstance(value, int) for value in integer_parameter[0]["value"])


def test_file_system_handler_caches_production_and_parameter_reads(simulation_models_path, mocker):
    production_spy = mocker.spy(file_system_model.db_model_upload, "read_production_tables")
    parameter_spy = mocker.spy(file_system_model.ascii_handler, "collect_data_from_file")
    query = {
        "parameter": "camera_body_diameter",
        "parameter_version": "2.0.0",
        "instrument": "LSTN-01",
        "site": "North",
    }

    for _ in range(2):
        handler = file_system_model.FileSystemModelHandler(simulation_models_path)
        handler.read_production_table("telescopes", "1.0.0")
        handler.query_model_parameters(query, "telescopes")

    assert production_spy.call_count == 1
    parameter_reads = [
        call
        for call in parameter_spy.call_args_list
        if "camera_body_diameter-2.0.0.json" in str(call.kwargs.get("file_name"))
    ]
    assert len(parameter_reads) == 1


def test_database_handler_uses_files_without_mongodb(simulation_models_path, mocker):
    settings_mock = mocker.patch("simtools.db.db_handler.settings")
    settings_mock.config.args = {"simulation_models_path": simulation_models_path}
    settings_mock.config.db_config = {"invalid": "mongo config must not be validated"}
    mongo_handler = mocker.patch("simtools.db.db_handler.MongoDBHandler")

    database = db_handler.DatabaseHandler()
    parameters = database.get_model_parameters("North", "LSTN-01", "telescopes", "1.0.0")
    layouts = database.get_model_parameter(
        "array_layouts", "North", None, parameter_version="1.0.0"
    )
    corsika = database.get_simulation_configuration_parameters("corsika", None, None, "1.0.0")
    sim_telarray = database.get_simulation_configuration_parameters(
        "sim_telarray", "North", "LSTN-01", "1.0.0"
    )

    assert database.is_configured()
    assert parameters["camera_body_diameter"]["value"] == pytest.approx(350.0)
    assert layouts["array_layouts"]["value"] == {"name": "test", "elements": ["LSTN-01"]}
    assert corsika["corsika_cherenkov_photon_bunch_size"]["value"] == pytest.approx(5.0)
    assert corsika["corsika_particle_kinetic_energy_cutoff"]["unit"] == "GeV"
    assert sim_telarray["min_photons"]["value"] == pytest.approx(2.0)
    mongo_handler.assert_not_called()


def test_file_export_and_mongodb_only_guard(simulation_models_path, tmp_test_directory):
    handler = file_system_model.FileSystemModelHandler(simulation_models_path)
    destination = Path(tmp_test_directory) / "export"

    result = handler.export_model_files(file_names="model.dat", dest=destination)

    assert result == {"model.dat": "copied from filesystem"}
    assert (destination / "model.dat").read_text(encoding="utf-8") == "model data\n"
    assert handler.get_ecsv_file_as_astropy_table("model.ecsv")["value"][0] == pytest.approx(1.0)

    parameters = {"model_file": {"file": True, "value": "model.dat"}}
    assert handler.export_model_files(parameters=parameters, dest=destination) == {
        "model.dat": "file exists"
    }


def test_patch_production_inherits_base_tables(simulation_models_path):
    patch_production = simulation_models_path / "simulation-models" / "productions" / "1.0.1"
    patch_production.mkdir()
    (patch_production / "info.yml").write_text(
        "model_update: patch_update\nmodel_version_history:\n  - 1.0.0\n",
        encoding="utf-8",
    )
    _write_json(
        patch_production / "LSTN-01.json",
        {
            "design_model": {"LSTN-01": "LSTN-design"},
            "model_version": "1.0.1",
            "production_table_name": "LSTN-01",
            "parameters": {"LSTN-01": {"camera_body_diameter": "2.0.0"}},
        },
    )

    handler = file_system_model.FileSystemModelHandler(simulation_models_path)
    production = handler.read_production_table("telescopes", "1.0.1")

    assert production["model_version"] == "1.0.1"
    assert production["parameters"]["LSTN-design"]["camera_body_diameter"] == "1.0.0"
    assert production["parameters"]["LSTN-01"]["camera_body_diameter"] == "2.0.0"


def test_database_handler_rejects_mongodb_operation(simulation_models_path, mocker):
    settings_mock = mocker.patch("simtools.db.db_handler.settings")
    settings_mock.config.args = {"simulation_models_path": simulation_models_path}
    settings_mock.config.db_config = {}
    database = db_handler.DatabaseHandler()

    with pytest.raises(RuntimeError, match="requires a MongoDB model source"):
        database.get_collections()


def test_invalid_model_path_fails_without_fallback(tmp_test_directory):
    with pytest.raises(FileNotFoundError, match="Expected simulation models directory"):
        file_system_model.FileSystemModelHandler(Path(tmp_test_directory) / "model")

    with pytest.raises(FileNotFoundError, match="path does not exist"):
        file_system_model.FileSystemModelHandler(Path(tmp_test_directory) / "missing")


def test_missing_model_data_reports_source(simulation_models_path):
    handler = file_system_model.FileSystemModelHandler(simulation_models_path)

    with pytest.raises(ValueError, match=r"Model version 2\.0\.0 not found"):
        handler.read_production_table("telescopes", "2.0.0")
    with pytest.raises(ValueError, match="No production table for collection"):
        handler.read_production_table("calibration_devices", "1.0.0")
    with pytest.raises(FileNotFoundError, match="Model parameter file not found"):
        handler.query_model_parameters(
            {
                "parameter": "camera_body_diameter",
                "parameter_version": "9.0.0",
                "instrument": "LSTN-01",
                "site": "North",
            },
            "telescopes",
        )
    with pytest.raises(ValueError, match="requires an array element name"):
        handler.query_model_parameters(
            {"parameter": "camera_body_diameter", "parameter_version": "1.0.0"},
            "telescopes",
        )


def test_model_file_export_errors(simulation_models_path, tmp_test_directory):
    handler = file_system_model.FileSystemModelHandler(simulation_models_path)

    with pytest.raises(ValueError, match="Destination path is required"):
        handler.export_model_files(file_names="model.dat")
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        handler.export_model_files(file_names="missing.dat", dest=tmp_test_directory)
    with pytest.raises(ValueError, match="escapes model"):
        handler.export_model_files(file_names="../outside.dat", dest=tmp_test_directory)
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        handler.get_ecsv_file_as_astropy_table("missing.ecsv")
