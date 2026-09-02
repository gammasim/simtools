"""Tests for reading simulation models from files."""

import json
from pathlib import Path

import pytest
from astropy.table import Table

from simtools.model_repository import reader as reader_module
from simtools.model_repository.reader import FileSystemModelSource, SimulationModelReader


def _write_json(path, data):
    """Write JSON test data, creating its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _parameter(instrument, site, name, version, value, file=False, parameter_type=None, unit=None):
    """Return minimal filesystem model-parameter data."""
    return {
        "file": file,
        "instrument": instrument,
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
                    "corsika_starting_grammage": "1.0.2",
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
            parameter_type="uint64",
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
        parameters
        / "configuration_corsika/corsika_starting_grammage/corsika_starting_grammage-1.0.2.json",
        _parameter(
            None,
            None,
            "corsika_starting_grammage",
            "1.0.2",
            [
                {"instrument": "LSTN-design", "primary_particle": "muon-", "value": 580.0},
                {"instrument": "LSTN-design", "primary_particle": "default", "value": 0.0},
            ],
            parameter_type="dict",
            unit="g/cm2",
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


def test_file_system_handler_reads_production_and_parameters(simulation_models_path):
    handler = FileSystemModelSource(simulation_models_path)

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


def test_file_system_handler_ignores_missing_files_in_or_query(simulation_models_path):
    handler = FileSystemModelSource(simulation_models_path)

    parameters = handler.query_model_parameters(
        {
            "$or": [
                {"parameter": "camera_body_diameter", "parameter_version": "9.0.0"},
                {"parameter": "camera_body_diameter", "parameter_version": "2.0.0"},
            ],
            "instrument": "LSTN-01",
            "site": "North",
        },
        "telescopes",
    )

    assert [parameter["parameter"] for parameter in parameters] == ["camera_body_diameter"]


def test_file_system_handler_caches_production_and_parameter_reads(simulation_models_path, mocker):
    production_spy = mocker.spy(reader_module.files, "read_production_tables")
    parameter_spy = mocker.spy(reader_module.ascii_handler, "collect_data_from_file")
    query = {
        "parameter": "camera_body_diameter",
        "parameter_version": "2.0.0",
        "instrument": "LSTN-01",
        "site": "North",
    }

    for _ in range(2):
        handler = FileSystemModelSource(simulation_models_path)
        handler.read_production_table("telescopes", "1.0.0")
        handler.query_model_parameters(query, "telescopes")

    assert production_spy.call_count == 2
    parameter_reads = [
        call
        for call in parameter_spy.call_args_list
        if "camera_body_diameter-2.0.0.json" in str(call.kwargs.get("file_name"))
    ]
    assert len(parameter_reads) == 2


def test_file_system_handler_reads_requested_production_collection_only(
    simulation_models_path, mocker
):
    parameter_spy = mocker.spy(reader_module.ascii_handler, "collect_data_from_file")
    file_index_spy = mocker.spy(reader_module.files, "get_production_table_files")
    handler = FileSystemModelSource(simulation_models_path)

    handler.read_production_table("sites", "1.0.0")
    handler.read_production_table("telescopes", "1.0.0")

    production_reads = [
        Path(call.kwargs["file_name"]).name
        for call in parameter_spy.call_args_list
        if "productions" in Path(call.kwargs["file_name"]).parts
    ]
    assert production_reads == ["OBS-North.json", "LSTN-01.json", "LSTN-design.json"]
    assert file_index_spy.call_count == 1


def test_file_export(simulation_models_path, tmp_test_directory):
    handler = FileSystemModelSource(simulation_models_path)
    destination = Path(tmp_test_directory) / "export"

    result = handler.export_model_files(file_names="model.dat", dest=destination)

    assert result == {"model.dat": "copied from filesystem"}
    assert (destination / "model.dat").read_text(encoding="utf-8") == "model data\n"
    assert handler.get_ecsv_file_as_astropy_table("model.ecsv")["value"][0] == pytest.approx(1.0)

    parameters = {"model_file": {"file": True, "value": "model.dat"}}
    assert handler.export_model_files(parameters=parameters, dest=destination) == {
        "model.dat": "file exists"
    }


def test_invalid_model_path_fails_without_fallback(tmp_test_directory):
    with pytest.raises(FileNotFoundError, match="Expected simulation models directory"):
        FileSystemModelSource(Path(tmp_test_directory) / "model")

    with pytest.raises(FileNotFoundError, match="path does not exist"):
        FileSystemModelSource(Path(tmp_test_directory) / "missing")


def test_missing_model_data_reports_source(simulation_models_path):
    handler = FileSystemModelSource(simulation_models_path)

    with pytest.raises(ValueError, match=r"Model version 2\.0\.0 not found"):
        handler.read_production_table("telescopes", "2.0.0")
    with pytest.raises(
        ValueError, match=r"No production table for calibration_devices in model version 1\.0\.0"
    ):
        handler.read_production_table("calibration_devices", "1.0.0")
    with pytest.raises(ValueError, match="No parameters found"):
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


def test_filesystem_source_routes_parameter_collections_and_filters(simulation_models_path):
    """Filesystem parameter lookups handle collection defaults and metadata filters."""
    handler = FileSystemModelSource(simulation_models_path)

    assert (
        handler.read_parameters({"array_layouts": "1.0.0"}, "sites", site="North")[0]["parameter"]
        == "array_layouts"
    )
    with pytest.raises(ValueError, match="requires an array element name"):
        handler.read_parameters({"array_layouts": "1.0.0"}, "sites")
    with pytest.raises(ValueError, match="No parameters found"):
        handler.read_parameters({"missing": "9.0.0"}, "telescopes", instrument="LSTN-01")
    assert handler.read_parameters(
        {"corsika_cherenkov_photon_bunch_size": "1.0.0"}, "configuration_corsika"
    )[0]["value"] == pytest.approx(5.0)

    assert not handler._matches_filters(  # pylint: disable=protected-access
        {"instrument": "LSTN-01", "site": "North"}, "MSTN-01", "North"
    )
    assert handler._matches_filters(  # pylint: disable=protected-access
        {"instrument": "LSTN-01", "site": ["North", "South"]}, "LSTN-01", "South"
    )
    assert not handler._matches_filters(  # pylint: disable=protected-access
        {"instrument": "LSTN-01", "site": ["North"]}, "LSTN-01", "South"
    )
    assert handler.query_model_parameters(
        {
            "$or": [{}, {"parameter": "camera_body_diameter", "parameter_version": "2.0.0"}],
            "instrument": "LSTN-01",
            "site": "North",
        },
        "telescopes",
    )


def test_reader_reads_file_based_simulation_configuration(simulation_models_path):
    """The source-neutral reader resolves CORSIKA and telescope configuration parameters."""
    reader = SimulationModelReader.from_files(simulation_models_path)

    assert reader.get_simulation_configuration_parameters("corsika", None, None, "1.0.0")[
        "corsika_cherenkov_photon_bunch_size"
    ]["value"] == pytest.approx(5.0)
    assert reader.get_simulation_configuration_parameters(
        "sim_telarray", "North", "LSTN-01", "1.0.0"
    )["min_photons"]["value"] == pytest.approx(2.0)


def test_model_file_export_errors(simulation_models_path, tmp_test_directory):
    handler = FileSystemModelSource(simulation_models_path)

    with pytest.raises(ValueError, match="Destination path is required"):
        handler.export_model_files(file_names="model.dat")
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        handler.export_model_files(file_names="missing.dat", dest=tmp_test_directory)
    with pytest.raises(ValueError, match="escapes model"):
        handler.export_model_files(file_names="../outside.dat", dest=tmp_test_directory)
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        handler.get_ecsv_file_as_astropy_table("missing.ecsv")
