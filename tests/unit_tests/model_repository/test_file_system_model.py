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
                "global": {
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
            "parameters": {
                "global": {"iobuf_maximum": "1.0.0"},
                "LSTN-design": {"min_photons": "1.0.0"},
            },
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
            "global/corsika_cherenkov_photon_bunch_size/"
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
            "global/corsika_particle_kinetic_energy_cutoff/"
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
        parameters / ("global/corsika_starting_grammage/corsika_starting_grammage-1.0.2.json"),
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
        parameters / "LSTN-design/min_photons/min_photons-1.0.0.json",
        _parameter("LSTN-design", "North", "min_photons", "1.0.0", 2.0),
    )
    _write_json(
        parameters / "global/iobuf_maximum/iobuf_maximum-1.0.0.json",
        _parameter(None, None, "iobuf_maximum", "1.0.0", 1000, unit="byte"),
    )
    (parameters / "model.dat").write_text("model data\n", encoding="utf-8")
    Table({"value": [1.0]}).write(parameters / "model.ecsv", format="ascii.ecsv")
    return model_root


def test_file_system_source_reads_production_descriptions(tmp_test_directory):
    """Production summaries can read info files from a filesystem source."""
    model_root = Path(tmp_test_directory) / "model-files"
    for model_version, suffix, description in (
        ("1.0.0", "yml", "Prod1"),
        ("2.0.0", "yaml", "Prod2"),
    ):
        info_file = (
            model_root / "simulation-models" / "productions" / model_version / f"info.{suffix}"
        )
        info_file.parent.mkdir(parents=True, exist_ok=True)
        info_file.write_text(f"description: {description}\n", encoding="utf-8")
    (model_root / "simulation-models" / "model_parameters").mkdir(parents=True)

    source = FileSystemModelSource(model_root)

    assert source.get_production_descriptions() == [("1.0.0", "Prod1"), ("2.0.0", "Prod2")]


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


def test_file_export_qualifies_ecsv_with_instrument(
    simulation_models_path, tmp_test_directory, mocker
):
    """ECSV model assets always use their instrument-qualified export name."""
    source = (
        Path(simulation_models_path)
        / "simulation-models/model_parameters/LSTN-design/selected/values.ecsv"
    )
    source.parent.mkdir(parents=True, exist_ok=True)
    table = Table({"value": [1.0]})
    table.meta.update(
        {
            "parameter_name": "selected",
            "parameter_version": "1.0.0",
            "instrument": "LSTN-design",
            "site": "North",
        }
    )
    table.write(source, format="ascii.ecsv")

    handler = FileSystemModelSource(simulation_models_path)
    mocker.patch.object(handler, "get_parameter_table")
    destination = Path(tmp_test_directory) / "export"
    parameter = {
        "file": True,
        "instrument": "LSTN-design",
        "parameter": "selected",
        "parameter_version": "1.0.0",
        "site": "North",
        "model_parameter_schema_version": "0.1.0",
        "value": "values.ecsv",
    }

    result = handler.export_model_files(parameters={"selected": parameter}, dest=destination)

    assert result == {"values-LSTN-design.ecsv": "copied from filesystem"}
    assert parameter["value"] == "values-LSTN-design.ecsv"
    assert parameter["_simtools_export_source_value"] == "values.ecsv"
    assert (destination / "values-LSTN-design.ecsv").is_file()


def test_invalid_model_path_fails_without_fallback(tmp_test_directory):
    with pytest.raises(FileNotFoundError, match="Expected simulation models directory"):
        FileSystemModelSource(Path(tmp_test_directory) / "model")

    with pytest.raises(FileNotFoundError, match="path does not exist"):
        FileSystemModelSource(Path(tmp_test_directory) / "missing")


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
    with pytest.raises(ValueError, match="escapes parameter"):
        handler.export_model_files(file_names="../outside.dat", dest=tmp_test_directory)
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        handler.get_ecsv_file_as_astropy_table("missing.ecsv")
