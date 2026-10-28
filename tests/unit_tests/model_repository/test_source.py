"""Tests for the storage-neutral simulation-model source."""

import json
from pathlib import Path

import pytest
from astropy.table import Column, Table

from simtools.model_repository.source import ModelSource
from simtools.model_repository.storage import MemoryStorage


def _json(value):
    """Encode one repository document."""
    return json.dumps(value).encode()


def _parameter(instrument, value):
    """Return a normalized model parameter document."""
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


def test_source_warms_a_model_version_and_reads_parameters():
    """Production and referenced parameter files are cached on first use."""
    production = "simulation-models/productions/1.0.0/LSTN-01.json"
    parameter_path = (
        "simulation-models/model_parameters/LSTN-01/camera_body_diameter/"
        "camera_body_diameter-1.0.0.json"
    )
    files = {
        production: _json({"parameters": {"LSTN-01": {"camera_body_diameter": "1.0.0"}}}),
        parameter_path: _parameter("LSTN-01", 350.0),
    }
    storage = MemoryStorage(files)
    source = ModelSource(storage)

    table = source.read_production_table("telescopes", "1.0.0")
    assert source.is_configured() is True
    assert source.read_production_table("telescopes", "1.0.0") == table
    first_reads = set(storage.files)
    parameter = source.read_parameters(
        {"camera_body_diameter": "1.0.0"}, "telescopes", "LSTN-01", "North"
    )

    assert table["parameters"]["LSTN-01"]["camera_body_diameter"] == "1.0.0"
    assert parameter[0]["value"] == pytest.approx(350.0)
    assert first_reads == set(files)
    assert source.source_name == "memory"
    assert source.source_config == {"type": "memory", "files": files}


def test_source_combines_patch_history_and_descriptions():
    """Patch versions include earlier production documents and info descriptions."""
    storage = MemoryStorage(
        {
            "simulation-models/productions/1.0.0/LSTN-design.json": _json(
                {"parameters": {"LSTN-design": {"old_parameter": "1.0.0"}}}
            ),
            "simulation-models/productions/1.1.0/info.yml": (
                b"description: Updated\nmodel_update: patch_update\n"
                b"model_version_history:\n  - 1.0.0\n"
            ),
            "simulation-models/productions/1.1.0/LSTN-01.json": _json(
                {"parameters": {"LSTN-01": {"new_parameter": "1.0.0"}}}
            ),
        }
    )
    source = ModelSource(storage)

    table = source.read_production_table("telescopes", "1.1.0")

    assert table["parameters"] == {
        "LSTN-design": {"old_parameter": "1.0.0"},
        "LSTN-01": {"new_parameter": "1.0.0"},
    }
    assert source.get_model_versions() == ["1.0.0", "1.1.0"]
    assert source.get_production_descriptions() == [("1.1.0", "Updated")]


def test_source_rejects_missing_patch_history_version():
    """A patch release must not silently drop a version listed in its history."""
    storage = MemoryStorage(
        {
            "simulation-models/productions/1.1.0/info.yml": (
                b"description: Updated\nmodel_update: patch_update\n"
                b"model_version_history:\n  - 1.0.0\n"
            ),
            "simulation-models/productions/1.1.0/LSTN-01.json": _json(
                {"parameters": {"LSTN-01": {"new_parameter": "1.0.0"}}}
            ),
        }
    )

    with pytest.raises(ValueError, match=r"1\.0\.0 in the patch history of 1\.1\.0 not found"):
        ModelSource(storage).read_production_table("telescopes", "1.1.0")


def test_source_reports_missing_tables_and_parameters():
    """Missing versions and scoped parameter data have actionable errors."""
    source = ModelSource(MemoryStorage({}))

    with pytest.raises(ValueError, match=r"Model version 2\.0\.0 not found"):
        source.read_production_table("telescopes", "2.0.0")
    with pytest.raises(ValueError, match="requires an array element name"):
        source.read_parameters({"x": "1.0.0"}, "telescopes")


@pytest.mark.parametrize(
    ("query", "collection", "instrument"),
    [
        ({"instrument": "LSTN-01"}, "telescopes", "LSTN-01"),
        ({"site": "North"}, "sites", "OBS-North"),
        ({}, "configuration_corsika", "global"),
    ],
)
def test_source_resolves_parameter_scope(query, collection, instrument):
    """Parameter lookup resolves element, site, and global scopes."""
    assert ModelSource._get_parameter_instrument(query, collection) == instrument


def test_source_filters_parameter_metadata_and_rejects_missing_values():
    """Parameter reads apply instrument and site filters and report no matches."""
    path = (
        "simulation-models/model_parameters/LSTN-01/camera_body_diameter/"
        "camera_body_diameter-1.0.0.json"
    )
    source = ModelSource(
        MemoryStorage(
            {
                path: _parameter("LSTN-01", 350.0),
                "simulation-models/model_parameters/MSTN-01/camera_body_diameter/"
                "camera_body_diameter-1.0.0.json": _parameter("MSTN-01", 370.0),
            }
        )
    )

    assert (
        source.read_parameters({"camera_body_diameter": "1.0.0"}, "telescopes", "LSTN-01", "North")[
            0
        ]["instrument"]
        == "LSTN-01"
    )
    with pytest.raises(ValueError, match="No parameters found"):
        source.read_parameters({"camera_body_diameter": "1.0.0"}, "telescopes", "LSTN-01", "South")
    with pytest.raises(ValueError, match="No parameters found"):
        source.read_parameters({"missing": "1.0.0"}, "telescopes", "LSTN-01", "North")


def test_source_reads_ecsv_with_parameter_metadata_and_checks_asset_paths(tmp_test_directory):
    """ECSV reads resolve within parameter metadata and validate the table schema."""
    asset_path = "simulation-models/model_parameters/LSTN-01/camera_trigger_groups/groups.ecsv"
    output = tmp_test_directory / "groups.ecsv"
    table = Table(
        {
            "group_id": Column([1, 2], description="Trigger group identifier"),
            "group_order": Column([0, 1], description="Trigger group order"),
            "kind": Column(["majority", "sum"], description="Trigger kind"),
            "use_default_multiplicity": Column([1, 1], description="Use default multiplicity"),
        }
    )
    table.meta.update(
        {
            "parameter_name": "camera_trigger_groups",
            "parameter_version": "1.0.0",
            "instrument": "LSTN-01",
            "site": "North",
        }
    )
    table.write(output, format="ascii.ecsv")
    parameter_data = {
        "instrument": "LSTN-01",
        "parameter": "camera_trigger_groups",
        "parameter_version": "1.0.0",
        "site": "North",
        "value": "groups.ecsv",
    }
    source = ModelSource(MemoryStorage({asset_path: Path(output).read_bytes()}))

    assert source.get_parameter_table(parameter_data)["group_id"].tolist() == [1, 2]
    with pytest.raises(ValueError, match="does not reference an ECSV table"):
        source.get_parameter_table({**parameter_data, "value": "groups.dat"})
    with pytest.raises(ValueError, match="escapes parameter directory"):
        source.get_parameter_table({**parameter_data, "value": "../groups.ecsv"})
    with pytest.raises(ValueError, match="requires parameter, parameter_version"):
        source.get_parameter_table({"value": "groups.ecsv"})
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        source.get_parameter_table({**parameter_data, "value": "missing.ecsv"})


def test_source_exports_parameter_assets_without_overwriting(tmp_test_directory):
    """Assets are copied from memory and colliding different contents are qualified."""
    storage = MemoryStorage(
        {
            "simulation-models/model_parameters/LSTN-01/data/model.dat": b"one",
            "simulation-models/model_parameters/MSTN-01/data/model.dat": b"two",
        }
    )
    source = ModelSource(storage)
    destination = tmp_test_directory / "export"
    first = {
        "file": True,
        "instrument": "LSTN-01",
        "parameter": "data",
        "parameter_version": "1.0.0",
        "value": "model.dat",
    }
    second = {**first, "instrument": "MSTN-01"}

    assert source.export_model_files({"data": first}, dest=destination) == {
        "model.dat": "copied from model source"
    }
    assert source.export_model_files({"data": second}, dest=destination) == {
        "model-MSTN-01.dat": "copied from model source"
    }
    assert Path(str(destination / "model.dat")).read_bytes() == b"one"
    assert Path(str(destination / "model-MSTN-01.dat")).read_bytes() == b"two"


def test_source_export_reuses_identical_files_and_rejects_ambiguous_exports(
    tmp_test_directory,
):
    """Identical assets are reused while file-name-only exports require metadata."""
    asset_path = "simulation-models/model_parameters/LSTN-01/data/model.dat"
    source = ModelSource(MemoryStorage({asset_path: b"same"}))
    destination = tmp_test_directory / "export"
    parameter = {
        "file": True,
        "instrument": "LSTN-01",
        "parameter": "data",
        "parameter_version": "1.0.0",
        "value": "model.dat",
    }

    source.export_model_files({"data": parameter}, dest=destination)
    assert source.export_model_files({"data": parameter}, dest=destination) == {
        "model.dat": "file exists"
    }
    with pytest.raises(ValueError, match="requires parameter metadata"):
        source.export_model_files(file_names="model.dat", dest=destination)
    with pytest.raises(ValueError, match="Destination path is required"):
        source.export_model_files(parameters={"data": parameter})


def test_source_export_skips_file_parameters_without_value(tmp_test_directory):
    """File parameters without a value (no file configured) are not exported."""
    source = ModelSource(MemoryStorage({}))
    parameters = {
        "unset_file": {"file": True, "value": None, "parameter": "unset_file"},
        "empty_file": {"file": True, "value": "", "parameter": "empty_file"},
    }

    assert source.export_model_files(parameters=parameters, dest=tmp_test_directory) == {}
