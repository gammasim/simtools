"""Tests for ECSV model-asset resolution and validation."""

from pathlib import Path

import astropy.units as u
import pytest
from astropy.table import QTable

from simtools.data_model.table_asset import read_ecsv_asset, resolve_asset_path


def _table():
    table = QTable({"time": [0.0, 1.0], "amplitude": [0.0, 1.0]})
    table["time"].unit = u.ns
    table["time"].info.description = "Time"
    table["amplitude"].info.description = "Amplitude"
    table.meta.update(
        {
            "parameter_name": "fadc_pulse_shape",
            "parameter_version": "1.0.0",
            "instrument": "LSTN-01",
            "site": "North",
            "source_file_name": "pulse.dat",
            "original_comments": [],
            "conversion_tool": "test",
        }
    )
    return table


def test_resolve_parameter_relative_path(tmp_test_directory):
    parameter_file = Path(tmp_test_directory) / "par" / "parameter.json"
    assert (
        resolve_asset_path("table.ecsv", parameter_file)
        == (parameter_file.parent / "table.ecsv").resolve()
    )


def test_resolve_asset_rejects_traversal(tmp_test_directory):
    parameter_file = Path(tmp_test_directory) / "par" / "parameter.json"
    with pytest.raises(ValueError, match="escapes parameter directory"):
        resolve_asset_path("../table.ecsv", parameter_file)


def test_read_ecsv_asset_validates_schema_and_identity(tmp_test_directory):
    table = _table()
    path = Path(tmp_test_directory) / "table.ecsv"
    table.write(path, format="ascii.ecsv")
    schema_entry = {
        "allow_extra_columns": True,
        "table_columns": [
            {
                "name": "time",
                "description": "Time",
                "type": "float64",
                "unit": "ns",
                "required": True,
            },
            {
                "name": "amplitude",
                "description": "Amplitude",
                "type": "float64",
                "unit": "dimensionless",
                "required": True,
            },
        ],
        "table_metadata": [
            {"name": "parameter_name", "description": "name", "type": "string", "required": True},
            {
                "name": "original_comments",
                "description": "comments",
                "type": "array",
                "required": True,
            },
        ],
    }
    result = read_ecsv_asset(
        path,
        schema_entry=schema_entry,
        parameter_data={
            "parameter": "fadc_pulse_shape",
            "parameter_version": "1.0.0",
            "instrument": "LSTN-01",
            "site": "North",
        },
    )
    assert result.colnames == ["time", "amplitude"]


def test_read_ecsv_asset_requires_standard_model_parameter_metadata(tmp_test_directory):
    table = _table()
    del table.meta["site"]
    path = Path(tmp_test_directory) / "table.ecsv"
    table.write(path, format="ascii.ecsv")

    with pytest.raises(ValueError, match="Missing required ECSV metadata: site"):
        read_ecsv_asset(
            path,
            parameter_data={
                "parameter": "fadc_pulse_shape",
                "parameter_version": "1.0.0",
                "instrument": "LSTN-01",
                "site": "North",
            },
        )


def test_read_ecsv_asset_rejects_missing_description(tmp_test_directory):
    table = _table()
    table["amplitude"].info.description = None
    path = Path(tmp_test_directory) / "table.ecsv"
    table.write(path, format="ascii.ecsv")
    with pytest.raises(ValueError, match="no description"):
        read_ecsv_asset(path)


def test_read_ecsv_asset_rejects_noncanonical_unit(tmp_test_directory):
    table = _table()
    table["time"] = table["time"].to(u.us)
    path = Path(tmp_test_directory) / "table.ecsv"
    table.write(path, format="ascii.ecsv")
    schema_entry = {
        "allow_extra_columns": True,
        "table_columns": [
            {
                "name": "time",
                "description": "Time",
                "type": "float64",
                "unit": "ns",
                "required": True,
            }
        ],
    }
    with pytest.raises(ValueError, match="expected ns"):
        read_ecsv_asset(path, schema_entry=schema_entry)
