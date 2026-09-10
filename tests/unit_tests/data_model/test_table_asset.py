"""Tests for ECSV model-asset resolution and validation."""

from pathlib import Path

import astropy.units as u
import pytest
from astropy.table import QTable, Table

from simtools.data_model.table_asset import (
    get_simtel_serialization,
    read_ecsv_asset,
    resolve_asset_path,
    validate_table_asset,
)


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


def test_resolve_asset_rejects_absolute_path(tmp_test_directory):
    parameter_file = Path(tmp_test_directory) / "parameter.json"

    with pytest.raises(ValueError, match="must be relative"):
        resolve_asset_path("/tmp/table.ecsv", parameter_file)


def test_read_ecsv_asset_rejects_non_ecsv(tmp_test_directory):
    path = Path(tmp_test_directory) / "table.dat"
    path.write_text("not an ECSV file", encoding="utf-8")

    with pytest.raises(ValueError, match="must use ECSV"):
        read_ecsv_asset(path)


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


def test_validate_table_asset_converts_plain_table():
    table = Table({"time": [0.0, 1.0], "amplitude": [0.0, 1.0]})
    table["time"].info.description = "Time"
    table["amplitude"].info.description = "Amplitude"

    result = validate_table_asset(table)

    assert isinstance(result, QTable)


def test_validate_table_asset_rejects_missing_and_unexpected_columns():
    table = _table()
    schema_entry = {"table_columns": [{"name": "time", "required": True, "description": "Time"}]}

    with pytest.raises(ValueError, match="Unexpected table columns"):
        validate_table_asset(table, schema_entry=schema_entry)

    with pytest.raises(ValueError, match="Missing required table columns"):
        validate_table_asset(QTable({"amplitude": [1.0]}), schema_entry=schema_entry)


def test_validate_table_asset_allows_extra_described_columns():
    table = _table()
    schema_entry = {
        "allow_extra_columns": True,
        "table_columns": [{"name": "time", "required": True, "description": "Time"}],
    }

    result = validate_table_asset(table, schema_entry=schema_entry)

    assert result.colnames == ["time", "amplitude"]


def test_validate_table_asset_rejects_undescribed_column():
    table = _table()
    table["extra"] = [1, 2]
    table["extra"].info.description = None

    with pytest.raises(ValueError, match=r"extra.*no description"):
        validate_table_asset(table, schema_entry={"allow_extra_columns": True})


def test_validate_table_asset_rejects_missing_unit():
    table = QTable({"time": [0.0, 1.0]})
    table["time"].info.description = "Time"
    entry = {"name": "time", "unit": "ns", "description": "Time"}

    with pytest.raises(ValueError, match="missing unit"):
        validate_table_asset(table, schema_entry={"table_columns": [entry]})


def test_validate_table_asset_rejects_invalid_unit():
    table = _table()
    entry = {"name": "time", "unit": "invalid", "description": "Time"}

    with pytest.raises(ValueError, match="expected invalid"):
        validate_table_asset(
            table,
            schema_entry={"allow_extra_columns": True, "table_columns": [entry]},
        )


def test_validate_table_asset_rejects_nonfinite_and_out_of_range_values():
    table = _table()
    table["amplitude"][0] = float("nan")
    entry = {"name": "amplitude", "description": "Amplitude", "type": "float64"}

    with pytest.raises(ValueError, match="NaN or infinite"):
        validate_table_asset(
            table,
            schema_entry={"allow_extra_columns": True, "table_columns": [entry]},
        )

    table["amplitude"][0] = 2.0
    entry["allowed_range"] = {"max": 1.0}
    with pytest.raises(ValueError, match=r"allowed_range\.max"):
        validate_table_asset(
            table,
            schema_entry={"allow_extra_columns": True, "table_columns": [entry]},
        )


def test_validate_table_asset_allows_nan_when_declared():
    table = _table()
    table["amplitude"][0] = float("nan")
    entry = {
        "name": "amplitude",
        "description": "Amplitude",
        "type": "float64",
        "input_processing": ["allow_nan"],
    }

    validate_table_asset(
        table,
        schema_entry={"allow_extra_columns": True, "table_columns": [entry]},
    )


def test_validate_table_asset_rejects_dtype_and_string_mismatch():
    table = _table()
    entry = {"name": "time", "description": "Time", "type": "string"}

    with pytest.raises(ValueError, match="must contain strings"):
        validate_table_asset(
            table,
            schema_entry={"allow_extra_columns": True, "table_columns": [entry]},
        )

    strings = QTable({"time": [1, 2]})
    strings["time"].info.description = "Time"
    entry["type"] = "int8"
    with pytest.raises(ValueError, match="has dtype"):
        validate_table_asset(
            strings,
            schema_entry={"allow_extra_columns": True, "table_columns": [entry]},
        )


def test_validate_table_asset_rejects_unsorted_and_duplicate_data():
    table = _table()
    table["time"] = [1.0, 0.0] * u.ns
    table["time"].info.description = "Time"
    entry = {"name": "time", "description": "Time", "input_processing": ["sort"]}

    with pytest.raises(ValueError, match="not sorted"):
        validate_table_asset(
            table,
            schema_entry={"allow_extra_columns": True, "table_columns": [entry]},
        )

    table = _table()
    table["time"][1] = table["time"][0]
    entry = {"name": "time", "description": "Time", "input_processing": ["remove_duplicates"]}
    with pytest.raises(ValueError, match="Duplicate values"):
        validate_table_asset(
            table,
            schema_entry={"allow_extra_columns": True, "table_columns": [entry]},
        )


def test_validate_table_asset_rejects_identity_mismatch_and_metadata_type():
    table = _table()
    parameter_data = {
        "parameter": "wrong",
        "parameter_version": "1.0.0",
        "instrument": "LSTN-01",
        "site": "North",
    }
    with pytest.raises(ValueError, match="does not match"):
        validate_table_asset(table, parameter_data=parameter_data)

    schema_entry = {"table_metadata": [{"name": "source_file_name", "type": "number"}]}
    with pytest.raises(ValueError, match="not of type number"):
        validate_table_asset(table, schema_entry=schema_entry)


def test_get_simtel_serialization_returns_contract():
    schema_dict = {
        "name": "pulse",
        "data": [{"table_columns": [{"name": "time", "unit": "ns"}]}],
        "simulation_software": [
            {
                "name": "sim_telarray",
                "table_format": "pulse",
                "serialization": {
                    "columns": ["time"],
                    "row_sort_keys": ["time"],
                    "float_format": ".3f",
                    "write_comments": False,
                },
            }
        ],
    }

    result = get_simtel_serialization(schema_dict)

    assert result["table_format"] == "pulse"
    assert result["units"] == {"time": "ns"}


@pytest.mark.parametrize(
    "schema_dict",
    [
        {"name": "missing"},
        {"name": "missing", "simulation_software": [{"name": "sim_telarray"}]},
        {
            "name": "missing",
            "simulation_software": [{"name": "sim_telarray", "table_format": "plain"}],
        },
    ],
)
def test_get_simtel_serialization_rejects_incomplete_schema(schema_dict):
    with pytest.raises(ValueError, match="serialization"):
        get_simtel_serialization(schema_dict)


def test_get_simtel_serialization_uses_fallback_software_entry():
    schema = {
        "name": "pulse",
        "data": [],
        "simulation_software": [
            {
                "name": "other",
                "table_format": "plain",
                "serialization": {
                    "columns": ["time"],
                    "row_sort_keys": [],
                    "float_format": ".3f",
                    "write_comments": False,
                },
            }
        ],
    }

    assert get_simtel_serialization(schema)["table_format"] == "plain"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda contract: contract.update(columns=[]), "unique and non-empty"),
        (lambda contract: contract.update(columns=["time", "time"]), "unique and non-empty"),
        (lambda contract: contract.update(allowed_columns=[]), "allowed_columns"),
        (lambda contract: contract.update(row_sort_keys=["missing"]), "sort keys"),
        (lambda contract: contract.update(matrix_axes=["time"]), "matrix_axes"),
        (lambda contract: contract.update(matrix_axes=["time", "missing"]), "matrix axes"),
        (lambda contract: contract.update(value_column="missing"), "value_column"),
    ],
)
def test_serialization_contract_rejects_invalid_references(change, message):
    from simtools.data_model.table_asset import _validate_serialization_contract

    contract = {
        "table_format": "plain",
        "columns": ["time"],
        "row_sort_keys": ["time"],
        "float_format": ".3f",
        "write_comments": False,
        "units": {"time": "ns"},
    }
    change(contract)

    with pytest.raises(ValueError, match=message):
        _validate_serialization_contract(contract)


def test_validate_table_asset_handles_string_and_unknown_dtypes():
    table = QTable({"label": ["a", "b"]})
    table["label"].info.description = "Label"
    validate_table_asset(
        table,
        schema_entry={
            "table_columns": [{"name": "label", "description": "Label", "type": "string"}]
        },
    )

    entry = {"name": "label", "description": "Label", "type": "not-a-dtype"}
    validate_table_asset(table, schema_entry={"table_columns": [entry]})


def test_validate_table_asset_validates_minimum_range_and_sorted_data():
    table = _table()
    entry = {
        "name": "amplitude",
        "description": "Amplitude",
        "type": "float64",
        "required_range": {"min": 0.5, "max": 1.5},
    }
    with pytest.raises(ValueError, match=r"required_range\.min"):
        validate_table_asset(
            table,
            schema_entry={"allow_extra_columns": True, "table_columns": [entry]},
        )

    table["amplitude"][0] = 0.5
    validate_table_asset(
        table,
        schema_entry={"allow_extra_columns": True, "table_columns": [entry]},
    )


def test_validate_table_asset_reports_missing_optional_metadata_and_site_lists():
    table = _table()
    validate_table_asset(
        table,
        schema_entry={"table_metadata": [{"name": "optional", "type": "string"}]},
    )
    table.meta["site"] = ["North", "South"]
    validate_table_asset(
        table,
        schema_entry={"table_metadata": [{"name": "site", "type": "string"}]},
    )


def test_validate_table_asset_rejects_missing_required_metadata():
    table = _table()
    del table.meta["source_file_name"]

    with pytest.raises(ValueError, match="Missing required table metadata"):
        validate_table_asset(
            table,
            schema_entry={"table_metadata": [{"name": "source_file_name", "required": True}]},
        )
