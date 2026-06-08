#!/usr/bin/python3
"""Tests for row_table_utils module."""

import pytest
from astropy.units import dimensionless_unscaled, ns

from simtools.data_model import row_table_utils


def test_is_row_table_dict_valid():
    """Identify valid row-table dict."""
    payload = {
        "columns": ["time", "amplitude"],
        "column_units": ["ns", "dimensionless"],
        "rows": [[0.0, 0.1], [1.0, 0.2]],
    }
    assert row_table_utils.is_row_table_dict(payload)


def test_is_row_table_dict_missing_key():
    """Reject incomplete dict."""
    payload = {
        "columns": ["time", "amplitude"],
        "rows": [[0.0, 0.1]],
    }
    assert not row_table_utils.is_row_table_dict(payload)


def test_is_row_table_dict_non_dict():
    """Reject non-dict values."""
    assert not row_table_utils.is_row_table_dict("string")
    assert not row_table_utils.is_row_table_dict([1, 2, 3])
    assert not row_table_utils.is_row_table_dict(None)


def test_is_row_table_schema_valid():
    """Identify valid row-table JSON schema."""
    json_schema = {
        "required": ["columns", "column_units", "rows"],
        "properties": {
            "columns": {},
            "column_units": {},
            "rows": {},
        },
    }
    assert row_table_utils.is_row_table_schema(json_schema)


def test_is_row_table_schema_missing_required():
    """Reject schema without all required keys."""
    json_schema = {
        "required": ["columns", "rows"],
        "properties": {
            "columns": {},
            "column_units": {},
            "rows": {},
        },
    }
    assert not row_table_utils.is_row_table_schema(json_schema)


def test_is_row_table_schema_missing_property():
    """Reject schema without all properties."""
    json_schema = {
        "required": ["columns", "column_units", "rows"],
        "properties": {
            "columns": {},
            "rows": {},
        },
    }
    assert not row_table_utils.is_row_table_schema(json_schema)


def test_is_row_table_schema_extra_keys_ok():
    """Accept schema with extra keys alongside required ones."""
    json_schema = {
        "required": ["columns", "column_units", "rows"],
        "properties": {
            "columns": {},
            "column_units": {},
            "rows": {},
            "extra_field": {},
        },
    }
    assert row_table_utils.is_row_table_schema(json_schema)


def test_validate_row_table_structure_valid():
    """Accept valid row-table structure."""
    payload = {
        "columns": ["time", "amplitude"],
        "column_units": ["ns", "dimensionless"],
        "rows": [[0.0, 0.1], [1.0, 0.2]],
    }
    row_table_utils.validate_row_table_structure("test_param", payload)


def test_validate_row_table_structure_missing_columns():
    """Reject missing columns key."""
    payload = {
        "column_units": ["ns", "dimensionless"],
        "rows": [[0.0, 0.1]],
    }
    with pytest.raises(ValueError, match="'columns'"):
        row_table_utils.validate_row_table_structure("test_param", payload)


def test_validate_row_table_structure_missing_rows():
    """Reject missing rows key."""
    payload = {
        "columns": ["time", "amplitude"],
        "column_units": ["ns", "dimensionless"],
    }
    with pytest.raises(ValueError, match="'rows'"):
        row_table_utils.validate_row_table_structure("test_param", payload)


def test_validate_row_table_structure_missing_column_units():
    """Reject missing column_units key."""
    payload = {
        "columns": ["time", "amplitude"],
        "rows": [[0.0, 0.1]],
    }
    with pytest.raises(ValueError, match="'column_units'"):
        row_table_utils.validate_row_table_structure("test_param", payload)


def test_validate_row_table_structure_column_units_length_mismatch():
    """Reject mismatched column_units length."""
    payload = {
        "columns": ["time", "amplitude"],
        "column_units": ["ns"],
        "rows": [[0.0, 0.1]],
    }
    with pytest.raises(ValueError, match="column_units length"):
        row_table_utils.validate_row_table_structure("test_param", payload)


def test_validate_row_table_structure_invalid_columns_type():
    """Reject columns when not list or tuple."""
    payload = {
        "columns": "time,amplitude",
        "column_units": ["ns", "dimensionless"],
        "rows": [[0.0, 0.1]],
    }
    with pytest.raises(ValueError, match="'columns' must be a list or tuple"):
        row_table_utils.validate_row_table_structure("test_param", payload)


def test_validate_row_table_structure_invalid_rows_type():
    """Reject rows when not list or tuple."""
    payload = {
        "columns": ["time", "amplitude"],
        "column_units": ["ns", "dimensionless"],
        "rows": {"time": 0.0, "amplitude": 0.1},
    }
    with pytest.raises(ValueError, match="'rows' must be a list or tuple"):
        row_table_utils.validate_row_table_structure("test_param", payload)


def test_validate_row_table_structure_non_string_column_name():
    """Reject non-string column names."""
    payload = {
        "columns": ["time", 1],
        "column_units": ["ns", "dimensionless"],
        "rows": [[0.0, 0.1]],
    }
    with pytest.raises(ValueError, match="all column names"):
        row_table_utils.validate_row_table_structure("test_param", payload)


@pytest.mark.parametrize(
    "invalid_rows",
    [
        [[0.0]],
        [[0.0, 0.1, 0.2]],
    ],
)
def test_validate_row_table_structure_row_length_mismatch(invalid_rows):
    """Reject rows with incorrect length."""
    payload = {
        "columns": ["time", "amplitude"],
        "column_units": ["ns", "dimensionless"],
        "rows": invalid_rows,
    }
    with pytest.raises(ValueError, match="row length"):
        row_table_utils.validate_row_table_structure("test_param", payload)


def test_validate_row_table_structure_non_numeric_value():
    """Reject non-numeric row values."""
    payload = {
        "columns": ["time", "amplitude"],
        "column_units": ["ns", "dimensionless"],
        "rows": [["not", "numeric"]],
    }
    with pytest.raises(ValueError, match=r"non-real-numeric|non-numeric"):
        row_table_utils.validate_row_table_structure("test_param", payload)


def test_validate_row_table_structure_complex_number():
    """Reject complex numbers in rows."""
    payload = {
        "columns": ["time", "amplitude"],
        "column_units": ["ns", "dimensionless"],
        "rows": [[0.0, 1 + 2j]],
    }
    with pytest.raises(ValueError, match="non-real-numeric"):
        row_table_utils.validate_row_table_structure("test_param", payload)


def test_validate_row_table_structure_non_sequence_row():
    """Reject non-sequence rows."""
    payload = {
        "columns": ["time", "amplitude"],
        "column_units": ["ns", "dimensionless"],
        "rows": [5.0],
    }
    with pytest.raises(ValueError, match="must be a sequence"):
        row_table_utils.validate_row_table_structure("test_param", payload)


def test_normalize_column_unit_none():
    """Convert None to dimensionless."""
    assert row_table_utils.normalize_column_unit(None) == "dimensionless"


def test_normalize_column_unit_empty_string():
    """Convert empty string to dimensionless."""
    assert row_table_utils.normalize_column_unit("") == "dimensionless"


def test_normalize_column_unit_string():
    """Pass through string units unchanged."""
    assert row_table_utils.normalize_column_unit("ns") == "ns"
    assert row_table_utils.normalize_column_unit("km") == "km"


def test_normalize_column_unit_dimensionless_unscaled():
    """Convert astropy dimensionless to string."""
    assert row_table_utils.normalize_column_unit(dimensionless_unscaled) == "dimensionless"


def test_normalize_column_unit_astropy_unit():
    """Convert astropy unit to string."""
    result = row_table_utils.normalize_column_unit(ns)
    assert isinstance(result, str)
    assert "ns" in result or result == "ns"
