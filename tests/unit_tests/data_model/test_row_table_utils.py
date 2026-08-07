#!/usr/bin/python3
"""Tests for row_table_utils module."""

import pytest
from astropy.units import dimensionless_unscaled, ns

from simtools.data_model import row_table_utils


def test_is_row_table_dict_valid():
    payload = {
        "columns": ["time", "amplitude"],
        "column_units": ["ns", "dimensionless"],
        "rows": [[0.0, 0.1], [1.0, 0.2]],
    }
    assert row_table_utils.is_row_table_dict(payload)


def test_is_row_table_dict_missing_key():
    payload = {
        "columns": ["time", "amplitude"],
        "rows": [[0.0, 0.1]],
    }
    assert not row_table_utils.is_row_table_dict(payload)


def test_is_row_table_dict_non_dict():
    assert not row_table_utils.is_row_table_dict("string")
    assert not row_table_utils.is_row_table_dict([1, 2, 3])
    assert not row_table_utils.is_row_table_dict(None)


def test_validate_row_table_structure_valid():
    payload = {
        "columns": ["time", "amplitude"],
        "column_units": ["ns", "dimensionless"],
        "rows": [[0.0, 0.1], [1.0, 0.2]],
    }
    assert row_table_utils.validate_row_table_structure("test_param", payload) is None


def test_validate_row_table_structure_missing_columns():
    payload = {
        "column_units": ["ns", "dimensionless"],
        "rows": [[0.0, 0.1]],
    }
    with pytest.raises(ValueError, match="'columns'"):
        row_table_utils.validate_row_table_structure("test_param", payload)


def test_validate_row_table_structure_missing_rows():
    payload = {
        "columns": ["time", "amplitude"],
        "column_units": ["ns", "dimensionless"],
    }
    with pytest.raises(ValueError, match="'rows'"):
        row_table_utils.validate_row_table_structure("test_param", payload)


def test_validate_row_table_structure_missing_column_units():
    payload = {
        "columns": ["time", "amplitude"],
        "rows": [[0.0, 0.1]],
    }
    with pytest.raises(ValueError, match="'column_units'"):
        row_table_utils.validate_row_table_structure("test_param", payload)


def test_validate_row_table_structure_column_units_length_mismatch():
    payload = {
        "columns": ["time", "amplitude"],
        "column_units": ["ns"],
        "rows": [[0.0, 0.1]],
    }
    with pytest.raises(ValueError, match="column_units length"):
        row_table_utils.validate_row_table_structure("test_param", payload)


def test_validate_row_table_structure_invalid_columns_type():
    payload = {
        "columns": "time,amplitude",
        "column_units": ["ns", "dimensionless"],
        "rows": [[0.0, 0.1]],
    }
    with pytest.raises(ValueError, match="'columns' must be a list or tuple"):
        row_table_utils.validate_row_table_structure("test_param", payload)


def test_validate_row_table_structure_invalid_rows_type():
    payload = {
        "columns": ["time", "amplitude"],
        "column_units": ["ns", "dimensionless"],
        "rows": {"time": 0.0, "amplitude": 0.1},
    }
    with pytest.raises(ValueError, match="'rows' must be a list or tuple"):
        row_table_utils.validate_row_table_structure("test_param", payload)


def test_validate_row_table_structure_non_string_column_name():
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
    payload = {
        "columns": ["time", "amplitude"],
        "column_units": ["ns", "dimensionless"],
        "rows": invalid_rows,
    }
    with pytest.raises(ValueError, match="row length"):
        row_table_utils.validate_row_table_structure("test_param", payload)


def test_validate_row_table_structure_non_numeric_value():
    payload = {
        "columns": ["time", "amplitude"],
        "column_units": ["ns", "dimensionless"],
        "rows": [["not", "numeric"]],
    }
    with pytest.raises(ValueError, match=r"non-real-numeric|non-numeric"):
        row_table_utils.validate_row_table_structure("test_param", payload)


def test_validate_row_table_structure_complex_number():
    payload = {
        "columns": ["time", "amplitude"],
        "column_units": ["ns", "dimensionless"],
        "rows": [[0.0, 1 + 2j]],
    }
    with pytest.raises(ValueError, match="non-real-numeric"):
        row_table_utils.validate_row_table_structure("test_param", payload)


def test_validate_row_table_structure_non_sequence_row():
    payload = {
        "columns": ["time", "amplitude"],
        "column_units": ["ns", "dimensionless"],
        "rows": [5.0],
    }
    with pytest.raises(ValueError, match="must be a sequence"):
        row_table_utils.validate_row_table_structure("test_param", payload)


def test_normalize_column_unit_none():
    assert row_table_utils.normalize_column_unit(None) == "dimensionless"


def test_normalize_column_unit_empty_string():
    assert row_table_utils.normalize_column_unit("") == "dimensionless"


def test_normalize_column_unit_string():
    assert row_table_utils.normalize_column_unit("ns") == "ns"
    assert row_table_utils.normalize_column_unit("km") == "km"


def test_normalize_column_unit_dimensionless_unscaled():
    assert row_table_utils.normalize_column_unit(dimensionless_unscaled) == "dimensionless"


def test_normalize_column_unit_astropy_unit():
    result = row_table_utils.normalize_column_unit(ns)
    assert isinstance(result, str)
    assert "ns" in result or result == "ns"
