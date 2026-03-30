#!/usr/bin/python3
"""Tests for row_table_utils module."""

import pytest
from astropy.units import dimensionless_unscaled, ns

from simtools.simtel import row_table_utils


class TestIsRowTableDict:
    """Test row-table dict identification."""

    def test_is_row_table_dict_valid(self):
        """Identify valid row-table dict."""
        payload = {
            "columns": ["time", "amplitude"],
            "column_units": ["ns", "dimensionless"],
            "rows": [[0.0, 0.1], [1.0, 0.2]],
        }
        assert row_table_utils.is_row_table_dict(payload)

    def test_is_row_table_dict_missing_key(self):
        """Reject incomplete dict."""
        payload = {
            "columns": ["time", "amplitude"],
            "rows": [[0.0, 0.1]],
        }
        assert not row_table_utils.is_row_table_dict(payload)

    def test_is_row_table_dict_non_dict(self):
        """Reject non-dict values."""
        assert not row_table_utils.is_row_table_dict("string")
        assert not row_table_utils.is_row_table_dict([1, 2, 3])
        assert not row_table_utils.is_row_table_dict(None)


class TestIsRowTableSchema:
    """Test row-table schema detection."""

    def test_is_row_table_schema_valid(self):
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

    def test_is_row_table_schema_missing_required(self):
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

    def test_is_row_table_schema_missing_property(self):
        """Reject schema without all properties."""
        json_schema = {
            "required": ["columns", "column_units", "rows"],
            "properties": {
                "columns": {},
                "rows": {},
            },
        }
        assert not row_table_utils.is_row_table_schema(json_schema)

    def test_is_row_table_schema_extra_keys_ok(self):
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


class TestValidateRowTableStructure:
    """Test row-table structure validation."""

    def test_validate_row_table_structure_valid(self):
        """Accept valid row-table structure."""
        payload = {
            "columns": ["time", "amplitude"],
            "column_units": ["ns", "dimensionless"],
            "rows": [[0.0, 0.1], [1.0, 0.2]],
        }
        # Should not raise
        row_table_utils.validate_row_table_structure("test_param", payload)

    def test_validate_row_table_structure_missing_columns(self):
        """Reject missing columns key."""
        payload = {
            "column_units": ["ns", "dimensionless"],
            "rows": [[0.0, 0.1]],
        }
        with pytest.raises(ValueError, match="'columns'"):
            row_table_utils.validate_row_table_structure("test_param", payload)

    def test_validate_row_table_structure_missing_rows(self):
        """Reject missing rows key."""
        payload = {
            "columns": ["time", "amplitude"],
            "column_units": ["ns", "dimensionless"],
        }
        with pytest.raises(ValueError, match="'rows'"):
            row_table_utils.validate_row_table_structure("test_param", payload)

    def test_validate_row_table_structure_missing_column_units(self):
        """Reject missing column_units key."""
        payload = {
            "columns": ["time", "amplitude"],
            "rows": [[0.0, 0.1]],
        }
        with pytest.raises(ValueError, match="'column_units'"):
            row_table_utils.validate_row_table_structure("test_param", payload)

    def test_validate_row_table_structure_column_units_length_mismatch(self):
        """Reject mismatched column_units length."""
        payload = {
            "columns": ["time", "amplitude"],
            "column_units": ["ns"],
            "rows": [[0.0, 0.1]],
        }
        with pytest.raises(ValueError, match="column_units length"):
            row_table_utils.validate_row_table_structure("test_param", payload)

    @pytest.mark.parametrize(
        "invalid_rows",
        [
            [[0.0]],  # Too few values
            [[0.0, 0.1, 0.2]],  # Too many values
        ],
    )
    def test_validate_row_table_structure_row_length_mismatch(self, invalid_rows):
        """Reject rows with incorrect length."""
        payload = {
            "columns": ["time", "amplitude"],
            "column_units": ["ns", "dimensionless"],
            "rows": invalid_rows,
        }
        with pytest.raises(ValueError, match="row length"):
            row_table_utils.validate_row_table_structure("test_param", payload)

    def test_validate_row_table_structure_non_numeric_value(self):
        """Reject non-numeric row values."""
        payload = {
            "columns": ["time", "amplitude"],
            "column_units": ["ns", "dimensionless"],
            "rows": [["not", "numeric"]],
        }
        with pytest.raises(ValueError, match=r"non-real-numeric|non-numeric"):
            row_table_utils.validate_row_table_structure("test_param", payload)

    def test_validate_row_table_structure_complex_number(self):
        """Reject complex numbers in rows."""
        payload = {
            "columns": ["time", "amplitude"],
            "column_units": ["ns", "dimensionless"],
            "rows": [[0.0, 1 + 2j]],
        }
        with pytest.raises(ValueError, match="non-real-numeric"):
            row_table_utils.validate_row_table_structure("test_param", payload)

    def test_validate_row_table_structure_non_sequence_row(self):
        """Reject non-sequence rows."""
        payload = {
            "columns": ["time", "amplitude"],
            "column_units": ["ns", "dimensionless"],
            "rows": [5.0],  # Scalar instead of sequence
        }
        with pytest.raises(ValueError, match="must be a sequence"):
            row_table_utils.validate_row_table_structure("test_param", payload)


class TestNormalizeColumnUnit:
    """Test column unit normalization."""

    def test_normalize_column_unit_none(self):
        """Convert None to dimensionless."""
        assert row_table_utils.normalize_column_unit(None) == "dimensionless"

    def test_normalize_column_unit_empty_string(self):
        """Convert empty string to dimensionless."""
        assert row_table_utils.normalize_column_unit("") == "dimensionless"

    def test_normalize_column_unit_string(self):
        """Pass through string units unchanged."""
        assert row_table_utils.normalize_column_unit("ns") == "ns"
        assert row_table_utils.normalize_column_unit("km") == "km"

    def test_normalize_column_unit_dimensionless_unscaled(self):
        """Convert astropy dimensionless to string."""
        assert row_table_utils.normalize_column_unit(dimensionless_unscaled) == "dimensionless"

    def test_normalize_column_unit_astropy_unit(self):
        """Convert astropy unit to string."""
        result = row_table_utils.normalize_column_unit(ns)
        assert isinstance(result, str)
        assert "ns" in result or result == "ns"


class TestValidateRowTableStructureParametrized:
    """Parametrized tests for various invalid row-table payloads."""

    @pytest.mark.parametrize(
        ("invalid_key", "payload"),
        [
            ("missing_columns", {"column_units": ["ns", "dimensionless"], "rows": [[0.0, 0.0]]}),
            (
                "missing_rows",
                {"columns": ["time", "amplitude"], "column_units": ["ns", "dimensionless"]},
            ),
            ("missing_column_units", {"columns": ["time", "amplitude"], "rows": [[0.0, 0.0]]}),
            (
                "column_units_mismatch",
                {"columns": ["time", "amplitude"], "column_units": ["ns"], "rows": [[0.0, 0.0]]},
            ),
            (
                "row_length_mismatch",
                {
                    "columns": ["time", "amplitude"],
                    "column_units": ["ns", "dimensionless"],
                    "rows": [[0.0]],
                },
            ),
            (
                "non_numeric",
                {
                    "columns": ["time", "amplitude"],
                    "column_units": ["ns", "dimensionless"],
                    "rows": [["not", "numeric"]],
                },
            ),
        ],
    )
    def test_validate_row_table_structure_invalid_payload(self, invalid_key, payload):
        """Test various invalid payloads with parametrization."""
        with pytest.raises(ValueError, match="Row-table"):
            row_table_utils.validate_row_table_structure("test_param", payload)
