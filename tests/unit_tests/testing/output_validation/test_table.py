"""Tests for ECSV table validators."""

from pathlib import Path

import pytest
import yaml
from astropy import units as u
from astropy.table import Table

from simtools.testing.output_validation import table


def _write_table(path, metadata=None):
    """Write a small ECSV table for validation tests."""
    output = Table({"id": [1, 2], "value": [1.0, 2.0], "label": ["a", "b"]})
    output["value"].unit = u.m
    output.meta = metadata or {"summary": {"rows": 2, "total": 3.0}}
    output.write(path, format="ascii.ecsv", overwrite=True)


def _write_schema(path):
    """Write a minimal data-product schema."""
    schema = {
        "schema_version": "0.1.0",
        "data": [
            {
                "type": "data_table",
                "table_columns": [
                    {"name": "id", "required": True, "type": "int64"},
                    {"name": "value", "required": True, "type": "float64", "unit": "m"},
                    {"name": "label", "required": True, "type": "string"},
                ],
            }
        ],
    }
    path.write_text(yaml.safe_dump(schema), encoding="utf-8")


def test_read_table_reports_parse_failure(tmp_test_directory):
    """Wrap parse errors with the output path."""
    output = Path(tmp_test_directory) / "table.ecsv"
    output.write_text("not an ECSV table\n", encoding="utf-8")

    with pytest.raises(AssertionError, match="not a parseable ECSV table"):
        table.read_table(output)


def test_table_and_metadata_validators(tmp_test_directory):
    """Validate table rows, columns, and metadata relations."""
    output = Path(tmp_test_directory) / "table.ecsv"
    _write_table(output)
    table.validate_table(
        output,
        {
            "minimum_rows": 2,
            "unique_columns": ["id"],
            "columns": {
                "label": {"allowed_values": ["a", "b"]},
                "value": {"range": {"minimum": 1.0, "maximum": 2.0, "unit": "m"}},
            },
        },
    )
    table.validate_metadata(
        output,
        {
            "required_keys": ["summary"],
            "relations": [
                {"left": "summary.rows", "equals": "table.row_count"},
                {"left": "summary.total", "equals": "table.column_sum", "column": "value"},
            ],
        },
    )


@pytest.mark.parametrize(
    ("rule", "message"),
    [
        ({"minimum_rows": 3}, "rows"),
        ({"columns": {"label": {"allowed_values": ["x"]}}}, "outside"),
        ({"columns": {"value": {"range": {"minimum": 3.0, "unit": "m"}}}}, "minimum"),
    ],
)
def test_table_validator_failures(tmp_test_directory, rule, message):
    """Report invalid table rows and column values."""
    output = Path(tmp_test_directory) / "table.ecsv"
    _write_table(output)
    with pytest.raises(AssertionError, match=message):
        table.validate_table(output, rule)


def test_metadata_validator_failures(tmp_test_directory):
    """Report missing metadata and invalid metadata relations."""
    output = Path(tmp_test_directory) / "table.ecsv"
    _write_table(output)
    with pytest.raises(AssertionError, match="no metadata key"):
        table.validate_metadata(output, {"required_keys": ["missing"]})
    with pytest.raises(AssertionError, match="expected"):
        table.validate_metadata(
            output,
            {"relations": [{"left": "summary.total", "equals": "table.row_count"}]},
        )


def test_data_schema_validator(tmp_test_directory):
    """Validate a table against a data-product schema."""
    output = Path(tmp_test_directory) / "table.ecsv"
    schema_file = Path(tmp_test_directory) / "table.schema.yml"
    _write_table(output)
    _write_schema(schema_file)
    table.validate_data_schema(output, schema_file)
    Table({"other": [1]}).write(output, format="ascii.ecsv", overwrite=True)
    with pytest.raises(AssertionError, match="data-product schema"):
        table.validate_data_schema(output, schema_file)


def test_table_validator_rejects_non_numeric_range_column(tmp_test_directory):
    """Reject a range rule applied to a non-numeric column."""
    output = Path(tmp_test_directory) / "table.ecsv"
    _write_table(output)

    with pytest.raises(AssertionError, match="not numerical"):
        table.validate_table(output, {"columns": {"label": {"range": {"minimum": 1}}}})


def test_table_validator_checks_exclusive_maximum(tmp_test_directory):
    """Apply an exclusive maximum to a numerical column."""
    output = Path(tmp_test_directory) / "table.ecsv"
    _write_table(output)

    with pytest.raises(AssertionError, match="violates maximum"):
        table.validate_table(
            output,
            {"columns": {"value": {"range": {"maximum": 2.0, "inclusive": False}}}},
        )


def test_table_validator_rejects_duplicate_unique_values(tmp_test_directory):
    """Reject duplicate values in a configured unique column."""
    output = Path(tmp_test_directory) / "table.ecsv"
    Table({"id": [1, 1]}).write(output, format="ascii.ecsv")

    with pytest.raises(AssertionError, match="not unique"):
        table.validate_table(output, {"unique_columns": ["id"]})
