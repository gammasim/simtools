"""Tests for reference-file comparison validators."""

import json
from pathlib import Path

import pytest
from astropy import units as u
from astropy.table import Table

from simtools.testing.output_validation import reference


def _write_table(path, metadata=None):
    """Write a small ECSV table for reference comparisons."""
    output = Table({"id": [1, 2], "value": [1.0, 2.0], "label": ["a", "b"]})
    output["value"].unit = u.m
    output.meta = metadata or {"summary": {"rows": 2, "total": 3.0}}
    output.write(path, format="ascii.ecsv", overwrite=True)


def test_compare_files_rejects_different_suffixes(tmp_test_directory):
    """Reject reference files with different suffixes."""
    first = Path(tmp_test_directory) / "first.json"
    second = Path(tmp_test_directory) / "second.yaml"
    first.write_text(json.dumps({"value": [1.0]}), encoding="utf-8")
    second.write_text("value: [1.0]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="suffixes do not match"):
        reference.compare_files(first, second)


def test_reference_comparison_all_column_types_and_metadata(tmp_test_directory):
    """Compare ECSV columns, values, and metadata."""
    first = Path(tmp_test_directory) / "first.ecsv"
    second = Path(tmp_test_directory) / "second.ecsv"
    _write_table(first)
    _write_table(second)
    assert reference.compare_files(first, second, metadata=True)
    changed = Table.read(second, format="ascii.ecsv")
    changed["label"][1] = "x"
    changed.write(second, format="ascii.ecsv", overwrite=True)
    assert not reference.compare_files(first, second)


def test_reference_comparison_rejects_different_column_dtypes(tmp_test_directory):
    """Reject ECSV columns with different data types."""
    first = Path(tmp_test_directory) / "first.ecsv"
    second = Path(tmp_test_directory) / "second.ecsv"
    Table({"id": [1, 2]}).write(first, format="ascii.ecsv")
    Table({"id": [1.0, 2.0]}).write(second, format="ascii.ecsv")

    assert not reference.compare_files(first, second)


def test_reference_comparison_typed_filters_and_key_order(tmp_test_directory):
    """Apply typed filters and deterministic key ordering before comparison."""
    first = Path(tmp_test_directory) / "first.ecsv"
    second = Path(tmp_test_directory) / "second.ecsv"
    Table({"id": [2, 1, 3], "group": ["keep", "keep", "drop"], "value": [2.0, 1.0, 9.0]}).write(
        first, format="ascii.ecsv"
    )
    Table({"id": [1, 2, 4], "group": ["keep", "keep", "drop"], "value": [1.0, 2.0, 8.0]}).write(
        second, format="ascii.ecsv"
    )

    assert reference.compare_files(
        first,
        second,
        filters=[{"column": "group", "operator": "equal", "value": "keep"}],
        key_columns=["id"],
    )


def test_reference_comparison_rejects_duplicate_keys(tmp_test_directory):
    """Reject non-unique reference key columns."""
    first = Path(tmp_test_directory) / "first.ecsv"
    second = Path(tmp_test_directory) / "second.ecsv"
    Table({"id": [1, 1]}).write(first, format="ascii.ecsv")
    Table({"id": [1, 1]}).write(second, format="ascii.ecsv")

    with pytest.raises(ValueError, match="not unique"):
        reference.compare_files(first, second, key_columns=["id"])


def test_json_reference_comparison_ignores_schema_version(tmp_test_directory):
    """Ignore schema-version changes when comparing JSON references."""
    first = Path(tmp_test_directory) / "first.json"
    second = Path(tmp_test_directory) / "second.json"
    first.write_text(json.dumps({"schema_version": "1", "value": [1.0]}), encoding="utf-8")
    second.write_text(json.dumps({"schema_version": "2", "value": [1.0]}), encoding="utf-8")
    assert reference.compare_json_or_yaml_files(first, second)


def test_reference_resolve_path_handles_absolute_and_repository_relative_paths():
    """Resolve absolute paths unchanged and relative paths from the repository root."""
    absolute = Path.cwd() / "reference.json"

    assert reference.resolve_path(absolute) == absolute
    assert reference.resolve_path("tests/reference.json") == Path.cwd() / "tests/reference.json"


@pytest.mark.parametrize(
    ("operator", "expected"),
    [
        ("not_equal", [True, False, True]),
        ("less", [True, False, False]),
        ("less_equal", [True, True, False]),
        ("greater", [False, False, True]),
        ("greater_equal", [False, True, True]),
        ("in", [True, False, True]),
        ("not_in", [False, True, False]),
    ],
)
def test_reference_filter_operators(operator, expected):
    """Apply every supported typed reference-table filter operator."""
    source = Table({"value": [1, 2, 3]})
    value = [1, 3] if operator in ("in", "not_in") else 2

    assert (
        reference._filter_mask(
            source, {"column": "value", "operator": operator, "value": value}
        ).tolist()
        == expected
    )


def test_reference_filter_rejects_unknown_operator():
    """Reject an unsupported reference-table filter operator."""
    with pytest.raises(ValueError, match="Unknown reference filter operator"):
        reference._filter_mask(
            Table({"value": [1]}),
            {"column": "value", "operator": "unknown", "value": 1},
        )


def test_reference_value_comparison_handles_model_and_resource_values():
    """Compare nested model values and equivalent resource paths."""
    assert reference._compare_values({"value": "1.0"}, {"value": [1.0]}, 1.0e-5)
    assert reference._compare_values([1, 2], (1, 2), 1.0e-5)
    assert reference._compare_values(
        "tests/generated/model.json", "/work/tests/generated/model.json", 1.0e-5
    )
    assert reference._compare_values("plain", "plain", 1.0e-5)
    assert not reference._compare_values("plain", "other", 1.0e-5)
    assert reference._compare_model_values("not numeric", "different", 1.0e-5) is False


def test_reference_ecsv_comparison_rejects_columns_and_names_not_present(tmp_test_directory):
    """Reject ECSV files with different columns or missing selected columns."""
    first = Path(tmp_test_directory) / "first.ecsv"
    second = Path(tmp_test_directory) / "second.ecsv"
    Table({"id": [1]}).write(first, format="ascii.ecsv")
    Table({"other": [1]}).write(second, format="ascii.ecsv")

    assert not reference.compare_ecsv_files(first, second)
    assert not reference.compare_ecsv_files(first, first, columns=["missing"])


def test_reference_comparison_returns_false_for_unknown_file_type(tmp_test_directory):
    """Return false for unsupported file suffixes."""
    first = Path(tmp_test_directory) / "first.txt"
    second = Path(tmp_test_directory) / "second.txt"
    first.write_text("one", encoding="utf-8")
    second.write_text("one", encoding="utf-8")

    assert not reference.compare_files(first, second)
