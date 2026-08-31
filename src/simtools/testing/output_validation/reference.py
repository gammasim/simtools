"""Reference-file comparison validators."""

import difflib
import logging
from pathlib import Path

import numpy as np
from astropy.table import Table

from simtools.io import ascii_handler
from simtools.utils import general

_logger = logging.getLogger(__name__)


def resolve_path(path):
    """Resolve a repository-relative reference path."""
    reference = Path(path)
    if reference.is_absolute():
        return reference
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent / reference
    return Path.cwd() / reference


def _resource_suffixes(value):
    if not isinstance(value, str) or "/" not in value:
        return set()
    parts = [part for part in Path(value).as_posix().split("/") if part not in ("", ".")]
    markers = ("integration_tests", "tests", "static", "generated", "downloaded")
    return {tuple(parts[index:]) for index, part in enumerate(parts) if part in markers}


def _compare_mappings(first, second, tolerance):
    """Compare nested mappings, applying tolerance to model-parameter values."""
    return first.keys() == second.keys() and all(
        _compare_values(first[key], second[key], tolerance, key == "value") for key in first
    )


def _compare_sequences(first, second, tolerance, value_field):
    """Compare ordered nested sequences."""
    return len(first) == len(second) and all(
        _compare_values(left, right, tolerance, value_field) for left, right in zip(first, second)
    )


def _compare_model_values(first, second, tolerance):
    """Compare possibly string-encoded numerical model-parameter values."""
    try:
        left = general.convert_string_to_list(first) if isinstance(first, str) else first
        right = general.convert_string_to_list(second) if isinstance(second, str) else second
        return np.allclose(np.atleast_1d(left), np.atleast_1d(right), rtol=tolerance)
    except TypeError, ValueError:
        return first == second


def _compare_values(first, second, tolerance, value_field=False):
    if isinstance(first, dict) and isinstance(second, dict):
        return _compare_mappings(first, second, tolerance)
    if isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
        return _compare_sequences(first, second, tolerance, value_field)
    if value_field:
        return _compare_model_values(first, second, tolerance)
    if isinstance(first, str) and isinstance(second, str):
        return first == second or bool(_resource_suffixes(first) & _resource_suffixes(second))
    return first == second


def compare_json_or_yaml_files(first_file, second_file, tolerance=1.0e-2):
    """Compare JSON or YAML documents, tolerating numerical model values."""
    first = ascii_handler.collect_data_from_file(first_file)
    second = ascii_handler.collect_data_from_file(second_file)
    if isinstance(first, dict):
        first.pop("schema_version", None)
    if isinstance(second, dict):
        second.pop("schema_version", None)
    return _compare_values(first, second, tolerance)


def _filter_mask(table, rule):
    """Return a boolean mask for one typed reference-table filter."""
    values = np.asarray(table[rule["column"]])
    expected = rule["value"]
    operators = {
        "equal": lambda: values == expected,
        "not_equal": lambda: values != expected,
        "less": lambda: values < expected,
        "less_equal": lambda: values <= expected,
        "greater": lambda: values > expected,
        "greater_equal": lambda: values >= expected,
        "in": lambda: np.isin(values, expected),
        "not_in": lambda: ~np.isin(values, expected),
    }
    try:
        return operators[rule["operator"]]()
    except KeyError as exc:
        raise ValueError(f"Unknown reference filter operator '{rule['operator']}'.") from exc


def _prepare_table(table, filters, key_columns):
    """Apply typed filters and optional deterministic key ordering."""
    prepared = table
    for rule in filters or []:
        prepared = prepared[_filter_mask(prepared, rule)]
    if key_columns:
        keys = zip(*(np.asarray(prepared[name]) for name in key_columns))
        if len(set(keys)) != len(prepared):
            raise ValueError(f"Reference key columns are not unique: {key_columns}")
        prepared = prepared.copy()
        prepared.sort(key_columns)
    return prepared


def _compare_column(left, right, tolerance):
    """Compare two ECSV columns, including their dtype and unit."""
    if left.dtype != right.dtype or left.unit != right.unit:
        return False
    if np.issubdtype(left.dtype, np.number):
        return np.allclose(left, right, rtol=tolerance, equal_nan=True)
    return np.array_equal(np.asarray(left), np.asarray(right))


def _compare_selected_columns(first, second, selected, tolerance):
    """Compare selected columns from two ECSV tables."""
    return all(_compare_column(first[name], second[name], tolerance) for name in selected)


def compare_ecsv_files(
    first_file,
    second_file,
    tolerance=1.0e-5,
    columns=None,
    metadata=False,
    filters=None,
    key_columns=None,
):
    """Compare ECSV rows, selected columns, units, and optional metadata."""
    first = _prepare_table(Table.read(first_file, format="ascii.ecsv"), filters, key_columns)
    second = _prepare_table(Table.read(second_file, format="ascii.ecsv"), filters, key_columns)
    selected = columns or first.colnames
    if columns is None and first.colnames != second.colnames:
        return False
    if len(first) != len(second) or any(
        name not in first.colnames or name not in second.colnames for name in selected
    ):
        return False
    return _compare_selected_columns(first, second, selected, tolerance) and (
        not metadata or first.meta == second.meta
    )


def compare_files(
    first_file,
    second_file,
    tolerance=1.0e-5,
    columns=None,
    metadata=False,
    filters=None,
    key_columns=None,
):
    """Compare supported structured files."""
    first_suffix = Path(first_file).suffix.lower()
    if first_suffix != Path(second_file).suffix.lower():
        raise ValueError(f"File suffixes do not match: {first_file} and {second_file}")
    if first_suffix == ".ecsv":
        return compare_ecsv_files(
            first_file,
            second_file,
            tolerance,
            columns,
            metadata,
            filters,
            key_columns,
        )
    if first_suffix in (".json", ".yaml", ".yml"):
        return compare_json_or_yaml_files(first_file, second_file, tolerance)
    _logger.warning(f"Unknown file type for files: {first_file} and {second_file}")
    return False


def difference_report(reference_file, output_file):
    """Return a unified diff between a reference file and generated output."""
    try:
        reference_lines = (
            Path(reference_file)
            .read_text(encoding="utf-8", errors="replace")
            .splitlines(keepends=True)
        )
        output_lines = (
            Path(output_file).read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        )
    except OSError as exc:
        return f"(Unable to read files for diff: {exc})"

    return "".join(
        difflib.unified_diff(
            reference_lines,
            output_lines,
            fromfile=f"reference: {reference_file}",
            tofile=f"generated: {output_file}",
        )
    )
