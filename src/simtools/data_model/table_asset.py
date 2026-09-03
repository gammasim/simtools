"""Read and validate ECSV assets referenced by model parameters."""

from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.table import QTable, unique


def resolve_asset_path(value, parameter_file, files_path, asset_location="parameter_directory"):
    """Resolve a model-parameter asset path.

    Assets are relative to their parameter JSON file by default. The temporary
    ``shared_files`` location names an asset below the repository's shared Files directory.
    Absolute paths and path traversal are rejected.
    """
    value_path = Path(value)
    if value_path.is_absolute():
        raise ValueError(f"Model asset path must be relative: {value}")

    if asset_location == "parameter_directory":
        root = Path(parameter_file).parent.resolve()
        candidate = (root / value_path).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError(f"Model asset path escapes parameter directory: {value}")
        return candidate

    if asset_location != "shared_files":
        raise ValueError(f"Unknown model asset location: {asset_location}")
    root = Path(files_path).resolve()
    candidate = (root / value_path).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Model asset path escapes model Files directory: {value}")
    return candidate


def read_ecsv_asset(path, schema_entry=None, parameter_data=None):
    """Read and validate one ECSV model asset.

    Parameters
    ----------
    path : str or pathlib.Path
        ECSV asset path.
    schema_entry : dict, optional
        ``data`` entry from the model-parameter schema. Its ``table_columns`` are validated when
        present.
    parameter_data : dict, optional
        JSON parameter record used to validate identity metadata.

    Returns
    -------
    astropy.table.QTable
        Validated table.
    """
    path = Path(path)
    if path.suffix.lower() != ".ecsv":
        raise ValueError(f"Model table asset must use ECSV format: {path}")
    table = QTable.read(path, format="ascii.ecsv")
    validate_table_asset(table, schema_entry=schema_entry, parameter_data=parameter_data)
    return table


def validate_table_asset(table, schema_entry=None, parameter_data=None):
    """Validate ECSV columns, units, ranges, and identity metadata in place."""
    if not isinstance(table, QTable):
        table = QTable(table)

    table_columns = (schema_entry or {}).get("table_columns", [])
    _validate_table_columns(
        table,
        table_columns,
        allow_extra_columns=(schema_entry or {}).get("allow_extra_columns", False),
    )
    _validate_processing(table, table_columns)
    _validate_metadata(table, (schema_entry or {}).get("table_metadata", []))
    _validate_identity(table, parameter_data)
    return table


def _validate_table_columns(table, table_columns, allow_extra_columns=False):
    """Validate table columns against schema declarations."""
    descriptions = {entry["name"]: entry for entry in table_columns}
    required = {entry["name"] for entry in table_columns if entry.get("required", False)}
    missing = sorted(required - set(table.colnames))
    if missing:
        raise ValueError(f"Missing required table columns: {missing}")
    if descriptions and not allow_extra_columns:
        unexpected = sorted(set(table.colnames) - set(descriptions))
        if unexpected:
            raise ValueError(f"Unexpected table columns: {unexpected}")
    for name in table.colnames:
        _validate_column(name, table[name], descriptions.get(name))


def _validate_column(name, column, entry):
    """Validate one column and its numeric constraints."""
    if entry is None:
        if not column.info.description:
            raise ValueError(f"Table column '{name}' has no description")
        return
    if not column.info.description:
        raise ValueError(f"Table column '{name}' has no description")
    _validate_unit(name, column, entry.get("unit"))
    _validate_dtype(name, column, entry.get("type"))
    if np.issubdtype(column.dtype, np.number):
        _validate_numeric_column(name, column, entry)


def _validate_unit(name, column, expected_unit):
    """Require the canonical schema unit for a physical column."""
    if not expected_unit or expected_unit == "dimensionless":
        return
    if getattr(column, "unit", None) is None:
        raise ValueError(f"Table column '{name}' is missing unit '{expected_unit}'")
    try:
        matches = u.Unit(column.unit) == u.Unit(expected_unit)
    except (TypeError, ValueError, u.UnitsError) as exc:
        raise ValueError(
            f"Table column '{name}' has unit {column.unit}; expected {expected_unit}"
        ) from exc
    if not matches:
        raise ValueError(f"Table column '{name}' has unit {column.unit}; expected {expected_unit}")


def _validate_numeric_column(name, column, entry):
    """Validate finite values and declared numeric ranges."""
    values = np.asarray(column)
    allow_nan = "allow_nan" in entry.get("input_processing", [])
    if not np.all(np.isfinite(values)) and not allow_nan:
        raise ValueError(f"Table column '{name}' contains NaN or infinite values")
    _validate_range(name, values, entry.get("allowed_range"), "allowed_range")
    _validate_range(name, values, entry.get("required_range"), "required_range")


def _validate_dtype(name, column, expected):
    """Validate a NumPy-compatible schema dtype."""
    if not expected or expected in ("file", "data_table"):
        return
    if expected == "string":
        if column.dtype.kind not in "OUS":
            raise ValueError(f"Table column '{name}' must contain strings")
        return
    try:
        expected_dtype = np.dtype(expected)
    except TypeError:
        return
    if (
        not np.can_cast(column.dtype, expected_dtype, casting="safe")
        and column.dtype != expected_dtype
    ):
        raise ValueError(f"Table column '{name}' has dtype {column.dtype}; expected {expected}")


def _validate_range(name, values, value_range, range_name):
    """Validate a numeric range declaration."""
    if not value_range:
        return
    if "min" in value_range and np.nanmin(values) < value_range["min"]:
        raise ValueError(f"Table column '{name}' violates {range_name}.min")
    if "max" in value_range and np.nanmax(values) > value_range["max"]:
        raise ValueError(f"Table column '{name}' violates {range_name}.max")


def _validate_processing(table, table_columns):
    """Validate uniqueness and ordering declarations."""
    sort_columns = [
        entry["name"] for entry in table_columns if "sort" in entry.get("input_processing", [])
    ]
    if sort_columns:
        values = np.asarray(table[sort_columns[0]])
        if np.any(values[1:] < values[:-1]):
            raise ValueError(f"Table is not sorted by '{sort_columns[0]}'")
    unique_columns = [
        entry["name"]
        for entry in table_columns
        if "remove_duplicates" in entry.get("input_processing", [])
    ]
    if unique_columns and len(unique(table, keys=unique_columns)) != len(table):
        raise ValueError(f"Duplicate values found for table key columns {unique_columns}")


def _validate_identity(table, parameter_data):
    """Validate ECSV identity metadata when a parameter record is supplied."""
    if parameter_data is None:
        return
    for metadata_key, parameter_key in (
        ("parameter_name", "parameter"),
        ("parameter_version", "parameter_version"),
        ("instrument", "instrument"),
        ("site", "site"),
    ):
        expected = parameter_data.get(parameter_key)
        if metadata_key not in table.meta:
            raise ValueError(f"Missing required ECSV metadata: {metadata_key}")
        if table.meta[metadata_key] != expected:
            raise ValueError(f"ECSV metadata '{metadata_key}' does not match model parameter")


def _validate_metadata(table, metadata_entries):
    """Validate declared table metadata keys and basic JSON-compatible types."""
    for entry in metadata_entries:
        name = entry["name"]
        if name not in table.meta:
            if entry.get("required", False):
                raise ValueError(f"Missing required table metadata: {name}")
            continue
        value = table.meta[name]
        expected = entry.get("type")
        if not _metadata_matches_type(name, value, expected):
            raise ValueError(f"Table metadata '{name}' is not of type {expected}")


def _metadata_matches_type(name, value, expected):
    """Return whether a metadata value has its declared JSON-compatible type."""
    if name == "site" and expected == "string":
        return isinstance(value, str) or (
            isinstance(value, list) and all(isinstance(item, str) for item in value)
        )
    return {
        "string": isinstance(value, str),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "array": isinstance(value, list),
        "object": isinstance(value, dict),
        "null": value is None,
    }.get(expected, True)
