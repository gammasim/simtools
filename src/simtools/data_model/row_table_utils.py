"""Utilities for row-oriented table data (columns, column_units, rows)."""

import logging
from typing import Any

import astropy.units as u
import numpy as np

logger = logging.getLogger(__name__)

# Canonical set of row-table dict keys
ROW_TABLE_KEYS = {"columns", "rows", "column_units"}


def is_row_table_dict(value: Any) -> bool:
    """
    Check if a dict has the row-table structure (columns, rows, column_units).

    Parameters
    ----------
    value : Any
        Value to check.

    Returns
    -------
    bool
        True if value is a dict with all three required keys.
    """
    if not isinstance(value, dict):
        return False
    return all(key in value for key in ROW_TABLE_KEYS)


def is_row_table_schema(json_schema: dict[str, Any]) -> bool:
    """
    Check if a JSON schema defines row-table shape.

    Requires ``columns``, ``rows``, and ``column_units`` keys.

    Parameters
    ----------
    json_schema : dict
        JSON schema properties dict.

    Returns
    -------
    bool
        True if schema specifies row-table structure.
    """
    required = set(json_schema.get("required", []))
    properties = set(json_schema.get("properties", {}).keys())
    return ROW_TABLE_KEYS.issubset(required) and ROW_TABLE_KEYS.issubset(properties)


def validate_row_table_structure(
    parameter_name: str, value: dict[str, Any], require_column_units: bool = True
) -> None:
    """
    Validate row-table dict structure and value consistency.

    Checks:
    - Dict contains 'columns' and 'rows' keys
    - 'columns' and 'rows' are sequences (list or tuple)
    - All column names are strings
    - Each row is a sequence with correct length and numeric values
    - If require_column_units is True (default), validates 'column_units' presence and length

    Parameters
    ----------
    parameter_name : str
        Parameter name (for error messages).
    value : dict
        Row-table dict to validate.
    require_column_units : bool, optional
        If True (default), requires 'column_units' and validates its length.

    Raises
    ------
    ValueError
        If structure is invalid or values are non-numeric.
    """
    if not isinstance(value, dict) or "columns" not in value or "rows" not in value:
        raise ValueError(
            f"Row-table value for '{parameter_name}' must be a dict with 'columns' and 'rows' keys."
        )

    columns = value["columns"]
    rows = value["rows"]

    if not isinstance(columns, (list, tuple)):
        raise ValueError(
            f"Row-table for '{parameter_name}': 'columns' must be a list or tuple, "
            f"got {type(columns).__name__}."
        )

    if not isinstance(rows, (list, tuple)):
        raise ValueError(
            f"Row-table for '{parameter_name}': 'rows' must be a list or tuple, "
            f"got {type(rows).__name__}."
        )

    if not all(isinstance(column_name, str) for column_name in columns):
        raise ValueError(
            f"Row-table for '{parameter_name}': all column names in 'columns' must be strings."
        )

    if require_column_units:
        if "column_units" not in value:
            raise ValueError(f"Row-table value for '{parameter_name}' must include 'column_units'.")
        column_units = value["column_units"]
        if len(column_units) != len(columns):
            raise ValueError(
                f"Row-table for '{parameter_name}': column_units length ({len(column_units)}) "
                f"must match columns length ({len(columns)})."
            )

    _validate_row_values(parameter_name, columns, rows)


def _validate_row_values(parameter_name: str, columns: list[str], rows: list) -> None:
    """Validate numeric scalars and row length consistency."""
    n_columns = len(columns)
    for row_index, row in enumerate(rows):
        if not isinstance(row, (list, tuple, np.ndarray)):
            raise ValueError(
                f"Row-table for '{parameter_name}' has invalid row at index {row_index}: "
                "each row must be a sequence with one numeric value per column."
            )

        if len(row) != n_columns:
            raise ValueError(
                f"Row-table for '{parameter_name}' has invalid row length at index {row_index}: "
                f"expected {n_columns} values, got {len(row)}."
            )

        for col_index, value in enumerate(row):
            if not np.isscalar(value):
                raise ValueError(
                    f"Row-table for '{parameter_name}' has non-numeric value at "
                    f"row {row_index}, column {col_index} ('{columns[col_index]}'): {value!r}."
                )

            value_dtype = np.asarray(value).dtype
            is_numeric = np.issubdtype(value_dtype, np.number)
            is_real = not np.issubdtype(value_dtype, np.complexfloating)

            if not (is_numeric and is_real):
                raise ValueError(
                    f"Row-table for '{parameter_name}' has non-real-numeric value at "
                    f"row {row_index}, column {col_index} ('{columns[col_index]}'): {value!r}."
                )


def normalize_column_unit(unit_value: Any) -> str:
    """
    Convert astropy unit or string to schema-compatible unit string.

    Parameters
    ----------
    unit_value : Any
        Unit value (astropy.units.Unit, str, or None).

    Returns
    -------
    str
        Normalized unit string compatible with model parameter schemas.
    """
    if unit_value is None:
        return "dimensionless"

    if isinstance(unit_value, str):
        return unit_value if unit_value else "dimensionless"

    if unit_value == u.dimensionless_unscaled:
        return "dimensionless"

    unit_str = str(unit_value)
    return unit_str if unit_str else "dimensionless"
