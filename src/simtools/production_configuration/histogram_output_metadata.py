"""Shared helpers for production outputs derived from trigger-histogram metadata."""

import astropy.units as u


def _read_source_value(source, column_name):
    """Read a value from a dict-like source, keeping units of table-column scalars."""
    try:
        value = source[column_name]
    except KeyError, IndexError, TypeError:
        return None
    table = getattr(source, "table", None)
    if table is not None and column_name in table.colnames:
        unit = getattr(table[column_name], "unit", None)
        if unit is not None:
            return u.Quantity(value, unit)
    return value


def extract_histogram_output_metadata(source, file_info_columns, include_array_name=False):
    """Return production-identifying metadata for one histogram-derived output row."""
    metadata = {out: _read_source_value(source, src) for out, src in file_info_columns.items()}
    if include_array_name:
        metadata["array_name"] = _read_source_value(source, "array_name")
    return {key: value for key, value in metadata.items() if value is not None}
