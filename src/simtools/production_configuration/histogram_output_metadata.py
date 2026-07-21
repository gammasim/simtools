"""Shared helpers for production outputs derived from trigger-histogram metadata."""

import astropy.units as u


def _read_source_value(source, column_name):
    """Read one value from a dict-like source, preserving table-column units for row scalars."""
    try:
        value = source[column_name]
    except KeyError, IndexError, TypeError:
        return None

    source_table = getattr(source, "table", None)
    if source_table is not None and column_name in source_table.colnames:
        unit = getattr(source_table[column_name], "unit", None)
        if unit is not None:
            return u.Quantity(value, unit)
    return value


def extract_histogram_output_metadata(source, file_info_columns, include_array_name=False):
    """Return shared production-identifying metadata for one histogram-derived output row."""
    metadata = {
        output_name: _read_source_value(source, input_name)
        for output_name, input_name in file_info_columns.items()
    }
    if include_array_name:
        metadata["array_name"] = _read_source_value(source, "array_name")
    return {key: value for key, value in metadata.items() if value is not None}
