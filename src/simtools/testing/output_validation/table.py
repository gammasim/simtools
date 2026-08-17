"""Generic table and metadata validators."""

from collections.abc import Mapping

import numpy as np
from astropy import units as u
from astropy.table import Table

from simtools.data_model import validate_data


def read_table(path):
    """Read an Astropy-supported output table using format auto-detection."""
    try:
        return Table.read(path)
    except Exception as exc:
        raise AssertionError(f"Output '{path}' is not a parseable table: {exc}") from exc


def get_path(value, dotted_path):
    """Read a dotted path from nested mappings."""
    current = value
    for part in dotted_path.split("."):
        current = current[part]
    return current


def has_path(value, dotted_path):
    """Return whether a dotted path exists in nested mappings."""
    current = value
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False
        current = current[part]
    return True


def validate_data_schema(path, schema_file):
    """Validate a table against a simtools data-product schema."""
    table = read_table(path)
    try:
        validate_data.DataValidator(
            schema_file=schema_file,
            data_table=table,
        ).validate_and_transform()
    except Exception as exc:
        raise AssertionError(
            f"Output '{path}' failed data-product schema '{schema_file}': {exc}"
        ) from exc


def _validate_range(path, column_name, column, rule):
    range_rule = rule.get("range")
    if not range_rule:
        return
    values = np.asarray(column)
    if not np.issubdtype(values.dtype, np.number):
        raise AssertionError(f"Output '{path}' column '{column_name}' is not numerical.")
    actual_unit = getattr(column, "unit", None) or u.dimensionless_unscaled
    quantities = values * actual_unit
    if range_rule.get("unit"):
        quantities = quantities.to(range_rule["unit"])
    values = np.asarray(quantities.value)
    inclusive = range_rule.get("inclusive", True)
    minimum = range_rule.get("minimum")
    maximum = range_rule.get("maximum")
    if minimum is not None:
        valid = values >= minimum if inclusive else values > minimum
        if not np.all(valid):
            raise AssertionError(
                f"Output '{path}' column '{column_name}' violates minimum {minimum}."
            )
    if maximum is not None:
        valid = values <= maximum if inclusive else values < maximum
        if not np.all(valid):
            raise AssertionError(
                f"Output '{path}' column '{column_name}' violates maximum {maximum}."
            )


def validate_table(path, rule):
    """Validate table row and column expectations."""
    table = read_table(path)
    minimum_rows = rule.get("minimum_rows", 0)
    if len(table) < minimum_rows:
        raise AssertionError(f"Output '{path}' has {len(table)} rows; expected {minimum_rows}.")
    for column_name in rule.get("unique_columns", []):
        values = np.asarray(table[column_name])
        if len(np.unique(values)) != len(values):
            raise AssertionError(f"Output '{path}' column '{column_name}' is not unique.")
    for column_name, column_rule in rule.get("columns", {}).items():
        values = np.asarray(table[column_name])
        allowed = column_rule.get("allowed_values")
        if allowed is not None and not np.all(np.isin(values, allowed)):
            raise AssertionError(
                f"Output '{path}' column '{column_name}' contains values outside {allowed}."
            )
        _validate_range(path, column_name, table[column_name], column_rule)


def validate_metadata(path, rule):
    """Validate table metadata keys and metadata-to-content relations."""
    table = read_table(path)
    for metadata_path in rule.get("required_keys", []):
        if not has_path(table.meta, metadata_path):
            raise AssertionError(f"Output '{path}' has no metadata key '{metadata_path}'.")
    for relation in rule.get("relations", []):
        metadata_value = get_path(table.meta, relation["left"])
        if relation["equals"] == "table.row_count":
            content_value = len(table)
        else:
            content_value = np.sum(np.asarray(table[relation["column"]]))
        if not np.isclose(metadata_value, content_value):
            raise AssertionError(
                f"Output '{path}' metadata '{relation['left']}' is {metadata_value!r}; "
                f"expected {content_value!r}."
            )
