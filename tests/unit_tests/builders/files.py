"""Write minimal model-parameter assets for parser tests."""

from pathlib import Path

from astropy.table import Table

from simtools.data_model import schema
from simtools.data_model.table_asset import validate_table_asset
from tests.unit_tests.builders.parameters import columns, table_value


def table_file(name, destination, rows=None):
    """Write an ECSV asset using a file-valued parameter schema and return its path.

    ``destination`` is a file path ending in ``.ecsv``. With no ``rows``, generate
    one row from the schema defaults and allowed ranges. Explicit rows are sequences
    or dictionaries with values in schema column order and units. All declared
    columns, including optional columns, are included.
    """
    description = schema.get_model_parameter_schema(name)
    data = description["data"]
    definitions = columns(data)
    if data[0]["type"] != "file" or not definitions:
        raise ValueError(f"{name} must be a file parameter with table_columns")
    destination = Path(destination)
    if destination.suffix.lower() != ".ecsv":
        raise ValueError("ECSV table destination must end in .ecsv")

    table = (
        table_value(definitions)
        if rows is None
        else Table(rows=rows, names=[column["name"] for column in definitions])
    )
    if len(table) == 0:
        raise ValueError("A model-table file must contain at least one row")
    for column in definitions:
        column_name = column["name"]
        dtype = str if column["type"] == "string" else column["type"]
        table[column_name] = table[column_name].astype(dtype)
        unit = column.get("unit")
        table[column_name].unit = None if unit == "dimensionless" else unit
        table[column_name].description = column["description"]
    validate_table_asset(table, schema_entry=data[0])
    table.write(destination, format="ascii.ecsv")
    return destination


def camera_configuration_file(destination):
    """Write a native camera configuration with one enabled pixel and return its path."""
    destination = Path(destination)
    destination.write_text(
        'Rotate 0\nPixType 0 0 1 1 1 1 0 "none" "none"\nPixel 0 0 0 0 0 0 0 0 1\n',
        encoding="utf-8",
    )
    return destination
