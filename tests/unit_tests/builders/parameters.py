"""Build model parameter records from the packaged parameter schemas.

The schemas are not changed for tests: every value is taken from what a schema declares
(``default``, ``json_schema``, ``table_columns``, ``allowed_values``, ``type`` with
``allowed_range``). Tests that need realistic values pass them explicitly.
"""

from copy import deepcopy

import yaml
from astropy.table import Table

from simtools.constants import SCHEMA_PATH
from simtools.data_model import schema
from simtools.data_model.validate_data import DataValidator

_COMMON_DEFINITIONS = "common_definitions.schema.yml"
_EXAMPLE_STRING = "example"
# Valid values for the custom string formats in simtools.data_model.format_checkers.
_FORMAT_VALUES = {
    "array_element": "LSTN-01",
    "array_trigger_name": "LSTN_array",
    "astropy_unit": "m",
    "astropy_unit_of_length": "m",
    "astropy_unit_of_time": "ns",
    "common_particle_name": "default",
}


def parameter(name, value=None, site=None, instrument=None, version="1.0.0"):
    """Return a validated model parameter record as stored in the model repository.

    Parameters
    ----------
    name : str
        Model parameter name (schema file name without suffix).
    value : object, optional
        Parameter value; generated from the latest schema version if not given.
    site : str, optional
        Site name (``North`` or ``South``).
    instrument : str, optional
        Instrument name.
    version : str
        Parameter version.

    Returns
    -------
    dict or astropy.table.Table
        Parameter record as stored in the model repository (validated, not transformed);
        a validated table for ``data_table`` parameters.
    """
    description = schema.get_model_parameter_schema(name)
    data = description["data"]
    if data[0]["type"] == "data_table":
        columns = _columns(data)
        table = Table(
            {column["name"]: [_scalar_value(column)] for column in columns},
            units={column["name"]: column["unit"] for column in columns if column.get("unit")},
        )
        return DataValidator(
            data_table=table, schema_file=schema.get_model_parameter_schema_file(name)
        ).validate_and_transform()

    record = {
        "file": data[0]["type"] == "file",
        "parameter": name,
        "parameter_version": version,
        "model_parameter_schema_version": description["schema_version"],
        "type": data[0]["type"],
        "unit": _units(data),
        "value": _value(name, data) if value is None else value,
    }
    if site is not None:
        record["site"] = site
    if instrument is not None:
        record["instrument"] = instrument
    DataValidator.validate_model_parameter(deepcopy(record))
    return record


def _columns(data):
    """Return the table column definitions of a table-valued parameter."""
    return data[0].get("table_columns", [])


def _units(data):
    """Return the declared unit, or one unit per value for multi-value parameters."""
    units = [datum.get("unit") for datum in data]
    return units[0] if len(units) == 1 else units


def _value(name, data):
    """Return a schema-declared value, or one value per datum for multi-value parameters."""
    values = [_datum_value(name, datum) for datum in data]
    return values[0] if len(values) == 1 else values


def _datum_value(name, datum):
    """Return the value of one schema datum."""
    if "default" in datum:
        return datum["default"]
    if datum["type"] == "file":
        suffix = ".ecsv" if datum.get("table_columns") else ".dat"
        return f"{name}{suffix}"
    if datum["type"] == "dict":
        return _json_schema_value(datum["json_schema"])
    return _scalar_value(datum)


def _scalar_value(definition):
    """Return a value for a scalar datum or table column from its type and limits."""
    if "default" in definition:
        return definition["default"]
    if definition.get("allowed_values"):
        return definition["allowed_values"][0]
    limits = definition.get("allowed_range", {})
    for limit in ("min", "max"):
        if limit in limits:
            return limits[limit]
    kind = definition.get("type", "string")
    if kind in ("string", "str"):
        return _EXAMPLE_STRING
    if kind in ("boolean", "bool"):
        return False
    if kind.startswith(("int", "uint")):
        return 0
    return 0.0


def _json_schema_value(json_schema):
    """Return a minimal instance of a JSON schema (resolving shared definitions)."""
    if "$ref" in json_schema:
        return _json_schema_value(_resolve_reference(json_schema["$ref"]))
    for key in ("default", "const"):
        if key in json_schema:
            return json_schema[key]
    for key in ("enum", "oneOf", "anyOf"):
        if key in json_schema:
            option = json_schema[key][0]
            return option if key == "enum" else _json_schema_value(option)
    return _JSON_TYPE_VALUES[json_schema.get("type", "null")](json_schema)


def _array_value(json_schema):
    """Return the shortest array allowed by the schema."""
    item = _json_schema_value(json_schema.get("items", {}))
    return [item] * max(1, json_schema.get("minItems", 1))


def _object_value(json_schema):
    """Return an object with all required properties."""
    properties = json_schema.get("properties", {})
    return {name: _json_schema_value(properties[name]) for name in json_schema.get("required", [])}


def _number_value(json_schema):
    """Return the smallest allowed number (exclusive minimum moved up by one)."""
    if "minimum" in json_schema:
        return json_schema["minimum"]
    if "exclusiveMinimum" in json_schema:
        return json_schema["exclusiveMinimum"] + 1
    return 0


def _string_value(json_schema):
    """Return a string valid for the declared format, of at least the minimum length."""
    if "format" in json_schema:
        return _FORMAT_VALUES[json_schema["format"]]
    return _EXAMPLE_STRING.ljust(json_schema.get("minLength", 0), "x")


_JSON_TYPE_VALUES = {
    "array": _array_value,
    "object": _object_value,
    "number": lambda json_schema: float(_number_value(json_schema)),
    "integer": lambda json_schema: int(_number_value(json_schema)),
    "string": _string_value,
    "boolean": lambda _json_schema: False,
    "null": lambda _json_schema: None,
}


def _resolve_reference(reference):
    """Return the JSON schema a ``common_definitions`` reference points to."""
    file_name, _, pointer = reference.partition("#")
    if file_name != _COMMON_DEFINITIONS:
        raise ValueError(f"Unsupported JSON schema reference: {reference}")
    resolved = yaml.safe_load((SCHEMA_PATH / _COMMON_DEFINITIONS).read_text(encoding="utf-8"))
    for key in pointer.strip("/").split("/"):
        resolved = resolved[key]
    return resolved
