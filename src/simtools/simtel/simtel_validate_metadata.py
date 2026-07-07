"""Validate and describe sim_telarray metadata emitted by simtools."""

import re

from simtools.constants import (
    SIM_TELARRAY_META_PARAMETER_METASCHEMA,
    SIM_TELARRAY_META_PARAMETER_REGISTRY,
)
from simtools.data_model import schema
from simtools.utils import names


def get_meta_parameter_registry(schema_version=None, validate=True):
    """
    Return the expanded sim_telarray meta-parameter registry.

    Generated metadata are read from the registry schema. Model-parameter-derived
    metadata are resolved from the model-parameter schemas.
    """
    registry_source = schema.load_schema(
        SIM_TELARRAY_META_PARAMETER_REGISTRY, schema_version=schema_version or "latest"
    )
    if validate:
        schema.validate_dict_using_schema(
            registry_source,
            schema_file=SIM_TELARRAY_META_PARAMETER_METASCHEMA,
            offline=True,
            ignore_software_version=True,
        )
    return _build_meta_parameter_registry(registry_source)


def validate_metadata(meta_parameters):
    """Validate emitted sim_telarray metadata lines against the registry."""
    registry = get_meta_parameter_registry(validate=False)["meta_parameters"]
    for line in meta_parameters:
        parsed = parse_metadata_line(line)
        definition = registry.get(parsed["name"])
        if definition is None:
            raise KeyError(f"Unknown sim_telarray metadata key emitted by writer: {parsed['name']}")
        if definition["mode"] != parsed["mode"]:
            raise ValueError(
                f"sim_telarray metadata mode mismatch for {parsed['name']}: "
                f"{parsed['mode']} != {definition['mode']}"
            )
        if (
            parsed["scope"] is not None
            and definition["source_type"] == "generated"
            and definition["scope"] != parsed["scope"]
        ):
            raise ValueError(
                f"sim_telarray metadata scope mismatch for {parsed['name']}: "
                f"{parsed['scope']} != {definition['scope']}"
            )
        if parsed["value"] is None:
            if _value_required(definition):
                raise ValueError(
                    f"sim_telarray metadata value missing for required key {parsed['name']}"
                )
            continue
        _validate_metadata_value(parsed["name"], parsed["value"], definition["value_schema"])


def validate_metadata_values(metadata):
    """Validate known decoded sim_telarray metadata values against the registry."""
    registry = get_meta_parameter_registry(validate=False)["meta_parameters"]
    for name, value in metadata.items():
        definition = registry.get(name)
        if definition is None or value is None:
            continue
        _validate_metadata_value(name, str(value), definition["value_schema"])


def parse_metadata_line(line):
    """Parse one emitted sim_telarray metadata line."""
    meta_match = re.fullmatch(
        r"metaparam (global|telescope) (add|set) ([^=\s]+)(?:\s*=\s*(.*))?",
        line,
    )
    if meta_match:
        return {
            "scope": meta_match.group(1),
            "mode": meta_match.group(2),
            "name": meta_match.group(3),
            "value": meta_match.group(4),
        }

    assign_match = re.fullmatch(r"(\w+)\s*=\s*(.*)", line)
    if assign_match:
        return {
            "scope": None,
            "mode": "assign",
            "name": assign_match.group(1),
            "value": assign_match.group(2),
        }

    raise ValueError(f"Unsupported sim_telarray metadata line: {line}")


def _build_meta_parameter_registry(registry_source):
    """Build expanded sim_telarray meta_parameter registry from source overlays."""
    registry = {
        key: value for key, value in registry_source.items() if key != "generated_meta_parameters"
    }
    meta_parameters = {}

    for emitted_name, definition in registry_source.get("generated_meta_parameters", {}).items():
        meta_parameters[emitted_name] = {
            "name": emitted_name,
            "source_type": "generated",
            **definition,
        }

    for source_name, model_schema in names.model_parameters().items():
        try:
            emitted_name, _ = _get_emitted_name_and_mode(model_schema)
        except KeyError:
            continue
        if emitted_name in meta_parameters:
            continue
        definition = _build_model_parameter_definition(source_name)
        meta_parameters[definition["name"]] = definition

    registry["meta_parameters"] = meta_parameters
    return registry


def _build_model_parameter_definition(source_name, emitted_name=None):
    """Build one sim_telarray meta_parameter definition from a model parameter schema."""
    model_schema = schema.get_model_parameter_schema(source_name)
    derived_name, mode = _get_emitted_name_and_mode(model_schema)
    emitted_name = emitted_name or derived_name
    value_schema = _get_output_value_schema(model_schema) or _derive_value_schema(model_schema)

    return {
        "name": emitted_name,
        "scope": _derive_scope(model_schema),
        "mode": mode,
        "source_type": "model_parameter",
        "value_schema": value_schema,
    }


def _get_emitted_name_and_mode(model_schema):
    """Return emitted sim_telarray metadata name and metadata mode."""
    software = _get_software_definition(model_schema)
    if software.get("set_meta_parameter", False):
        return software.get("internal_parameter_name", model_schema["name"]), "set"
    return software.get("internal_parameter_name", model_schema["name"]), "add"


def _get_software_definition(model_schema):
    """Return the sim_telarray simulation_software definition for a model schema."""
    for software in model_schema.get("simulation_software", []):
        if software.get("name") == "sim_telarray":
            return software
    raise KeyError(f"Model parameter without sim_telarray mapping: {model_schema['name']}")


def _get_output_value_schema(model_schema):
    """Return an explicit sim_telarray emitted-value schema when declared."""
    return _get_software_definition(model_schema).get("output_value_schema")


def _derive_scope(model_schema):
    """Infer sim_telarray metadata scope from instrument class."""
    if model_schema.get("instrument", {}).get("class") == "Site":
        return "global"
    return "telescope"


def _derive_value_schema(model_schema):
    """Infer value schema from model parameter schema structure."""
    data = model_schema.get("data", [])
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list) or len(data) == 0:
        return {"kind": "scalar", "data_type": "string"}

    if len(data) > 1 and all(_is_numeric_type(entry.get("type")) for entry in data):
        return {"kind": "fixed_numeric_tuple", "item_type": "number", "length": len(data)}

    datum = data[0]
    data_type = datum.get("type")

    if data_type == "file":
        value_schema = {"kind": "file_name"}
        if datum.get("default") is None:
            value_schema["allow_none_literal"] = True
        return value_schema

    return {"kind": "scalar", "data_type": _derive_scalar_data_type(data_type)}


def _derive_scalar_data_type(data_type):
    """Return the sim_telarray scalar data_type string for a model parameter type."""
    if _is_integer_type(data_type):
        return "integer"
    if _is_numeric_type(data_type):
        return "number"
    if data_type == "boolean":
        return "boolean"
    return "string"


def _is_integer_type(data_type):
    """Check whether a schema data type is integer-like."""
    return data_type in {"int", "int64", "uint", "uint32", "uint64"}


def _is_numeric_type(data_type):
    """Check whether a schema data type is numeric-like."""
    return _is_integer_type(data_type) or data_type in {"double", "float64", "float32"}


def _value_required(definition):
    """Return whether an emitted metadata line must include a value."""
    return definition["mode"] in {"set", "assign"}


def _validate_metadata_value(name, value, value_schema):
    """Validate one emitted metadata value against its value schema."""
    kind = value_schema["kind"]

    if kind == "scalar":
        _validate_scalar_value(name, value, value_schema)
    elif kind == "file_name":
        _validate_file_name(name, value, value_schema)
    elif kind == "fixed_numeric_tuple":
        _validate_fixed_numeric_tuple(name, value, value_schema)
    elif kind == "sim_telarray_key_value_string":
        if not re.fullmatch(value_schema["regex"], value):
            raise ValueError(f"sim_telarray metadata value for {name} does not match regex")
    else:
        raise ValueError(f"Unsupported value schema kind for {name}: {kind}")


def _validate_file_name(name, value, value_schema):
    """Validate one emitted file-like metadata value against its value schema."""
    if value_schema.get("allow_none_literal") and value == "none":
        return
    if value == "" and not value_schema.get("allow_empty", False):
        raise ValueError(f"Empty file-like metadata value for {name}")


def _validate_fixed_numeric_tuple(name, value, value_schema):
    """Validate one emitted fixed numeric tuple metadata value against its value schema."""
    parts = value.split()
    if len(parts) != value_schema["length"]:
        raise ValueError(
            f"sim_telarray metadata tuple length mismatch for {name}: "
            f"{len(parts)} != {value_schema['length']}"
        )
    for part in parts:
        float(part)


def _validate_scalar_value(name, value, value_schema):
    """Validate one scalar metadata value."""
    data_type = value_schema["data_type"]
    if data_type == "string":
        if value == "" and not value_schema.get("allow_empty", False):
            raise ValueError(f"Empty string metadata value for {name}")
        return
    if data_type == "integer":
        int(value)
        return
    if data_type == "number":
        float(value)
        return
    if data_type == "boolean":
        if value not in ("0", "1", "true", "false", "True", "False"):
            raise ValueError(f"Invalid boolean metadata value for {name}: {value}")
        return
    raise ValueError(f"Unsupported scalar metadata data type for {name}: {data_type}")
