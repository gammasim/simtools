"""
Definition of metadata model for input to and output of simtools.

Follows CTAO top-level data model definition.

* data products submitted to SimPipe ('input')
* data products generated by SimPipe ('output')

"""

import logging
from importlib.resources import files

import astropy.units as u
import jsonschema

import simtools.constants
import simtools.utils.general as gen
from simtools.utils import names

_logger = logging.getLogger(__name__)


@jsonschema.Draft7Validator.FORMAT_CHECKER.checks("astropy_unit", ValueError)
def check_astropy_unit(unit_string):
    """Validate astropy units (including dimensionless) for jsonschema."""
    try:
        u.Unit(unit_string)
        return True
    except ValueError:
        return unit_string == "dimensionless"


def validate_schema(data, schema_file):
    """
    Validate dictionary against schema.

    Parameters
    ----------
    data
        dictionary to be validated
    schema_file (dict)
        schema used for validation

    Raises
    ------
    jsonschema.exceptions.ValidationError
        if validation fails

    """
    schema, schema_file = _load_schema(schema_file)

    try:
        jsonschema.validate(
            data, schema=schema, format_checker=jsonschema.Draft7Validator.FORMAT_CHECKER
        )
    except jsonschema.exceptions.ValidationError:
        _logger.error(f"Failed using {schema}")
        raise
    _logger.debug(f"Successful validation of data using schema from {schema_file}")


def get_default_metadata_dict(schema_file=None, observatory="CTA"):
    """
    Return metadata schema with default values.

    Follows the CTA Top-Level Data Model.

    Parameters
    ----------
    schema_file: str
        Schema file (jsonschema format) used for validation
    observatory: str
        Observatory name

    Returns
    -------
    dict
        Reference schema dictionary.


    """
    schema, _ = _load_schema(schema_file)
    return _fill_defaults(schema["definitions"], observatory)


def _load_schema(schema_file=None):
    """
    Load parameter schema from file from simpipe metadata schema.

    Returns
    -------
    schema_file dict
        Schema used for validation.
    schema_file str
        File name schema is loaded from. If schema_file is not given,
        the default schema file name is returned.

    Raises
    ------
    FileNotFoundError
        if schema file is not found

    """
    if schema_file is None:
        schema_file = files("simtools").joinpath(simtools.constants.METADATA_JSON_SCHEMA)

    try:
        schema = gen.collect_data_from_file_or_dict(file_name=schema_file, in_dict=None)
    except FileNotFoundError:
        schema_file = files("simtools").joinpath("schemas") / schema_file
        schema = gen.collect_data_from_file_or_dict(file_name=schema_file, in_dict=None)
    _logger.debug(f"Loading schema from {schema_file}")
    _add_array_elements("InstrumentTypeElement", schema)

    return schema, schema_file


def _add_array_elements(key, schema):
    """
    Add list of array elements to schema.

    This assumes an element [key]['enum'] is a list of elements.

    Parameters
    ----------
    key: str
        Key in schema dictionary
    schema: dict
        Schema dictionary

    Returns
    -------
    dict
        Schema dictionary with added array elements.

    """
    _list_of_array_elements = sorted(names.array_elements().keys())

    def recursive_search(sub_schema, key):
        if key in sub_schema:
            if "enum" in sub_schema[key] and isinstance(sub_schema[key]["enum"], list):
                sub_schema[key]["enum"] = list(
                    set(sub_schema[key]["enum"] + _list_of_array_elements)
                )
            else:
                sub_schema[key]["enum"] = _list_of_array_elements
        else:
            for _, v in sub_schema.items():
                if isinstance(v, dict):
                    recursive_search(v, key)

    recursive_search(schema, key)
    return schema


def _resolve_references(yaml_data, observatory="CTA"):
    """
    Resolve references in yaml data and expand the received dictionary accordingly.

    Parameters
    ----------
    yaml_data: dict
        Dictionary with yaml data.
    observatory: str
        Observatory name

    Returns
    -------
    dict
        Dictionary with resolved references.

    """

    def expand_ref(ref):
        ref_path = ref.lstrip("#/")
        parts = ref_path.split("/")
        ref_data = yaml_data
        for part in parts:
            if part in ("definitions", observatory):
                continue
            ref_data = ref_data.get(part, {})
        return ref_data

    def resolve_dict(data):
        if "$ref" in data:
            ref = data["$ref"]
            resolved_data = expand_ref(ref)
            if isinstance(resolved_data, dict) and len(resolved_data) > 1:
                return _resolve_references_recursive(resolved_data)
            return resolved_data
        return {k: _resolve_references_recursive(v) for k, v in data.items()}

    def resolve_list(data):
        return [_resolve_references_recursive(item) for item in data]

    def _resolve_references_recursive(data):
        if isinstance(data, dict):
            return resolve_dict(data)
        if isinstance(data, list):
            return resolve_list(data)
        return data

    return _resolve_references_recursive(yaml_data)


def _fill_defaults(schema, observatory="CTA"):
    """
    Fill default values from json schema.

    Parameters
    ----------
    schema: dict
        Schema describing the input data.
    observatory: str
        Observatory name

    Returns
    -------
    dict
        Dictionary with default values.
    """
    defaults = {observatory: {}}
    resolved_schema = _resolve_references(schema[observatory])
    _fill_defaults_recursive(resolved_schema, defaults[observatory])
    return defaults


def _fill_defaults_recursive(subschema, current_dict):
    """
    Recursively fill default values from the subschema into the current dictionary.

    Parameters
    ----------
    subschema: dict
        Subschema describing part of the input data.
    current_dict: dict
        Current dictionary to fill with default values.
    """
    if "properties" not in subschema:
        _raise_missing_properties_error()

    for prop, prop_schema in subschema["properties"].items():
        _process_property(prop, prop_schema, current_dict)


def _process_property(prop, prop_schema, current_dict):
    """
    Process each property and fill the default values accordingly.

    Parameters
    ----------
    prop: str
        Property name.
    prop_schema: dict
        Schema of the property.
    current_dict: dict
        Current dictionary to fill with default values.
    """
    if "default" in prop_schema:
        current_dict[prop] = prop_schema["default"]
    elif "type" in prop_schema:
        if prop_schema["type"] == "object":
            current_dict[prop] = {}
            _fill_defaults_recursive(prop_schema, current_dict[prop])
        elif prop_schema["type"] == "array":
            current_dict[prop] = [{}]
            if "items" in prop_schema and isinstance(prop_schema["items"], dict):
                _fill_defaults_recursive(prop_schema["items"], current_dict[prop][0])


def _raise_missing_properties_error():
    """Raise an error when the 'properties' key is missing in the schema."""
    msg = "Missing 'properties' key in schema."
    _logger.error(msg)
    raise KeyError(msg)
