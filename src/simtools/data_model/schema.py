"""Module providing functionality to read and validate dictionaries using schema."""

import logging
from pathlib import Path

import jsonschema
from referencing import Registry, Resource

import simtools.utils.general as gen
from simtools.constants import (
    METADATA_JSON_SCHEMA,
    MODEL_PARAMETER_METASCHEMA,
    MODEL_PARAMETER_SCHEMA_PATH,
    SCHEMA_PATH,
)
from simtools.data_model import format_checkers
from simtools.dependencies import get_software_version
from simtools.io import ascii_handler
from simtools.utils import names
from simtools.version import check_version_constraint

_logger = logging.getLogger(__name__)


def get_model_parameter_schema_files(schema_directory=MODEL_PARAMETER_SCHEMA_PATH):
    """
    Return list of parameters and schema files located in schema file directory.

    Returns
    -------
    list
        List of parameters found in schema file directory.
    list
        List of schema files found in schema file directory.

    """
    schema_files = sorted(Path(schema_directory).rglob("*.schema.yml"))
    if not schema_files:
        raise FileNotFoundError(f"No schema files found in {schema_directory}")
    parameters = []
    for schema_file in schema_files:
        # reading parameter 'name' only - first document in schema file should be ok
        schema_dict = ascii_handler.collect_data_from_file(file_name=schema_file, yaml_document=0)
        parameters.append(schema_dict.get("name"))
    return parameters, schema_files


def get_model_parameter_schema_file(parameter):
    """
    Return schema file path for a given model parameter.

    Parameters
    ----------
    parameter: str
        Model parameter name.

    Returns
    -------
    Path
        Schema file path.

    """
    schema_file = MODEL_PARAMETER_SCHEMA_PATH / f"{parameter}.schema.yml"
    if not schema_file.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_file}")
    return schema_file


def get_model_parameter_schema_version(schema_version=None):
    """
    Validate and return schema versions.

    If no schema_version is given, the most recent version is provided.

    Parameters
    ----------
    schema_version: str
        Schema version.

    Returns
    -------
    str
        Schema version.

    """
    schemas = ascii_handler.collect_data_from_file(MODEL_PARAMETER_METASCHEMA)

    if schema_version is None and schemas:
        return schemas[0].get("schema_version")

    if any(schema.get("schema_version") == schema_version for schema in schemas):
        return schema_version

    raise ValueError(f"Schema version {schema_version} not found in {MODEL_PARAMETER_METASCHEMA}.")


def validate_dict_using_schema(
    data, schema_file=None, json_schema=None, ignore_software_version=False, offline=False
):
    """
    Validate a data dictionary against a schema.

    Parameters
    ----------
    data
        dictionary to be validated
    schema_file (dict)
        schema used for validation
    json_schema (dict)
        schema used for validation
    ignore_software_version: bool
        If True, ignore software version check.

    Raises
    ------
    jsonschema.exceptions.ValidationError
        if validation fails

    """
    if json_schema is None and schema_file is None:
        _logger.warning(f"No schema provided for validation of {data}")
        return None
    if json_schema is None:
        json_schema = load_schema(schema_file, get_schema_version_from_data(data))

    validate_deprecation_and_version(data, ignore_software_version=ignore_software_version)

    validator = jsonschema.Draft6Validator(
        schema=json_schema,
        format_checker=format_checkers.format_checker,
        registry=Registry(retrieve=_retrieve_yaml_schema_from_uri),
    )

    try:
        validator.validate(instance=data)
    except jsonschema.exceptions.ValidationError as exc:
        _logger.error(f"Validation failed using schema: {json_schema} for data: {data}")
        raise exc

    if not offline:
        _validate_meta_schema_url(data)

    _logger.debug(f"Successful validation of data using schema ({json_schema.get('name')})")
    return data


def _validate_meta_schema_url(data):
    """Validate meta_schema_url if present in data."""
    if (
        isinstance(data, dict)
        and data.get("meta_schema_url")
        and not gen.url_exists(data["meta_schema_url"])
    ):
        raise FileNotFoundError(f"Meta schema URL does not exist: {data['meta_schema_url']}")


def _retrieve_yaml_schema_from_uri(uri):
    """Load schema from a file URI."""
    path = SCHEMA_PATH / Path(uri.removeprefix("file:/"))
    contents = ascii_handler.collect_data_from_file(file_name=path)
    return Resource.from_contents(contents)


def get_schema_version_from_data(data, observatory="cta"):
    """
    Get schema version from data dictionary.

    Parameters
    ----------
    data: dict
        data dictionary.

    Returns
    -------
    str
        Schema version. If not found, returns 'latest'.
    """
    schema_version = data.get("schema_version") or data.get("SCHEMA_VERSION")
    if schema_version:
        return schema_version
    reference_version = data.get(observatory.upper(), {}).get("REFERENCE", {}).get(
        "VERSION"
    ) or data.get(observatory.lower(), {}).get("reference", {}).get("version")
    if reference_version:
        return reference_version
    return "latest"


def load_schema(schema_file=None, schema_version="latest"):
    """
    Load parameter schema from file.

    Parameters
    ----------
    schema_file: str
        Path to schema file.
    schema_version: str
        Schema version.

    Returns
    -------
    schema: dict
        Schema dictionary.

    Raises
    ------
    FileNotFoundError
        if schema file is not found

    """
    schema_file = schema_file or METADATA_JSON_SCHEMA

    for path in (schema_file, SCHEMA_PATH / schema_file):
        try:
            schema = ascii_handler.collect_data_from_file(file_name=path)
            break
        except FileNotFoundError:
            continue
    else:
        raise FileNotFoundError(f"Schema file not found: {schema_file}")

    _logger.debug(f"Loading schema from {schema_file} for schema version {schema_version}")
    schema = _get_schema_for_version(schema, schema_file, schema_version)
    _add_array_elements("InstrumentTypeElement", schema)

    return schema


def _get_schema_for_version(schema, schema_file, schema_version):
    """
    Get schema for a specific version.

    Allow for 'latest' version to return the most recent schema.

    Parameters
    ----------
    schema: dict or list
        Schema dictionary or list of dictionaries.
    schema_file: str
        Path to schema file.
    schema_version: str or None
        Schema version to retrieve. If 'latest', the most recent version is returned.

    Returns
    -------
    dict
        Schema dictionary for the specified version.
    """
    if schema_version is None:
        raise ValueError(f"Schema version not given in {schema_file}.")

    if isinstance(schema, list):  # schema file with several schemas defined
        if len(schema) == 0:
            raise ValueError(f"No schemas found in {schema_file}.")
        if schema_version == "latest":
            schema_version = schema[0].get("schema_version")
        schema = next((doc for doc in schema if doc.get("schema_version") == schema_version), None)
    if schema is None:
        raise ValueError(f"Schema version {schema_version} not found in {schema_file}.")
    if schema_version not in (None, "latest") and schema_version != schema.get("schema_version"):
        _logger.warning(
            f"Schema version {schema_version} does not match {schema.get('schema_version')}"
        )
    return schema


def _get_array_element_list():
    """Build complete list of array elements including design types."""
    elements = set(names.array_elements().keys())
    for array_element in names.array_elements():
        for design_type in names.array_element_design_types(array_element):
            elements.add(f"{array_element}-{design_type}")
    return sorted(elements)


def _add_array_elements(key, schema):
    """
    Add list of array elements to schema.

    Avoids having to list all array elements in multiple schema.
    Assumes an element [key]['enum'] is a list of elements.

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
    array_elements = _get_array_element_list()

    def update_enum(sub_schema):
        if "enum" in sub_schema and isinstance(sub_schema["enum"], list):
            sub_schema["enum"] = list(set(sub_schema["enum"] + array_elements))
        else:
            sub_schema["enum"] = array_elements

    def recursive_search(sub_schema, target_key):
        if target_key in sub_schema:
            update_enum(sub_schema[target_key])
            return

        for v in sub_schema.values():
            if isinstance(v, dict):
                recursive_search(v, target_key)

    recursive_search(schema, key)
    return schema


def validate_deprecation_and_version(data, software_name=None, ignore_software_version=False):
    """
    Check if data contains deprecated parameters or version mismatches.

    Parameters
    ----------
    data: dict
        Data dictionary to check.
    software_name: str or None
        Name of the software to check version against. If None, use complete list
    ignore_software_version: bool
        If True, ignore software version check.
    """
    if not isinstance(data, dict):
        return

    data_name = data.get("name", "<unknown>")

    if data.get("deprecated", False):
        note = data.get("deprecation_note", "(no deprecation note provided)")
        _logger.warning(f"Data for {data_name} is deprecated. Note: {note}")

    for sw in data.get("simulation_software", []):
        name, constraint = sw.get("name"), sw.get("version")
        if not name or not constraint:
            continue
        if software_name is not None and name.lower() != software_name.lower():
            continue

        software_version = get_software_version(name)
        if check_version_constraint(software_version, constraint):
            _logger.debug(
                f"{data_name}: version {software_version} of {name} matches "
                f"constraint {constraint}."
            )
            continue

        msg = f"{data_name}: version {software_version} of {name} does not match {constraint}."
        if ignore_software_version:
            _logger.warning(f"{msg}, but version check is ignored.")
        else:
            raise ValueError(msg)
