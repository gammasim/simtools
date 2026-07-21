"""Cached, dependency-neutral loading of schema definitions."""

import logging
from functools import cache
from pathlib import Path

import simtools.utils.general as gen
from simtools.constants import SCHEMA_PATH
from simtools.io import ascii_handler

_logger = logging.getLogger(__name__)


def get_model_parameter_schema_files(schema_directory):
    """
    Return model-parameter names and schema files from a directory.

    Parameters
    ----------
    schema_directory : str or Path
        Directory containing model-parameter schema files.

    Returns
    -------
    list
        Model-parameter names.
    list
        Corresponding schema file paths.

    Raises
    ------
    FileNotFoundError
        If the directory contains no model-parameter schema files.
    """
    schema_files = sorted(Path(schema_directory).rglob("*.schema.yml"))
    if not schema_files:
        raise FileNotFoundError(f"No schema files found in {schema_directory}")
    parameters = [load_schema(schema_file, "latest").get("name") for schema_file in schema_files]
    return parameters, schema_files


@cache
def load_schema(schema_file, schema_version="latest"):
    """
    Load and cache an immutable schema definition by source and version.

    Parameters
    ----------
    schema_file : str or Path
        Local path or URL of the schema file.
    schema_version : str
        Schema version to return, or ``latest`` for the first document.

    Returns
    -------
    dict
        Shared, immutable schema definition.

    Raises
    ------
    FileNotFoundError
        If the schema cannot be loaded locally or from its URL.
    ValueError
        If the requested schema version is unavailable.
    """
    schema = None
    for path in _get_local_schema_candidates(schema_file):
        try:
            schema = ascii_handler.collect_data_from_file(file_name=path)
            break
        except FileNotFoundError:
            continue

    if schema is None and gen.is_url(str(schema_file)):
        try:
            schema = ascii_handler.collect_data_from_file(file_name=schema_file)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Schema file not found: {schema_file}") from exc

    if schema is None:
        raise FileNotFoundError(f"Schema file not found: {schema_file}")

    return get_schema_for_version(schema, schema_file, schema_version)


def _get_local_schema_candidates(schema_file):
    """Build local candidate paths for a schema file reference."""
    schema_path = Path(str(schema_file))
    candidates = [schema_path, SCHEMA_PATH / schema_path]

    if gen.is_url(str(schema_file)):
        schema_name = Path(str(schema_file)).name
        if schema_name:
            candidates.extend([SCHEMA_PATH / schema_name, Path(schema_name)])

    return list(dict.fromkeys(candidates))


def get_schema_for_version(schema, schema_file, schema_version):
    """
    Return a requested version from a parsed schema definition.

    Parameters
    ----------
    schema : dict or list
        Parsed schema definition or versioned schema documents.
    schema_file : str or Path
        Schema source used in error and warning messages.
    schema_version : str
        Requested schema version, or ``latest`` for the first document.

    Returns
    -------
    dict
        Selected schema definition.

    Raises
    ------
    ValueError
        If no version is requested, the schema list is empty, or the requested version is absent.
    """
    if schema_version is None:
        raise ValueError(f"Schema version not given in {schema_file}.")

    if isinstance(schema, list):
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
