"""Build small simulation-model repositories from packaged parameter schemas."""

import json
from pathlib import Path, PurePosixPath

from simtools.constants import SCHEMA_PATH
from simtools.data_model import parameter_catalog, schema
from simtools.model_repository.reader import SimulationModelReader
from tests.unit_tests.builders.parameters import parameter

_PARAMETERS_PATH = PurePosixPath("simulation-models/model_parameters")
_PRODUCTIONS_PATH = PurePosixPath("simulation-models/productions")
_PRODUCTION_TABLE_SCHEMA = SCHEMA_PATH / "production_tables.schema.yml"


def model_repository_files(
    model_version="6.0.0",
    elements=("LSTN-01", "MSTS-01"),
    sites=("North", "South"),
    overrides=None,
):
    """Return simulation-model repository paths and encoded file contents."""
    overrides = overrides or {}
    files = {}
    telescope_parameters = parameter_catalog.model_parameters(
        parameter_catalog.collection_to_instrument_class_key("telescopes")
    )
    site_parameters = parameter_catalog.site_parameters()

    for element in elements:
        records, parameters = _build_parameter_records(
            telescope_parameters, element, overrides, "telescopes"
        )
        _add_production_table(files, model_version, element, records, parameters)

    for site in sites:
        instrument = f"OBS-{site}"
        records, parameters = _build_parameter_records(
            site_parameters, instrument, overrides, "sites", site
        )
        _add_production_table(files, model_version, instrument, records, parameters)

    return files


def model_reader(
    model_version="6.0.0",
    elements=("LSTN-01", "MSTS-01"),
    sites=("North", "South"),
    overrides=None,
):
    """Return a memory-backed reader for a generated model repository."""
    return SimulationModelReader.from_memory(
        model_repository_files(model_version, elements, sites, overrides)
    )


def write_model_repository(
    destination,
    model_version="6.0.0",
    elements=("LSTN-01", "MSTS-01"),
    sites=("North", "South"),
    overrides=None,
):
    """Write generated repository files below a directory destination."""
    destination = Path(destination)
    for relative_path, contents in model_repository_files(
        model_version, elements, sites, overrides
    ).items():
        file_path = destination / PurePosixPath(relative_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(contents)


def _build_parameter_records(definitions, instrument, overrides, collection, site=None):
    """Build records and repository paths for the parameters in one scope."""
    records = {}
    parameter_versions = {}
    for name, definition in definitions.items():
        if definition["data"][0]["type"] == "data_table":
            continue
        override_scope = site or instrument
        value = overrides.get((override_scope, name))
        if collection == "sites":
            record = parameter(name, value=value, site=site, instrument=instrument, version="1.0.0")
        else:
            record = parameter(name, value=value, instrument=instrument, version="1.0.0")
        path = _parameter_path(collection, instrument, name, "1.0.0")
        records[path] = _encode(record)
        parameter_versions[name] = "1.0.0"
    return records, parameter_versions


def _parameter_path(collection, instrument, name, version):
    """Return the model repository path for a generated parameter record."""
    scope = parameter_catalog.get_model_parameter_scope(collection, instrument, name)
    return (_PARAMETERS_PATH / scope / name / f"{name}-{version}.json").as_posix()


def _add_production_table(files, model_version, instrument, records, parameters):
    """Validate and add one production table and its parameter records."""
    production = {
        "production_table_name": instrument,
        "model_version": model_version,
        "parameters": {instrument: parameters},
    }
    schema.validate_dict_using_schema(
        production, schema_file=_PRODUCTION_TABLE_SCHEMA, offline=True
    )
    production_path = (_PRODUCTIONS_PATH / model_version / f"{instrument}.json").as_posix()
    files[production_path] = _encode(production)
    files.update(records)


def _encode(data):
    """Encode a JSON repository document as UTF-8 bytes."""
    return json.dumps(data, indent=2).encode("utf-8")
