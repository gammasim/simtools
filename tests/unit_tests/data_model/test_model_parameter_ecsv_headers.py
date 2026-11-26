#!/usr/bin/python3

"""Checks the metadata headers of model-parameter ecsv files in a simulation-models checkout."""

import json
import os
import warnings
from pathlib import Path

import pytest
import yaml
from astropy.table import Table

from simtools.constants import MODEL_PARAMETER_SCHEMA_PATH

warnings.filterwarnings("ignore", category=UserWarning, module="astropy")


@pytest.fixture(scope="module")
def model_parameter_ecsv_files():
    """Return ecsv model-parameter files of the simulation-models checkout.

    Uses SIMTOOLS_SIMULATION_MODELS_PATH (repository root containing simulation-models/).
    """
    root = os.getenv("SIMTOOLS_SIMULATION_MODELS_PATH")
    if not root:
        pytest.skip("SIMTOOLS_SIMULATION_MODELS_PATH not set; no simulation-models checkout")
    parameters_path = Path(root) / "simulation-models" / "model_parameters"
    if not parameters_path.is_dir():
        pytest.skip(f"No model_parameters directory at {parameters_path}")
    return sorted(parameters_path.glob("**/*.ecsv"))


def _header_meta(path):
    """Return the metadata of an ecsv file as read by astropy."""
    return dict(Table.read(path, format="ascii.ecsv").meta)


def _schema_versions(parameter_name):
    """Return the model parameter schema versions defined in the simtools schema file."""
    schema_file = MODEL_PARAMETER_SCHEMA_PATH / f"{parameter_name}.schema.yml"
    documents = [
        doc for doc in yaml.safe_load_all(schema_file.read_text()) if isinstance(doc, dict)
    ]
    return {str(doc.get("schema_version")) for doc in documents}


def test_description_key_is_lowercase_and_single(model_parameter_ecsv_files):
    """Each header has exactly one description key, spelled 'description', with text."""
    problems = []
    for path in model_parameter_ecsv_files:
        keys = [key for key in _header_meta(path) if key.lower().startswith("description")]
        if keys != ["description"]:
            problems.append(f"{path.name}: description keys {keys}")
        elif not str(_header_meta(path)["description"]).strip():
            problems.append(f"{path.name}: empty description")
    assert not problems, "\n".join(problems)


def test_schema_version_is_present_and_defined(model_parameter_ecsv_files):
    """Each header names a model_parameter_schema_version defined in the simtools schema."""
    problems = []
    for path in model_parameter_ecsv_files:
        meta = _header_meta(path)
        version = meta.get("model_parameter_schema_version")
        if version is None:
            problems.append(f"{path.name}: no model_parameter_schema_version")
        elif str(version) not in _schema_versions(meta["parameter_name"]):
            problems.append(f"{path.name}: version {version} not in simtools schema")
    assert not problems, "\n".join(problems)


def test_schema_version_matches_json_twin(model_parameter_ecsv_files):
    """The ecsv header schema version equals the one of the JSON record next to it."""
    problems = []
    for path in model_parameter_ecsv_files:
        twin = path.with_suffix(".json")
        if not twin.exists():
            continue
        json_version = json.loads(twin.read_text()).get("model_parameter_schema_version")
        ecsv_version = _header_meta(path).get("model_parameter_schema_version")
        if str(ecsv_version) != str(json_version):
            problems.append(f"{path.name}: ecsv {ecsv_version} vs json {json_version}")
    assert not problems, "\n".join(problems)


def test_correct_nsb_spectrum_description_describes_the_parameter(model_parameter_ecsv_files):
    """The NSB altitude correction files do not carry the atmospheric transmission text."""
    nsb_files = [
        path
        for path in model_parameter_ecsv_files
        if path.parent.name == "correct_nsb_spectrum_to_telescope_altitude"
    ]
    problems = [
        f"{path.parent.parent.name}/{path.name}: {_header_meta(path)['description']!r}"
        for path in nsb_files
        if "night-sky background" not in str(_header_meta(path)["description"])
    ]
    assert nsb_files, "no correct_nsb_spectrum_to_telescope_altitude files found"
    assert not problems, "\n".join(problems)
