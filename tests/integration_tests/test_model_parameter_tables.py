"""Repository-level validation for the canonical model-parameter assets."""

import json
import os
from pathlib import Path

import pytest
from astropy.table import Table

from simtools.data_model import schema
from simtools.data_model.json_validation import validate_finite_json_values
from simtools.data_model.table_asset import get_simtel_serialization
from simtools.model_repository.reader import SimulationModelReader
from simtools.simtel import simtel_table_writer
from simtools.utils import names


def _model_path(request, simtools_root_path):
    """Return the configured local simulation-model repository, if any."""
    configured = request.config.getoption("simulation_models_path", default=None)
    configured = configured or os.environ.get("SIMTOOLS_SIMULATION_MODELS_PATH")
    path = (
        Path(configured)
        if configured
        else Path(simtools_root_path).parent / "simulation-models-dev1"
    )
    return path if path.is_absolute() else Path(simtools_root_path) / path


def _simtel_contract(parameter_data):
    """Return a table contract for a parameter, or ``None`` for non-table data."""
    parameter_schema = schema.get_model_parameter_schema(
        parameter_data["parameter"], parameter_data.get("model_parameter_schema_version")
    )
    try:
        return get_simtel_serialization(parameter_schema)
    except ValueError:
        return None


def _write_deterministic_table(table, contract, destination, output_name):
    """Write a table twice after permuting input layout and compare bytes."""
    first = destination / f"first-{output_name}"
    second = destination / f"second-{output_name}"
    simtel_table_writer.write_simtel_table(
        table, destination, contract=contract, output_name=first.name
    )
    permuted = Table(table[table.colnames[::-1]][::-1], copy=True)
    simtel_table_writer.write_simtel_table(
        permuted, destination, contract=contract, output_name=second.name
    )
    assert first.read_bytes() == second.read_bytes()
    return {first.name, second.name}


def _validate_parameter(parameter_data, reader, destination, output_stem):
    """Validate one resolved parameter and serialize its simulator assets."""
    generated_names = set()
    value = parameter_data.get("value")
    validate_finite_json_values(value)
    parameter_schema = schema.get_model_parameter_schema(
        parameter_data["parameter"], parameter_data.get("model_parameter_schema_version")
    )
    if parameter_data.get("file") and isinstance(value, str) and value.lower().endswith(".ecsv"):
        table = reader.get_parameter_table(parameter_data)
        contract = _simtel_contract(parameter_data)
        if contract is not None:
            generated_names.update(
                _write_deterministic_table(
                    table,
                    contract,
                    destination,
                    f"{parameter_data['parameter']}-{output_stem}.dat",
                )
            )
    data_entry = next(
        (entry for entry in parameter_schema.get("data", []) if entry.get("type") == "dict"),
        None,
    )
    if data_entry is not None and parameter_data.get("parameter", "").endswith("segmentation"):
        simtel = next(
            (
                entry
                for entry in parameter_schema.get("simulation_software", [])
                if entry.get("name") == "sim_telarray"
            ),
            None,
        )
        if simtel is not None:
            generated_names.add(
                simtel_table_writer.write_mirror_segmentation(
                    value,
                    destination / f"{parameter_data['parameter']}-{output_stem}.dat",
                    parameter_data["parameter"],
                    parameter_data["model_parameter_schema_version"],
                )
            )
    return generated_names


def _production_parameters(reader):
    """Yield each resolved production parameter asset once."""
    seen = set()
    for model_version in reader.get_model_versions("telescopes"):
        yield from _version_parameters(reader, model_version, seen)


def _version_parameters(reader, model_version, seen):
    """Yield unique assets selected by one model version."""
    for array_element in reader.get_array_elements(model_version, "telescopes"):
        yield from _element_parameters(reader, model_version, array_element, seen)


def _element_parameters(reader, model_version, array_element, seen):
    """Yield unique assets selected by one array element."""
    sites = names.get_site_from_array_element_name(array_element)
    sites = sites if isinstance(sites, list) else [sites]
    for site in sites:
        parameters = reader.get_model_parameters(site, array_element, "telescopes", model_version)
        for parameter_data in parameters.values():
            identity = json.dumps(
                {
                    "parameter": parameter_data["parameter"],
                    "parameter_version": parameter_data.get("parameter_version"),
                    "instrument": parameter_data.get("instrument"),
                    "site": parameter_data.get("site"),
                },
                sort_keys=True,
            )
            if identity not in seen:
                seen.add(identity)
                yield parameter_data, f"{model_version}-{array_element}-{site}"


def _serialize_production_assets(reader, destination):
    """Serialize production assets and verify generated names are unique."""
    generated_names = set()
    for parameter_data, output_stem in _production_parameters(reader):
        for generated_name in _validate_parameter(parameter_data, reader, destination, output_stem):
            assert generated_name not in generated_names
            generated_names.add(generated_name)


def test_production_model_parameter_assets_are_canonical(
    request, simtools_root_path, tmp_test_directory
):
    """Validate and serialize every local production model asset deterministically."""
    model_path = _model_path(request, simtools_root_path)
    if not model_path.is_dir():
        pytest.skip("No local simulation-model repository is configured")

    reader = SimulationModelReader.from_files(model_path)
    destination = Path(tmp_test_directory) / "model-assets"
    destination.mkdir()
    _serialize_production_assets(reader, destination)
