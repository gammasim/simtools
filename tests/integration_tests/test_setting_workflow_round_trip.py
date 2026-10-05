"""Check workflow creation, submission, conflicts, and reruns through the CLI."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from simtools.constants import METADATA_JSON_SCHEMA, MODEL_PARAMETER_METASCHEMA, SCHEMA_PATH
from simtools.data_model import schema
from simtools.io import ascii_handler
from simtools.testing import options


@pytest.mark.parametrize(
    ("parameter", "value", "expected", "unit"),
    [("min_photons", "121", 121, None), ("asum_threshold", "0.2 V", 200, "mV")],
)
def test_setting_workflow_round_trip(
    tmp_test_directory, simtools_root_path, request, parameter, value, expected, unit
):
    model_path = options.get_mirrored_option(request.config, "simulation_models_path")
    model_path = Path(model_path or simtools_root_path.parent / "simulation-models")
    if not model_path.is_absolute():
        model_path = simtools_root_path / model_path
    if not model_path.is_dir():
        pytest.skip("A simulation-models checkout is required.")
    root = Path(tmp_test_directory) / "settings"
    command = [
        sys.executable,
        "-m",
        "simtools.applications.create_setting_workflow",
        "--instrument",
        "LSTN-design",
        "--parameter",
        parameter,
        "--parameter_version",
        "99.0.1",
        "--value",
        value,
        "--description",
        "Example setting",
        "--source_url",
        "https://example.org/measurement",
        "--output_path",
        str(root),
        "--simulation_models_path",
        str(model_path.resolve()),
        "--disable_log_file",
    ]
    subprocess.run(command, capture_output=True, text=True, check=True)
    (config_file,) = (root / "input").rglob("config.yml")
    metadata_file = config_file.with_name("input.meta.yml")
    original_inputs = (config_file.read_bytes(), metadata_file.read_bytes())
    assert not (root / "output").exists()
    schema.validate_dict_using_schema(
        ascii_handler.collect_data_from_file(config_file),
        schema_file=SCHEMA_PATH / "application_workflow.metaschema.yml",
        offline=True,
    )
    subprocess.run([*command, "--run"], capture_output=True, text=True, check=True)
    (parameter_file,) = (root / "output").rglob(f"{parameter}-99.0.1.json")
    output = json.loads(parameter_file.read_text(encoding="utf-8"))
    assert output["value"] == pytest.approx(expected)
    assert output["unit"] == unit
    assert output["parameter_version"] == "99.0.1"
    schema.validate_dict_using_schema(output, schema_file=MODEL_PARAMETER_METASCHEMA, offline=True)
    output_metadata = ascii_handler.collect_data_from_file(parameter_file.with_suffix(".meta.yml"))
    schema.validate_dict_using_schema(
        output_metadata, schema_file=METADATA_JSON_SCHEMA, offline=True
    )
    assert output_metadata["cta"]["activity"]["id"] == config_file.parent.name
    assert (
        "https://example.org/measurement"
        in output_metadata["cta"]["context"]["associated_data"][0]["description"]
    )
    original_output = parameter_file.read_bytes()
    subprocess.run([*command, "--run"], capture_output=True, text=True, check=True)
    assert parameter_file.read_bytes() == original_output
    assert len(list((root / "output").rglob(f"{parameter}-99.0.1.json"))) == 2
    assert (config_file.read_bytes(), metadata_file.read_bytes()) == original_inputs
    changed = list(command)
    changed[changed.index("--value") + 1] = "122" if unit is None else f"122 {unit}"
    conflict = subprocess.run(changed, capture_output=True, text=True, check=False)
    assert conflict.returncode != 0
    assert "Conflicting inputs" in conflict.stderr
    assert (config_file.read_bytes(), metadata_file.read_bytes()) == original_inputs
