"""Unit tests for parameter_scan_generator module."""

from pathlib import Path
from unittest import mock

import pytest
import yaml

from simtools.job_execution.parameter_scan_generator import (
    _generate_condor_submit_file,
    _generate_overwrite_file,
    _generate_parameter_combinations,
    _generate_submit_script,
    _parse_parameter_scan_config,
    _set_nested_value,
    generate_parameter_scan_htcondor,
)


def test_set_nested_value():
    data = {}
    _set_nested_value(data, ["level1", "level2", "key"], "value")
    assert data == {"level1": {"level2": {"key": {"value": "value"}}}}


def test_set_nested_value_existing_path():
    data = {"level1": {"existing": {"value": "old"}}}
    _set_nested_value(data, ["level1", "level2", "key"], "value")
    assert data["level1"]["existing"] == {"value": "old"}
    assert data["level1"]["level2"]["key"] == {"value": "value"}


def test_generate_overwrite_file(tmp_test_directory):
    template_path = Path(tmp_test_directory) / "template.yaml"
    template_data = {"existing_key": "existing_value"}
    with open(template_path, "w", encoding="utf-8") as f:
        yaml.dump(template_data, f)

    param_combo = {
        "threshold": ("changes.LSTN-01.asum_threshold", 220),
    }
    combo_name = "threshold_220"
    label = "test_label"

    result = _generate_overwrite_file(
        template_path, param_combo, combo_name, Path(tmp_test_directory), label
    )

    assert result.exists()
    assert result.name == "overwrite_test_label_threshold_220.yaml"

    with open(result, encoding="utf-8") as f:
        content = yaml.safe_load(f)

    assert "existing_key" in content
    assert content["existing_key"] == "existing_value"
    assert "description" in content
    assert "threshold=220" in content["description"]
    assert "changes" in content
    assert content["changes"]["LSTN-01"]["asum_threshold"]["value"] == 220


def test_generate_overwrite_file_missing_template(tmp_test_directory):
    template_path = Path(tmp_test_directory) / "nonexistent.yaml"
    param_combo = {"threshold": ("path", 220)}

    with pytest.raises(FileNotFoundError, match="Overwrite template file not found"):
        _generate_overwrite_file(
            template_path, param_combo, "name", Path(tmp_test_directory), "label"
        )


def test_parse_parameter_scan_config_single_parameter(tmp_test_directory):
    template_path = Path(tmp_test_directory) / "template.yaml"
    template_path.touch()

    param_scan = {
        "overwrite_template": str(template_path),
        "parameters": [
            {
                "name": "threshold",
                "path": "changes.LSTN-01.asum_threshold",
                "values": [220, 230, 240],
            }
        ],
    }

    params, parsed_template = _parse_parameter_scan_config(param_scan)

    assert len(params) == 1
    assert params[0]["name"] == "threshold"
    assert params[0]["path"] == "changes.LSTN-01.asum_threshold"
    assert params[0]["values"] == [220, 230, 240]
    assert parsed_template == template_path


def test_parse_parameter_scan_config_multiple_parameters(tmp_test_directory):
    template_path = Path(tmp_test_directory) / "template.yaml"
    template_path.touch()

    param_scan = {
        "overwrite_template": str(template_path),
        "parameters": [
            {"name": "threshold", "path": "path1", "values": [220, 230]},
            {"name": "test", "path": "path2", "values": [5.0, 5.5]},
        ],
    }

    params, _ = _parse_parameter_scan_config(param_scan)

    assert len(params) == 2
    assert params[0]["name"] == "threshold"
    assert params[1]["name"] == "test"


def test_generate_parameter_combinations_single_parameter():
    param_specs = [
        {"name": "threshold", "path": "path", "values": [220, 230, 240]},
    ]

    combos = _generate_parameter_combinations(param_specs)

    assert len(combos) == 3
    assert combos[0]["name"] == "threshold_220"
    assert combos[0]["combo"]["threshold"] == ("path", 220)
    assert combos[1]["name"] == "threshold_230"
    assert combos[2]["name"] == "threshold_240"


def test_generate_parameter_combinations_cartesian_product():
    param_specs = [
        {"name": "threshold", "path": "path1", "values": [220, 230]},
        {"name": "test", "path": "path2", "values": [5.0, 5.5]},
    ]

    combos = _generate_parameter_combinations(param_specs)

    assert len(combos) == 4
    assert combos[0]["name"] == "threshold_220_test_5.0"
    assert combos[1]["name"] == "threshold_220_test_5.5"
    assert combos[2]["name"] == "threshold_230_test_5.0"
    assert combos[3]["name"] == "threshold_230_test_5.5"


def test_generate_parameter_combinations_three_parameters():
    param_specs = [
        {"name": "p1", "path": "path1", "values": [1, 2]},
        {"name": "p2", "path": "path2", "values": [3, 4]},
        {"name": "p3", "path": "path3", "values": [5, 6]},
    ]

    combos = _generate_parameter_combinations(param_specs)

    assert len(combos) == 8


def test_generate_submit_script_basic():
    sim_params = {
        "simulation_software": "simtools",
        "site": "North",
        "model_version": "6.0.1",
        "array_layout_name": "LSTN-01",
        "primary": "gamma",
        "azimuth_angle": 0,
        "zenith_angle": 20,
        "nshow": 1000,
        "energy_range": "0.01 100 GeV",
        "core_scatter": "0 1000 m",
        "view_cone": "0 5 deg",
        "corsika_le_interaction": "urqmd",
        "corsika_he_interaction": "epos",
        "label": "test",
    }

    script = _generate_submit_script(sim_params)

    assert "#!/usr/bin/env bash" in script
    assert "OVERWRITE_FILE=" in script
    assert "RUN_NUMBER=" in script
    assert "COMBO_LABEL=" in script
    assert 'FULL_LABEL="test_$COMBO_LABEL"' in script
    assert "simtools-simulate-prod" in script
    assert "--site North" in script
    assert "--model_version 6.0.1" in script
    assert '--label "$FULL_LABEL"' in script
    assert "--output_path /tmp/simtools-output" in script


def test_generate_submit_script_with_optional_params():
    sim_params = {
        "simulation_software": "simtools",
        "site": "South",
        "model_version": "7.0.0",
        "array_layout_name": "MSTS-01",
        "primary": "proton",
        "azimuth_angle": 45,
        "zenith_angle": 30,
        "nshow": 5000,
        "energy_range": "1 1000 GeV",
        "core_scatter": "0 2000 m",
        "view_cone": "0 10 deg",
        "corsika_le_interaction": "urqmd",
        "corsika_he_interaction": "epos",
        "label": "proton_scan",
        "run_number_offset": 100,
        "save_reduced_event_lists": True,
        "pack_for_grid_register": "some_value",
    }

    script = _generate_submit_script(sim_params)

    assert "--run_number_offset 100" in script
    assert "--save_reduced_event_lists" in script
    assert "--pack_for_grid_register some_value" in script


def test_generate_submit_script_with_custom_output_path():
    sim_params = {
        "simulation_software": "simtools",
        "site": "North",
        "model_version": "6.0.1",
        "array_layout_name": "LSTN-01",
        "primary": "gamma",
        "azimuth_angle": 0,
        "zenith_angle": 20,
        "nshow": 1000,
        "energy_range": "0.01 100 GeV",
        "core_scatter": "0 1000 m",
        "view_cone": "0 5 deg",
        "corsika_le_interaction": "urqmd",
        "corsika_he_interaction": "epos",
        "label": "test",
        "output_path": "/custom/output/path",
    }

    script = _generate_submit_script(sim_params)

    assert "--output_path /custom/output/path" in script


def test_generate_condor_submit_file(tmp_test_directory):
    script_name = "test_script.sh"
    apptainer_image = Path("/path/to/image.sif")
    priority = 5
    param_file = "params.txt"

    result = _generate_condor_submit_file(
        script_name, apptainer_image, priority, param_file, Path(tmp_test_directory)
    )

    assert "universe = container" in result
    assert f"container_image = {apptainer_image}" in result
    assert f"executable = {script_name}" in result
    assert "arguments = $(overwrite_file) $(run_number) $(combo_label)" in result
    assert f"priority = {priority}" in result
    assert f"queue overwrite_file,run_number,combo_label from {param_file}" in result

    log_dir = Path(tmp_test_directory) / "htcondor_logs"
    assert (log_dir / "log").exists()
    assert (log_dir / "error").exists()
    assert (log_dir / "output").exists()


@pytest.fixture
def scan_config(tmp_test_directory):
    template_path = Path(tmp_test_directory) / "overwrite_template.yaml"
    with open(template_path, "w", encoding="utf-8") as f:
        yaml.dump({"existing": "value"}, f)

    config = {
        "simulation": {
            "simulation_software": "simtools",
            "site": "North",
            "model_version": "6.0.1",
            "array_layout_name": "LSTN-01",
            "primary": "gamma",
            "azimuth_angle": 0,
            "zenith_angle": 20,
            "nshow": 1000,
            "energy_range": "0.01 100 GeV",
            "core_scatter": "0 1000 m",
            "view_cone": "0 5 deg",
            "corsika_le_interaction": "urqmd",
            "corsika_he_interaction": "epos",
            "label": "test_scan",
            "number_of_runs": 2,
            "run_number": 10,
        },
        "parameter_scan": {
            "overwrite_template": str(template_path),
            "parameters": [
                {
                    "name": "threshold",
                    "path": "changes.LSTN-01.asum_threshold",
                    "values": [220, 230],
                }
            ],
        },
        "htcondor": {
            "apptainer_image": "/path/to/image.sif",
            "output_path": str(Path(tmp_test_directory) / "output"),
            "priority": 5,
        },
    }

    config_path = Path(tmp_test_directory) / "scan_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    return config_path


@mock.patch("simtools.job_execution.parameter_scan_generator._resolve_apptainer_images")
def test_generate_complete_submission(mock_resolve_images, scan_config, tmp_test_directory):
    mock_resolve_images.return_value = {"default": Path("/path/to/image.sif")}

    generate_parameter_scan_htcondor(scan_config)

    output_path = Path(tmp_test_directory) / "output"

    overwrite_files = list(output_path.glob("overwrite_*.yaml"))
    assert len(overwrite_files) == 2

    params_file = output_path / "scan_parameters_test_scan.txt"
    assert params_file.exists()

    with open(params_file, encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 4
    for line in lines:
        parts = line.strip().split(", ")
        assert len(parts) == 3

    script_file = output_path / "simulate_prod_scan_test_scan.sh"
    assert script_file.exists()

    condor_file = output_path / "simulate_prod_scan_test_scan.condor"
    assert condor_file.exists()


@mock.patch("simtools.job_execution.parameter_scan_generator._resolve_apptainer_images")
def test_generate_multi_parameter_scan(mock_resolve_images, tmp_test_directory):
    mock_resolve_images.return_value = {"default": Path("/path/to/image.sif")}

    template_path = Path(tmp_test_directory) / "template.yaml"
    with open(template_path, "w", encoding="utf-8") as f:
        yaml.dump({}, f)

    config = {
        "simulation": {
            "simulation_software": "simtools",
            "site": "North",
            "model_version": "6.0.1",
            "array_layout_name": "LSTN-01",
            "primary": "gamma",
            "azimuth_angle": 0,
            "zenith_angle": 20,
            "nshow": 1000,
            "energy_range": "0.01 100 GeV",
            "core_scatter": "0 1000 m",
            "view_cone": "0 5 deg",
            "corsika_le_interaction": "urqmd",
            "corsika_he_interaction": "epos",
            "label": "multi_param",
            "number_of_runs": 1,
        },
        "parameter_scan": {
            "overwrite_template": str(template_path),
            "parameters": [
                {"name": "threshold", "path": "path1", "values": [220, 230]},
                {"name": "test", "path": "path2", "values": [5.0, 5.5]},
            ],
        },
        "htcondor": {
            "apptainer_image": "/path/to/image.sif",
            "output_path": str(Path(tmp_test_directory) / "output"),
            "priority": 3,
        },
    }

    config_path = Path(tmp_test_directory) / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    generate_parameter_scan_htcondor(config_path)

    output_path = Path(tmp_test_directory) / "output"

    overwrite_files = list(output_path.glob("overwrite_*.yaml"))
    assert len(overwrite_files) == 4

    params_file = output_path / "scan_parameters_multi_param.txt"
    with open(params_file, encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 4
