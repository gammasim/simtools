"""Unit tests for parameter_scan_generator module."""

from pathlib import Path
from unittest import mock

import astropy.units as u
import pytest
import yaml

from simtools.job_execution.parameter_scan_generator import (
    _generate_overwrite_file,
    _generate_parameter_combinations,
    _parse_parameter_scan_config,
    _set_nested_value,
    expand_job_grid_with_scan,
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


@mock.patch("simtools.job_execution.parameter_scan_generator.serialize_job_grid")
@mock.patch("simtools.job_execution.parameter_scan_generator.read_job_grid")
def test_expand_job_grid_with_scan(mock_read_grid, mock_serialize_grid, tmp_test_directory):
    template_path = Path(tmp_test_directory) / "template.yaml"
    with open(template_path, "w", encoding="utf-8") as f:
        yaml.dump({"existing_key": "value"}, f)

    base_row = {
        "primary": "gamma",
        "azimuth_angle": 0.0 * u.deg,
        "zenith_angle": 20.0 * u.deg,
        "energy_min": 0.01 * u.TeV,
        "energy_max": 100.0 * u.TeV,
        "core_scatter_number": 5,
        "core_scatter_max": 1000.0 * u.m,
        "view_cone_min": 0.0 * u.deg,
        "view_cone_max": 5.0 * u.deg,
        "showers_per_run": 1000,
        "model_version": "6.0.1",
        "array_layout_name": "LSTN-01",
        "corsika_le_interaction": "urqmd",
        "corsika_he_interaction": "epos",
        "run_number": 1,
    }
    mock_read_grid.return_value = (
        [base_row],
        {"site": "North", "simulation_software": "simtools"},
    )

    scan_config = {
        "label": "test_scan",
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
    }
    scan_config_path = Path(tmp_test_directory) / "scan.yaml"
    with open(scan_config_path, "w", encoding="utf-8") as f:
        yaml.dump(scan_config, f)

    output_file = Path(tmp_test_directory) / "scan_grid.ecsv"
    expand_job_grid_with_scan(
        Path(tmp_test_directory) / "base_grid.ecsv",
        scan_config_path,
        output_file,
    )

    mock_serialize_grid.assert_called_once()
    expanded_rows, written_file = (
        mock_serialize_grid.call_args[0][0],
        mock_serialize_grid.call_args[0][1],
    )
    assert written_file == output_file
    assert len(expanded_rows) == 2  # 2 scan combos x 1 base row
    assert expanded_rows[0]["scan_label"] == "threshold_220"
    assert expanded_rows[1]["scan_label"] == "threshold_230"
    assert "overwrite_model_parameters" in expanded_rows[0]
    assert "overwrite_model_parameters" in expanded_rows[1]
    assert (Path(tmp_test_directory) / "overwrite_test_scan_threshold_220.yaml").exists()
    assert (Path(tmp_test_directory) / "overwrite_test_scan_threshold_230.yaml").exists()


@mock.patch("simtools.job_execution.parameter_scan_generator.serialize_job_grid")
@mock.patch("simtools.job_execution.parameter_scan_generator.read_job_grid")
def test_expand_job_grid_with_scan_multiple_base_rows(
    mock_read_grid, mock_serialize_grid, tmp_test_directory
):
    template_path = Path(tmp_test_directory) / "template.yaml"
    with open(template_path, "w", encoding="utf-8") as f:
        yaml.dump({}, f)

    base_row = {
        "primary": "proton",
        "azimuth_angle": 0.0 * u.deg,
        "zenith_angle": 20.0 * u.deg,
        "energy_min": 0.8 * u.TeV,
        "energy_max": 2.0 * u.TeV,
        "core_scatter_number": 10,
        "core_scatter_max": 2000.0 * u.m,
        "view_cone_min": 0.0 * u.deg,
        "view_cone_max": 10.0 * u.deg,
        "showers_per_run": 500,
        "model_version": "6.0.1",
        "array_layout_name": "LSTN-01",
        "corsika_le_interaction": "urqmd",
        "corsika_he_interaction": "epos",
        "run_number": 1,
    }
    mock_read_grid.return_value = (
        [{**base_row, "run_number": r} for r in range(1, 4)],
        {"site": "North", "simulation_software": "simtools"},
    )

    scan_config = {
        "label": "scan",
        "parameter_scan": {
            "overwrite_template": str(template_path),
            "parameters": [{"name": "thr", "path": "p.thr", "values": [220, 230, 240]}],
        },
    }
    scan_config_path = Path(tmp_test_directory) / "scan.yaml"
    with open(scan_config_path, "w", encoding="utf-8") as f:
        yaml.dump(scan_config, f)

    expand_job_grid_with_scan(
        Path(tmp_test_directory) / "base_grid.ecsv",
        scan_config_path,
        Path(tmp_test_directory) / "scan_grid.ecsv",
    )

    expanded_rows = mock_serialize_grid.call_args[0][0]
    # 3 scan combos x 3 base rows = 9
    assert len(expanded_rows) == 9
    assert expanded_rows[0]["scan_label"] == "thr_220"
    assert expanded_rows[3]["scan_label"] == "thr_230"
    assert expanded_rows[6]["scan_label"] == "thr_240"
