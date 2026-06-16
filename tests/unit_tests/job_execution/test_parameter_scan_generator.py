"""Unit tests for parameter_scan_generator module."""

from pathlib import Path
from unittest import mock

import astropy.units as u
import pytest
import yaml

from simtools.job_execution import parameter_scan_generator


def _base_row(run_number=1):
    return {
        "primary": "gamma",
        "azimuth_angle": 0.0 * u.deg,
        "zenith_angle": 20.0 * u.deg,
        "energy_min": 0.02 * u.GeV,
        "energy_max": 0.025 * u.GeV,
        "core_scatter_number": 20,
        "core_scatter_max": 1900.0 * u.m,
        "view_cone_min": 0.0 * u.deg,
        "view_cone_max": 5.0 * u.deg,
        "showers_per_run": 1000,
        "model_version": "7.0.0",
        "array_layout_name": "LSTN-01",
        "corsika_le_interaction": "urqmd",
        "corsika_he_interaction": "epos",
        "run_number": run_number,
    }


def _write_scan_config(tmp_test_directory, scan_config):
    scan_config_path = Path(tmp_test_directory) / "scan_config.yml"

    with open(scan_config_path, "w", encoding="utf-8") as file_handle:
        yaml.safe_dump(scan_config, file_handle, sort_keys=False)

    return scan_config_path


def _inline_scan_config(tmp_test_directory, values=None):
    return {
        "label": "test_scan",
        "parameter_scan": {
            "overwrite": {
                "model_version": "7.0.0",
                "model_update": "patch_update",
                "model_version_history": ["7.0.0"],
                "description": "Base overwrite for test scan",
                "changes": {
                    "LSTN-01": {
                        "asum_threshold": {
                            "version": "2.0.0",
                            "value": 0,
                        }
                    }
                },
            },
            "parameters": [
                {
                    "name": "asum_threshold",
                    "path": "changes.LSTN-01.asum_threshold",
                    "values": values or [220, 230],
                }
            ],
        },
        "output_path": str(tmp_test_directory),
    }


@mock.patch("simtools.job_execution.parameter_scan_generator.serialize_job_grid")
@mock.patch("simtools.job_execution.parameter_scan_generator.read_job_grid")
def test_expand_job_grid_with_scan_writes_one_row_per_scan_value(
    mock_read_grid,
    mock_serialize_grid,
    tmp_test_directory,
):
    mock_read_grid.return_value = (
        [_base_row()],
        {"site": "North", "simulation_software": "corsika_sim_telarray"},
    )

    scan_config_path = _write_scan_config(
        tmp_test_directory,
        _inline_scan_config(tmp_test_directory, values=[220, 230]),
    )

    output_file = Path(tmp_test_directory) / "scan_grid.ecsv"

    parameter_scan_generator.expand_job_grid_with_scan(
        Path(tmp_test_directory) / "base_grid.ecsv",
        scan_config_path,
        output_file,
    )

    mock_serialize_grid.assert_called_once()

    expanded_rows, written_file = (
        mock_serialize_grid.call_args.args[0],
        mock_serialize_grid.call_args.args[1],
    )

    assert written_file == output_file
    assert len(expanded_rows) == 2
    assert expanded_rows[0]["scan_label"] == "asum_threshold_220"
    assert expanded_rows[1]["scan_label"] == "asum_threshold_230"
    assert "overwrite_model_parameters" in expanded_rows[0]
    assert "overwrite_model_parameters" in expanded_rows[1]

    overwrite_220 = Path(expanded_rows[0]["overwrite_model_parameters"])
    overwrite_230 = Path(expanded_rows[1]["overwrite_model_parameters"])

    assert overwrite_220.exists()
    assert overwrite_230.exists()
    assert overwrite_220.name == "overwrite_test_scan_asum_threshold_220.yaml"
    assert overwrite_230.name == "overwrite_test_scan_asum_threshold_230.yaml"


@mock.patch("simtools.job_execution.parameter_scan_generator.serialize_job_grid")
@mock.patch("simtools.job_execution.parameter_scan_generator.read_job_grid")
def test_expand_job_grid_with_scan_preserves_base_row_content(
    mock_read_grid,
    mock_serialize_grid,
    tmp_test_directory,
):
    base_row = _base_row(run_number=7)
    mock_read_grid.return_value = (
        [base_row],
        {"site": "North", "simulation_software": "corsika_sim_telarray"},
    )

    scan_config_path = _write_scan_config(
        tmp_test_directory,
        _inline_scan_config(tmp_test_directory, values=[220]),
    )

    output_file = Path(tmp_test_directory) / "scan_grid.ecsv"

    parameter_scan_generator.expand_job_grid_with_scan(
        Path(tmp_test_directory) / "base_grid.ecsv",
        scan_config_path,
        output_file,
    )

    expanded_rows = mock_serialize_grid.call_args.args[0]

    assert len(expanded_rows) == 1
    assert expanded_rows[0]["primary"] == base_row["primary"]
    assert expanded_rows[0]["run_number"] == 7
    assert expanded_rows[0]["array_layout_name"] == "LSTN-01"
    assert expanded_rows[0]["scan_label"] == "asum_threshold_220"


@mock.patch("simtools.job_execution.parameter_scan_generator.serialize_job_grid")
@mock.patch("simtools.job_execution.parameter_scan_generator.read_job_grid")
def test_expand_job_grid_with_scan_writes_overwrite_content(
    mock_read_grid,
    mock_serialize_grid,
    tmp_test_directory,
):
    mock_read_grid.return_value = (
        [_base_row()],
        {"site": "North", "simulation_software": "corsika_sim_telarray"},
    )

    scan_config_path = _write_scan_config(
        tmp_test_directory,
        _inline_scan_config(tmp_test_directory, values=[220]),
    )

    parameter_scan_generator.expand_job_grid_with_scan(
        Path(tmp_test_directory) / "base_grid.ecsv",
        scan_config_path,
        Path(tmp_test_directory) / "scan_grid.ecsv",
    )

    expanded_rows = mock_serialize_grid.call_args.args[0]
    overwrite_file = Path(expanded_rows[0]["overwrite_model_parameters"])

    with open(overwrite_file, encoding="utf-8") as file_handle:
        overwrite = yaml.safe_load(file_handle)

    assert overwrite["model_version"] == "7.0.0"
    assert overwrite["model_update"] == "patch_update"
    assert overwrite["changes"]["LSTN-01"]["asum_threshold"]["version"] == "2.0.0"
    assert overwrite["changes"]["LSTN-01"]["asum_threshold"]["value"] == 220
    assert "asum_threshold=220" in overwrite["description"]


@mock.patch("simtools.job_execution.parameter_scan_generator.serialize_job_grid")
@mock.patch("simtools.job_execution.parameter_scan_generator.read_job_grid")
def test_expand_job_grid_with_scan_multiple_base_rows(
    mock_read_grid,
    mock_serialize_grid,
    tmp_test_directory,
):
    mock_read_grid.return_value = (
        [_base_row(run_number=run_number) for run_number in range(1, 4)],
        {"site": "North", "simulation_software": "corsika_sim_telarray"},
    )

    scan_config_path = _write_scan_config(
        tmp_test_directory,
        _inline_scan_config(tmp_test_directory, values=[220, 230, 240]),
    )

    parameter_scan_generator.expand_job_grid_with_scan(
        Path(tmp_test_directory) / "base_grid.ecsv",
        scan_config_path,
        Path(tmp_test_directory) / "scan_grid.ecsv",
    )

    expanded_rows = mock_serialize_grid.call_args.args[0]

    assert len(expanded_rows) == 9
    assert expanded_rows[0]["scan_label"] == "asum_threshold_220"
    assert expanded_rows[3]["scan_label"] == "asum_threshold_230"
    assert expanded_rows[6]["scan_label"] == "asum_threshold_240"


@mock.patch("simtools.job_execution.parameter_scan_generator.serialize_job_grid")
@mock.patch("simtools.job_execution.parameter_scan_generator.read_job_grid")
def test_expand_job_grid_with_scan_cartesian_product(
    mock_read_grid,
    mock_serialize_grid,
    tmp_test_directory,
):
    mock_read_grid.return_value = (
        [_base_row()],
        {"site": "North", "simulation_software": "corsika_sim_telarray"},
    )

    scan_config = {
        "label": "test_scan",
        "parameter_scan": {
            "overwrite": {
                "model_version": "7.0.0",
                "model_update": "patch_update",
                "model_version_history": ["7.0.0"],
                "description": "Base overwrite for test scan",
                "changes": {
                    "LSTN-01": {
                        "asum_threshold": {
                            "version": "2.0.0",
                            "value": 0,
                        },
                        "min_photoelectrons": {
                            "version": "2.0.0",
                            "value": 0,
                        },
                    }
                },
            },
            "parameters": [
                {
                    "name": "asum_threshold",
                    "path": "changes.LSTN-01.asum_threshold",
                    "values": [220, 230],
                },
                {
                    "name": "min_photoelectrons",
                    "path": "changes.LSTN-01.min_photoelectrons",
                    "values": [0, 1],
                },
            ],
        },
        "output_path": str(tmp_test_directory),
    }

    scan_config_path = _write_scan_config(tmp_test_directory, scan_config)

    parameter_scan_generator.expand_job_grid_with_scan(
        Path(tmp_test_directory) / "base_grid.ecsv",
        scan_config_path,
        Path(tmp_test_directory) / "scan_grid.ecsv",
    )

    expanded_rows = mock_serialize_grid.call_args.args[0]

    assert len(expanded_rows) == 4
    assert expanded_rows[0]["scan_label"] == "asum_threshold_220_min_photoelectrons_0"
    assert expanded_rows[1]["scan_label"] == "asum_threshold_220_min_photoelectrons_1"
    assert expanded_rows[2]["scan_label"] == "asum_threshold_230_min_photoelectrons_0"
    assert expanded_rows[3]["scan_label"] == "asum_threshold_230_min_photoelectrons_1"


def test_expand_job_grid_with_scan_requires_existing_scan_config(tmp_test_directory):
    with pytest.raises(FileNotFoundError):
        parameter_scan_generator.expand_job_grid_with_scan(
            Path(tmp_test_directory) / "base_grid.ecsv",
            Path(tmp_test_directory) / "missing_scan_config.yml",
            Path(tmp_test_directory) / "scan_grid.ecsv",
        )
