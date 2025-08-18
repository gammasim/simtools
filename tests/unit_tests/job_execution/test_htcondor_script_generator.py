from pathlib import Path
from unittest import mock

import astropy.units as u
import pytest

from simtools.job_execution.htcondor_script_generator import (
    _get_submit_script,
    generate_submission_script,
)


@pytest.fixture
def args_dict():
    return {
        "output_path": "/test_output",
        "apptainer_image": "/path/to/apptainer_image.sif",
        "priority": 5,
        "azimuth_angle": 45 * u.deg,
        "zenith_angle": 20 * u.deg,
        "energy_range": [1 * u.GeV, 10 * u.GeV],
        "core_scatter": [0, 100 * u.m],
        "label": "test_label",
        "simulation_software": "simtools",
        "model_version": "v1.0",
        "site": "test_site",
        "array_layout_name": "test_layout",
        "primary": "gamma",
        "nshow": 1000,
        "run_number_offset": 1,
        "run_number": 1,
        "number_of_runs": 10,
        "log_level": "INFO",
    }


@mock.patch("simtools.job_execution.htcondor_script_generator.Path.mkdir")
@mock.patch("simtools.job_execution.htcondor_script_generator.open", new_callable=mock.mock_open)
@mock.patch("simtools.job_execution.htcondor_script_generator.Path.chmod")
def test_generate_submission_script(mock_chmod, mock_open, mock_mkdir, args_dict):
    generate_submission_script(args_dict)

    work_dir = Path(args_dict["output_path"])
    submit_file_name = "simulate_prod.submit"

    mock_mkdir.assert_any_call(parents=True, exist_ok=True)

    # Check if files are created and written
    mock_open.assert_any_call(work_dir / f"{submit_file_name}.condor", "w", encoding="utf-8")
    mock_open.assert_any_call(work_dir / f"{submit_file_name}.sh", "w", encoding="utf-8")

    # Check if chmod is called
    mock_chmod.assert_called_once_with(0o755)


def test_get_submit_script(args_dict):
    # Use abbreviated argument names to avoid overly long lines
    e_range_low = args_dict["energy_range"][0].to(u.GeV).value
    e_range_high = args_dict["energy_range"][1].to(u.GeV).value
    core_scatter_low = args_dict["core_scatter"][0]
    core_scatter_high = args_dict["core_scatter"][1].to(u.m).value

    expected_script = f"""#!/usr/bin/env bash

# Process ID used to generate run number
process_id="$1"
# Load environment variables (for DB access)
set -a; source "$2"

simtools-simulate-prod \\
    --simulation_software {args_dict["simulation_software"]} \\
    --label {args_dict["label"]} \\
    --model_version {args_dict["model_version"]} \\
    --site {args_dict["site"]} \\
    --array_layout_name {args_dict["array_layout_name"]} \\
    --primary {args_dict["primary"]} \\
    --azimuth_angle {args_dict["azimuth_angle"].to(u.deg).value} \\
    --zenith_angle {args_dict["zenith_angle"].to(u.deg).value} \\
    --nshow {args_dict["nshow"]} \\
    --energy_range "{e_range_low} GeV {e_range_high} GeV" \\
    --core_scatter "{core_scatter_low} {core_scatter_high} m" \\
    --run_number $((process_id)) \\
    --run_number_offset {args_dict["run_number_offset"]} \\
    --number_of_runs 1 \\
    --data_directory /tmp/simtools-data \\
    --output_path /tmp/simtools-output \\
    --log_level {args_dict["log_level"]} \\
    --pack_for_grid_register simtools-output
"""
    generated_script = _get_submit_script(args_dict)
    assert generated_script == expected_script
