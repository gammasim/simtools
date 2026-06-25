from pathlib import Path
from unittest import mock

import astropy.units as u
import pytest

from simtools.job_execution import script_job_generator


@pytest.fixture
def args_dict(tmp_test_directory):
    return {
        "output_path": str(Path(tmp_test_directory) / "output"),
        "job_grid_file": str(Path(tmp_test_directory) / "job_grid.ecsv"),
        "job_grid_line": 5,
        "run_script": False,
        "label": "test_label",
        "simulation_software": "corsika_sim_telarray",
        "site": "North",
        "log_level": "INFO",
        "simulation_output": "simtools-output",
        "run_number_offset": 0,
    }


@pytest.fixture
def job_rows():
    rows = []
    for run_number in range(1, 7):
        rows.append(
            {
                "primary": "gamma",
                "azimuth_angle": 45 * u.deg,
                "zenith_angle": 20 * u.deg,
                "energy_min": 1 * u.GeV,
                "energy_max": 10 * u.GeV,
                "cores_per_shower": 10,
                "core_scatter_max": 100 * u.m,
                "view_cone_min": 0 * u.deg,
                "view_cone_max": 5 * u.deg,
                "showers_per_run": 1000,
                "model_version": "7.0.0",
                "array_layout_name": "CTAO-North-Alpha",
                "corsika_le_interaction": "urqmd",
                "corsika_he_interaction": "epos",
                "run_number": run_number,
            }
        )
    return rows


@pytest.fixture
def job_grid_metadata():
    return {
        "site": "North",
        "simulation_software": "corsika_sim_telarray",
    }


@mock.patch("simtools.job_execution.htcondor_script_generator.read_job_grid")
def test_generate_script_jobs_writes_selected_line_script(
    mock_read_job_grid, args_dict, job_rows, job_grid_metadata
):
    mock_read_job_grid.return_value = (job_rows, job_grid_metadata)

    script_job_generator.generate_script_jobs(args_dict)

    script_path = Path(args_dict["output_path"]) / "simulate_prod.job_5.sh"
    script_content = script_path.read_text(encoding="utf-8")

    assert script_path.is_file()
    assert "simtools-simulate-prod" in script_content
    assert "--run_number \\\n    5" in script_content
    assert "--pack_for_grid_register \\\n    simtools-output/default" in script_content
    assert script_path.stat().st_mode & 0o100


@mock.patch("simtools.job_execution.htcondor_script_generator.read_job_grid")
def test_generate_script_jobs_writes_all_line_scripts(
    mock_read_job_grid, args_dict, job_rows, job_grid_metadata
):
    args_dict["job_grid_line"] = None
    mock_read_job_grid.return_value = (job_rows[:2], job_grid_metadata)

    script_job_generator.generate_script_jobs(args_dict)

    assert (Path(args_dict["output_path"]) / "simulate_prod.job_1.sh").is_file()
    assert (Path(args_dict["output_path"]) / "simulate_prod.job_2.sh").is_file()


@mock.patch("simtools.job_execution.script_job_generator.subprocess.run")
@mock.patch("simtools.job_execution.htcondor_script_generator.read_job_grid")
def test_generate_script_jobs_runs_selected_script(
    mock_read_job_grid, mock_subprocess_run, args_dict, job_rows, job_grid_metadata
):
    args_dict["run_script"] = True
    mock_read_job_grid.return_value = (job_rows, job_grid_metadata)

    script_job_generator.generate_script_jobs(args_dict)

    mock_subprocess_run.assert_called_once_with(
        [str(Path(args_dict["output_path"]) / "simulate_prod.job_5.sh")], check=True
    )


def test_generate_script_jobs_requires_selected_line_for_run_script(args_dict):
    args_dict["job_grid_line"] = None
    args_dict["run_script"] = True

    with pytest.raises(ValueError, match="--run_script requires --job_grid_line"):
        script_job_generator.generate_script_jobs(args_dict)
