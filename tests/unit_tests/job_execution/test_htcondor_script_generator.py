from pathlib import Path
from unittest import mock

import astropy.units as u
import pytest

from simtools.job_execution.htcondor_script_generator import (
    _build_job_specs,
    _get_submit_file,
    _get_submit_script,
    _resolve_apptainer_images,
    _write_params_file,
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
        "view_cone": [0 * u.deg, 10 * u.deg],
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
        "pack_for_grid_register": "simtools-output",
        "corsika_le_interaction": "urqmd",
        "corsika_he_interaction": "epos",
    }


@mock.patch("simtools.job_execution.htcondor_script_generator.Path.mkdir")
@mock.patch("simtools.job_execution.htcondor_script_generator.open", new_callable=mock.mock_open)
@mock.patch("simtools.job_execution.htcondor_script_generator.Path.chmod")
@mock.patch("simtools.job_execution.htcondor_script_generator.Path.is_file", return_value=True)
def test_generate_submission_script(mock_is_file, mock_chmod, mock_open, mock_mkdir, args_dict):
    generate_submission_script(args_dict)

    work_dir = Path(args_dict["output_path"])
    submit_file_name = "simulate_prod.submit"

    mock_is_file.assert_called_once()
    mock_mkdir.assert_any_call(parents=True, exist_ok=True)

    # Check if files are created and written
    mock_open.assert_any_call(work_dir / f"{submit_file_name}.condor", "w", encoding="utf-8")
    mock_open.assert_any_call(work_dir / f"{submit_file_name}.params.txt", "w", encoding="utf-8")
    mock_open.assert_any_call(work_dir / f"{submit_file_name}.sh", "w", encoding="utf-8")

    # Check if chmod is called
    mock_chmod.assert_called_once_with(0o755)


@mock.patch("simtools.job_execution.htcondor_script_generator.Path.is_file", return_value=False)
@mock.patch("simtools.job_execution.htcondor_script_generator.open", new_callable=mock.mock_open)
def test_generate_submission_script_raises_for_missing_apptainer_image(
    mock_open, mock_is_file, args_dict
):
    with pytest.raises(FileNotFoundError, match="Apptainer image file not found"):
        generate_submission_script(args_dict)

    mock_is_file.assert_called_once()
    mock_open.assert_not_called()


def test_get_submit_script(args_dict):
    core_scatter_low = args_dict["core_scatter"][0]
    core_scatter_high = args_dict["core_scatter"][1].to(u.m).value
    view_cone_low = args_dict["view_cone"][0].to(u.deg).value
    view_cone_high = args_dict["view_cone"][1].to(u.deg).value

    expected_script = f"""#!/usr/bin/env bash

# Process ID used to generate run number
process_id="$1"
# Load environment variables (for DB access)
set -a; source "$2"
apptainer_label="$3"
primary="$4"
model_version="${{11}}"
corsika_le_interaction="${{12}}"
corsika_he_interaction="${{13}}"
run_number="${{14}}"
pack_for_grid_register="${{15}}"
energy_range_tag="erange-$7$8-$9${10}"
job_label="{args_dict["label"]}_${{corsika_he_interaction}}-${{corsika_le_interaction}}_${{energy_range_tag}}"

simtools-simulate-prod \\
    --simulation_software {args_dict["simulation_software"]} \\
    --label "$job_label" \\
    --model_version "$model_version" \\
    --site {args_dict["site"]} \\
    --array_layout_name {args_dict["array_layout_name"]} \\
    --primary "$primary" \\
    --azimuth_angle "$5" \\
    --zenith_angle "$6" \\
    --nshow {args_dict["nshow"]} \\
    --energy_range "$7 $8 $9 ${10}" \
    --core_scatter "{core_scatter_low} {core_scatter_high} m" \\
    --view_cone "{view_cone_low} deg {view_cone_high} deg" \\
    --corsika_le_interaction "$corsika_le_interaction" \\
    --corsika_he_interaction "$corsika_he_interaction" \\
    --run_number "$run_number" \\
    --run_number_offset {args_dict["run_number_offset"]} \\
    --output_path /tmp/simtools-output \\
    --log_level {args_dict["log_level"]} \\
    --pack_for_grid_register "$pack_for_grid_register"
"""
    generated_script = _get_submit_script(args_dict)
    assert generated_script == expected_script


def test_get_submit_file_uses_queue_from_params():
    content = _get_submit_file(
        executable="simulate_prod.submit.sh",
        apptainer_image=Path("/tmp/image.sif"),
        priority=1,
        params_file_name="simulate_prod.submit.params.txt",
    )

    assert "queue apptainer_label,primary" in content
    assert "energy_min_value,energy_min_unit,energy_max_value,energy_max_unit" in content
    assert "from simulate_prod.submit.params.txt" in content
    assert 'arguments = "$(process) env.txt' in content


@mock.patch("simtools.job_execution.htcondor_script_generator.Path.is_file", return_value=True)
def test_resolve_apptainer_images_dict(mock_is_file):
    images = _resolve_apptainer_images({"7.0.0": "/tmp/v7.sif", "6.3.0": "/tmp/v63.sif"})

    assert set(images.keys()) == {"7.0.0", "6.3.0"}
    assert mock_is_file.call_count == 2


def test_build_job_specs_expands_model_version_list(args_dict):
    args_dict["model_version"] = ["6.3.0", "7.0.0"]

    job_specs = _build_job_specs(args_dict, ["7.0.0"])
    model_versions = {job_spec["model_version"] for job_spec in job_specs}

    assert model_versions == {"6.3.0", "7.0.0"}
    assert len(job_specs) == 2 * args_dict["number_of_runs"]


def test_build_job_specs_expands_energy_range_list_of_pairs(args_dict):
    args_dict["number_of_runs"] = 1
    args_dict["energy_range"] = [
        (30 * u.GeV, 30 * u.GeV),
        (300 * u.GeV, 300 * u.GeV),
    ]

    job_specs = _build_job_specs(args_dict, ["7.0.0"])
    energy_pairs = {(job_spec["energy_min"], job_spec["energy_max"]) for job_spec in job_specs}

    assert len(job_specs) == 2
    assert energy_pairs == {
        (30 * u.GeV, 30 * u.GeV),
        (300 * u.GeV, 300 * u.GeV),
    }


def test_build_job_specs_increments_run_number(args_dict):
    args_dict["number_of_runs"] = 2
    args_dict["run_number"] = 10
    args_dict["model_version"] = ["6.3.0", "7.0.0"]

    job_specs = _build_job_specs(args_dict, ["7.0.0"])
    run_numbers = [job_spec["run_number"] for job_spec in job_specs]

    assert run_numbers == [10, 11, 12, 13]


def test_write_params_file_keeps_energy_units(tmp_test_directory):
    params_file_path = Path(tmp_test_directory) / "params.txt"
    label_job_specs = [
        {
            "apptainer_label": "7.0.0",
            "primary": "gamma",
            "azimuth_angle": 0 * u.deg,
            "zenith_angle": 20 * u.deg,
            "energy_min": 30 * u.GeV,
            "energy_max": 10 * u.TeV,
            "model_version": "7.0.0",
            "corsika_le_interaction": "urqmd",
            "corsika_he_interaction": "epos",
            "run_number": 10,
            "pack_for_grid_register": "simtools-output/7.0.0",
        }
    ]

    _write_params_file(params_file_path, label_job_specs)

    assert params_file_path.read_text(encoding="utf-8") == (
        "7.0.0 gamma 0.0 20.0 30.0 GeV 10.0 TeV 7.0.0 urqmd epos 10 simtools-output/7.0.0\n"
    )
