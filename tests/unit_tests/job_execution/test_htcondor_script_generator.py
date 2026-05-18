from pathlib import Path
from unittest import mock

import astropy.units as u
import pytest

from simtools.job_execution.htcondor_script_generator import (
    _format_param_value,
    _format_quantity,
    _get_submit_file,
    _get_submit_script,
    _resolve_apptainer_images,
    _sanitize_label_for_filename,
    _write_params_file,
    build_job_specs,
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


@mock.patch("simtools.job_execution.htcondor_script_generator.Path.mkdir")
@mock.patch("simtools.job_execution.htcondor_script_generator.open", new_callable=mock.mock_open)
@mock.patch("simtools.job_execution.htcondor_script_generator.Path.chmod")
@mock.patch("simtools.job_execution.htcondor_script_generator.Path.is_file", return_value=True)
def test_generate_submission_script_writes_label_specific_files(
    mock_is_file, mock_chmod, mock_open, mock_mkdir, args_dict
):
    args_dict["output_path"] = "/test_output"
    args_dict["htcondor_log_path"] = "/custom_logs"
    args_dict["number_of_runs"] = 1
    args_dict["apptainer_image"] = {
        "prod 7.0.0": "/path/to/prod.sif",
        "beta": "/path/to/beta.sif",
    }

    generate_submission_script(args_dict)

    work_dir = Path(args_dict["output_path"])

    mock_is_file.assert_has_calls(
        [mock.call(), mock.call()],
        any_order=False,
    )
    mock_open.assert_any_call(
        work_dir / "simulate_prod.submit.prod_7.0.0.condor", "w", encoding="utf-8"
    )
    mock_open.assert_any_call(
        work_dir / "simulate_prod.submit.prod_7.0.0.params.txt", "w", encoding="utf-8"
    )
    mock_open.assert_any_call(work_dir / "simulate_prod.submit.beta.condor", "w", encoding="utf-8")
    mock_open.assert_any_call(
        work_dir / "simulate_prod.submit.beta.params.txt", "w", encoding="utf-8"
    )
    mock_open.assert_any_call(work_dir / "simulate_prod.submit.sh", "w", encoding="utf-8")
    mock_mkdir.assert_any_call(parents=True, exist_ok=True)
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
    n_core_scatter = args_dict["core_scatter"][0]
    view_cone_low = args_dict["view_cone"][0].to(u.deg).value
    view_cone_high = args_dict["view_cone"][1].to(u.deg).value
    container_output_path = str(Path("/", "tmp", "simtools-output"))

    expected_script = f"""#!/usr/bin/env bash

# Process ID used to generate run number
process_id="$1"
# Load environment variables (for DB access)
set -a; source "$2"
apptainer_label="${{3}}"
primary="${{4}}"
model_version="${{14}}"
array_layout_name="${{15}}"
corsika_le_interaction="${{16}}"
corsika_he_interaction="${{17}}"
run_number="${{18}}"
pack_for_grid_register="${{19}}"
energy_range_tag="erange-${{7}}${{8}}-${{9}}${{10}}"
job_label="{args_dict["label"]}_${{corsika_he_interaction}}-${{corsika_le_interaction}}_${{energy_range_tag}}"

simtools-simulate-prod \\
    --simulation_software {args_dict["simulation_software"]} \\
    --label "$job_label" \\
    --model_version "$model_version" \\
    --site {args_dict["site"]} \\
    --array_layout_name "$array_layout_name" \\
    --primary "$primary" \\
    --azimuth_angle "${{5}}" \\
    --zenith_angle "${{6}}" \\
    --nshow "${{13}}" \\
    --energy_range "${{7}} ${{8}} ${{9}} ${{10}}" \\
    --core_scatter "{n_core_scatter} ${{11}} ${{12}}" \\
    --view_cone "{view_cone_low} deg {view_cone_high} deg" \\
    --corsika_le_interaction "$corsika_le_interaction" \\
    --corsika_he_interaction "$corsika_he_interaction" \\
    --run_number "$run_number" \\
    --run_number_offset {args_dict["run_number_offset"]} \\
    --save_reduced_event_lists \\
    --output_path {container_output_path} \\
    --log_level {args_dict["log_level"]} \\
    --pack_for_grid_register "$pack_for_grid_register"
"""
    generated_script = _get_submit_script(args_dict)
    assert generated_script == expected_script


def test_get_submit_file_uses_queue_from_params(tmp_test_directory):
    apptainer_image = Path(tmp_test_directory) / "image.sif"
    log_dir = Path(tmp_test_directory) / "htcondor_logs" / "log"
    error_dir = Path(tmp_test_directory) / "htcondor_logs" / "error"
    output_dir = Path(tmp_test_directory) / "htcondor_logs" / "output"
    content = _get_submit_file(
        executable="simulate_prod.submit.sh",
        apptainer_image=apptainer_image,
        priority=1,
        params_file_name="simulate_prod.submit.params.txt",
        log_dir=log_dir,
        error_dir=error_dir,
        output_dir=output_dir,
    )

    assert "queue apptainer_label,primary" in content
    assert "energy_min_value,energy_min_unit,energy_max_value,energy_max_unit" in content
    assert "core_scatter_max_value,core_scatter_max_unit" in content
    assert "nshow,model_version,array_layout_name" in content
    assert "model_version,array_layout_name,corsika_le_interaction" in content
    assert "from simulate_prod.submit.params.txt" in content
    assert 'arguments = "$(process) env.txt' in content
    assert str(log_dir) in content
    assert str(error_dir) in content
    assert str(output_dir) in content


@mock.patch("simtools.job_execution.htcondor_script_generator.Path.is_file", return_value=True)
def test_resolve_apptainer_images_dict(mock_is_file, tmp_test_directory):
    images = _resolve_apptainer_images(
        {
            "7.0.0": str(Path(tmp_test_directory) / "v7.sif"),
            "6.3.0": str(Path(tmp_test_directory) / "v63.sif"),
        }
    )

    assert set(images.keys()) == {"7.0.0", "6.3.0"}
    assert mock_is_file.call_count == 2


def test_resolve_apptainer_images_string(tmp_test_directory):
    image_path = Path(tmp_test_directory) / "image.sif"
    image_path.touch()

    images = _resolve_apptainer_images(str(image_path))

    assert images == {"default": image_path}


def test_resolve_apptainer_images_none():
    with pytest.raises(ValueError, match="Missing required apptainer_image path"):
        _resolve_apptainer_images(None)


def test_resolve_apptainer_images_blank_string():
    with pytest.raises(FileNotFoundError, match="Apptainer image file not found"):
        _resolve_apptainer_images("   ")


def test_resolve_apptainer_images_empty_dict():
    with pytest.raises(ValueError, match="Missing required apptainer_image path"):
        _resolve_apptainer_images({})


def test_resolve_apptainer_images_raises_for_truthy_empty_mapping():
    class TruthyEmptyDict(dict):
        def __bool__(self):
            return True

    with pytest.raises(
        ValueError, match="At least one apptainer image label/path must be configured"
    ):
        _resolve_apptainer_images(TruthyEmptyDict())


def test_resolve_apptainer_images_invalid_type(tmp_test_directory):
    with pytest.raises(
        TypeError, match="apptainer_image must be a string path or a label-to-path dictionary"
    ):
        _resolve_apptainer_images([str(Path(tmp_test_directory) / "tmp/image.sif")])


def test_resolve_apptainer_images_raises_for_missing_file(tmp_test_directory):
    missing_path = Path(tmp_test_directory) / "missing.sif"

    with pytest.raises(FileNotFoundError, match="Apptainer image file not found"):
        _resolve_apptainer_images(str(missing_path))


def test_format_quantity_full_coverage():
    value, unit = _format_quantity(5 * u.TeV)
    assert float(value) == pytest.approx(5.0)
    assert unit == "TeV"

    value, unit = _format_quantity(100 * u.cm, convert_to=u.m)
    assert float(value) == pytest.approx(1.0)
    assert unit == "m"

    value, unit = _format_quantity(42, default_unit=u.GeV)
    assert value == "42"
    assert unit == "GeV"

    value, unit = _format_quantity("abc")
    assert value == "abc"
    assert unit is None


def test_format_param_value_raises_for_missing_required_value():
    with pytest.raises(ValueError, match="Missing required value for field 'primary'"):
        _format_param_value(None, "primary")


def test_sanitize_label_for_filename():
    assert _sanitize_label_for_filename("  my label:7/0*0?  ") == "my_label_7_0_0_"
    assert _sanitize_label_for_filename("v7.0.0-beta_1") == "v7.0.0-beta_1"
    assert _sanitize_label_for_filename(42) == "42"


def test_write_params_file_resolves_array_layout_name_by_model_version(
    args_dict, tmp_test_directory
):
    args_dict["number_of_runs"] = 1
    args_dict["model_version"] = ["6.3.0", "7.0.0"]
    args_dict["array_layout_name"] = {
        "by_version": {
            "<7.0.0": "alpha",
            ">=7.0.0": "CTAO-North-Alpha",
        }
    }

    params_file_path = Path(tmp_test_directory) / "params.txt"
    job_specs = build_job_specs(args_dict, ["7.0.0"])

    _write_params_file(params_file_path, job_specs)

    params_lines = params_file_path.read_text(encoding="utf-8").splitlines()

    assert "6.3.0 alpha urqmd epos 1 simtools-output/7.0.0" in params_lines[0]
    assert "7.0.0 CTAO-North-Alpha urqmd epos 2 simtools-output/7.0.0" in params_lines[1]


def test_write_params_file_resolves_stringified_by_version_layout(args_dict, tmp_test_directory):
    args_dict["number_of_runs"] = 1
    args_dict["model_version"] = ["6.3.0", "7.0.0"]
    args_dict["array_layout_name"] = str(
        {
            "by_version": {
                "<7.0.0": "alpha",
                ">=7.0.0": "CTAO-North-Alpha",
            }
        }
    )

    params_file_path = Path(tmp_test_directory) / "params.txt"
    job_specs = build_job_specs(args_dict, ["7.0.0"])

    _write_params_file(params_file_path, job_specs)

    params_lines = params_file_path.read_text(encoding="utf-8").splitlines()

    assert "6.3.0 alpha urqmd epos 1 simtools-output/7.0.0" in params_lines[0]
    assert "7.0.0 CTAO-North-Alpha urqmd epos 2 simtools-output/7.0.0" in params_lines[1]


def test_write_params_file_keeps_energy_units(tmp_test_directory):
    params_file_path = Path(tmp_test_directory) / "params.txt"
    label_job_specs = [
        {
            "image_label": "7.0.0",
            "primary": "gamma",
            "azimuth_angle": 0 * u.deg,
            "zenith_angle": 20 * u.deg,
            "energy_min": 30 * u.GeV,
            "energy_max": 10 * u.TeV,
            "core_scatter_max": 200 * u.m,
            "nshow": 1000,
            "model_version": "7.0.0",
            "array_layout_name": {
                "by_version": {
                    "<7.0.0": "alpha",
                    ">=7.0.0": "CTAO-North-Alpha",
                }
            },
            "corsika_le_interaction": "urqmd",
            "corsika_he_interaction": "epos",
            "run_number": 10,
            "pack_for_grid_register": "simtools-output/7.0.0",
        }
    ]

    _write_params_file(params_file_path, label_job_specs)

    assert params_file_path.read_text(encoding="utf-8") == (
        "7.0.0 gamma 0.0 20.0 30.0 GeV 10.0 TeV 200.0 m "
        "1000 7.0.0 CTAO-North-Alpha urqmd epos 10 simtools-output/7.0.0\n"
    )


def test_write_params_file_replaces_whitespace_in_apptainer_label(tmp_test_directory):
    params_file_path = Path(tmp_test_directory) / "params.txt"
    label_job_specs = [
        {
            "image_label": "grid label 7.0.0",
            "primary": "gamma",
            "azimuth_angle": 0 * u.deg,
            "zenith_angle": 20 * u.deg,
            "energy_min": 30 * u.GeV,
            "energy_max": 10 * u.TeV,
            "core_scatter_max": 200 * u.m,
            "nshow": 1000,
            "model_version": "7.0.0",
            "array_layout_name": "CTAO-North-Alpha",
            "corsika_le_interaction": "urqmd",
            "corsika_he_interaction": "epos",
            "run_number": 10,
            "pack_for_grid_register": "simtools-output/grid label 7.0.0",
        }
    ]

    _write_params_file(params_file_path, label_job_specs)

    assert params_file_path.read_text(encoding="utf-8") == (
        "grid_label_7.0.0 gamma 0.0 20.0 30.0 GeV 10.0 TeV 200.0 m "
        "1000 7.0.0 CTAO-North-Alpha urqmd epos 10 simtools-output/grid_label_7.0.0\n"
    )
