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
def args_dict(tmp_test_directory):
    return {
        "output_path": str(Path(tmp_test_directory) / "output"),
        "apptainer_image": "/path/to/apptainer_image.sif",
        "priority": 5,
        "job_grid_file": str(Path(tmp_test_directory) / "output" / "job_grid.ecsv"),
        "label": "test_label",
        "simulation_software": "simtools",
        "site": "test_site",
        "log_level": "INFO",
        "simulation_output": "simtools-output",
    }


@pytest.fixture
def job_rows():
    return [
        {
            "primary": "gamma",
            "azimuth_angle": 45 * u.deg,
            "zenith_angle": 20 * u.deg,
            "energy_min": 1 * u.GeV,
            "energy_max": 10 * u.GeV,
            "core_scatter_number": 10,
            "core_scatter_max": 100 * u.m,
            "view_cone_min": 0 * u.deg,
            "view_cone_max": 5 * u.deg,
            "showers_per_run": 1000,
            "model_version": "7.0.0",
            "array_layout_name": "CTAO-North-Alpha",
            "corsika_le_interaction": "urqmd",
            "corsika_he_interaction": "epos",
            "run_number": 1,
        }
    ]


@pytest.fixture
def job_grid_metadata():
    return {
        "site": "test_site",
        "simulation_software": "simtools",
    }


@mock.patch("simtools.job_execution.htcondor_script_generator.read_job_grid")
@mock.patch("simtools.job_execution.htcondor_script_generator.Path.mkdir")
@mock.patch("simtools.job_execution.htcondor_script_generator.open", new_callable=mock.mock_open)
@mock.patch("simtools.job_execution.htcondor_script_generator.Path.chmod")
@mock.patch("simtools.job_execution.htcondor_script_generator.Path.is_file", return_value=True)
def test_generate_submission_script(
    mock_is_file,
    mock_chmod,
    mock_open,
    mock_mkdir,
    mock_read_job_grid,
    args_dict,
    job_rows,
    job_grid_metadata,
):
    mock_read_job_grid.return_value = (job_rows, job_grid_metadata)

    generate_submission_script(args_dict)

    work_dir = Path(args_dict["output_path"])
    submit_file_name = "simulate_prod.submit"

    mock_is_file.assert_called_once()
    mock_read_job_grid.assert_called_once_with(args_dict["job_grid_file"])
    mock_mkdir.assert_any_call(parents=True, exist_ok=True)
    mock_open.assert_any_call(work_dir / f"{submit_file_name}.condor", "w", encoding="utf-8")
    mock_open.assert_any_call(work_dir / f"{submit_file_name}.params.txt", "w", encoding="utf-8")
    mock_open.assert_any_call(work_dir / f"{submit_file_name}.sh", "w", encoding="utf-8")
    mock_chmod.assert_called_once_with(0o755)


@mock.patch("simtools.job_execution.htcondor_script_generator.read_job_grid")
@mock.patch("simtools.job_execution.htcondor_script_generator.Path.mkdir")
@mock.patch("simtools.job_execution.htcondor_script_generator.open", new_callable=mock.mock_open)
@mock.patch("simtools.job_execution.htcondor_script_generator.Path.chmod")
@mock.patch("simtools.job_execution.htcondor_script_generator.Path.is_file", return_value=True)
def test_generate_submission_script_writes_label_specific_files(
    mock_is_file,
    mock_chmod,
    mock_open,
    mock_mkdir,
    mock_read_job_grid,
    args_dict,
    tmp_test_directory,
    job_rows,
    job_grid_metadata,
):
    args_dict["output_path"] = str(Path(tmp_test_directory) / "output")
    args_dict["htcondor_log_path"] = str(Path(tmp_test_directory) / "custom_logs")
    args_dict["apptainer_image"] = {
        "prod 7.0.0": "/path/to/prod.sif",
        "beta": "/path/to/beta.sif",
    }
    mock_read_job_grid.return_value = (job_rows, job_grid_metadata)

    generate_submission_script(args_dict)

    work_dir = Path(args_dict["output_path"])

    mock_is_file.assert_has_calls([mock.call(), mock.call()], any_order=False)
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
    expected_script = f"""#!/usr/bin/env bash

# Process ID used to generate run number
process_id="$1"
# Load environment variables (for DB access)
set -a; source "$2"
apptainer_label="${{3}}"
primary="${{4}}"
model_version="${{19}}"
array_layout_name="${{20}}"
corsika_le_interaction="${{21}}"
corsika_he_interaction="${{22}}"
run_number="${{23}}"
pack_for_grid_register="${{24}}"
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
    --showers_per_run "${{18}}" \\
    --energy_range "${{7}} ${{8}} ${{9}} ${{10}}" \\
    --core_scatter "${{11}} ${{12}} ${{13}}" \\
    --view_cone "${{14}} ${{15}} ${{16}} ${{17}}" \\
    --corsika_le_interaction "$corsika_le_interaction" \\
    --corsika_he_interaction "$corsika_he_interaction" \\
    --run_number "$run_number" \\
    --run_number_offset 0 \\
    --save_reduced_event_lists \\
    --output_path /tmp/simtools-output \\
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
    htcondor_dirs = {"log": log_dir, "error": error_dir, "output": output_dir}
    content = _get_submit_file(
        executable="simulate_prod.submit.sh",
        apptainer_image=apptainer_image,
        priority=1,
        params_file_name="simulate_prod.submit.params.txt",
        htcondor_dirs=htcondor_dirs,
    )

    assert "queue apptainer_label,primary" in content
    assert "core_scatter_number,core_scatter_max_value,core_scatter_max_unit" in content
    assert (
        "view_cone_min_value,view_cone_min_unit,view_cone_max_value,view_cone_max_unit" in content
    )
    assert "showers_per_run,model_version,array_layout_name" in content
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


def test_build_job_specs_reads_grid_file(args_dict, job_rows, job_grid_metadata):
    with mock.patch(
        "simtools.job_execution.htcondor_script_generator.read_job_grid",
        return_value=(job_rows, job_grid_metadata),
    ) as mock_read_job_grid:
        job_specs, metadata = build_job_specs(args_dict, ["7.0.0"])

    mock_read_job_grid.assert_called_once_with(args_dict["job_grid_file"])
    assert metadata == job_grid_metadata
    assert job_specs[0]["image_label"] == "7.0.0"
    assert job_specs[0]["pack_for_grid_register"] == "simtools-output/7.0.0"
    assert job_specs[0]["array_layout_name"] == "CTAO-North-Alpha"


def test_build_job_specs_raises_for_missing_required_metadata(args_dict, job_rows):
    with mock.patch(
        "simtools.job_execution.htcondor_script_generator.read_job_grid",
        return_value=(job_rows, {"coordinate_system": "horizontal"}),
    ):
        with pytest.raises(
            ValueError, match=r"missing required field\(s\): site, simulation_software"
        ):
            build_job_specs(args_dict, ["7.0.0"])


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
            "core_scatter_number": 10,
            "core_scatter_max": 200 * u.m,
            "view_cone_min": 0 * u.deg,
            "view_cone_max": 5 * u.deg,
            "showers_per_run": 1000,
            "model_version": "7.0.0",
            "array_layout_name": "CTAO-North-Alpha",
            "corsika_le_interaction": "urqmd",
            "corsika_he_interaction": "epos",
            "run_number": 10,
            "pack_for_grid_register": "simtools-output/7.0.0",
        }
    ]

    _write_params_file(params_file_path, label_job_specs)

    assert params_file_path.read_text(encoding="utf-8") == (
        "7.0.0 gamma 0.0 20.0 30.0 GeV 10.0 TeV 10 200.0 m 0.0 deg 5.0 deg "
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
            "core_scatter_number": 10,
            "core_scatter_max": 200 * u.m,
            "view_cone_min": 0 * u.deg,
            "view_cone_max": 5 * u.deg,
            "showers_per_run": 1000,
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
        "grid_label_7.0.0 gamma 0.0 20.0 30.0 GeV 10.0 TeV 10 200.0 m 0.0 deg 5.0 deg "
        "1000 7.0.0 CTAO-North-Alpha urqmd epos 10 simtools-output/grid_label_7.0.0\n"
    )
