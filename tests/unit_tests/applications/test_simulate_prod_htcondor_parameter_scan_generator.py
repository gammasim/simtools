#!/usr/bin/python3

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import simtools.applications.simulate_prod_htcondor_parameter_scan_generator as app


@patch(
    "simtools.applications.simulate_prod_htcondor_parameter_scan_generator._generate_parameter_submission_scripts"
)
@patch(
    "simtools.applications.simulate_prod_htcondor_parameter_scan_generator._generate_overwrite_files"
)
@patch("simtools.applications.simulate_prod_htcondor_parameter_scan_generator.build_application")
def test_main_uses_standard_build_application(
    mock_build_application,
    mock_generate_overwrite_files,
    mock_generate_parameter_submission_scripts,
):
    mock_build_application.return_value = SimpleNamespace(
        args={
            "output_path": "htcondor_parameter_scan",
            "parameter_values": [10, 20, 30],
            "parameter_path": "changes.LSTN-02.asum_threshold",
            "parameter_name": "threshold",
            "overwrite_template": "overwrite_template.yaml",
        }
    )
    mock_generate_overwrite_files.return_value = {
        10: Path("overwrite_threshold_10.yaml"),
        20: Path("overwrite_threshold_20.yaml"),
        30: Path("overwrite_threshold_30.yaml"),
    }

    app.main()

    assert mock_build_application.call_args.kwargs["initialization_kwargs"] == {
        "db_config": False,
        "simulation_model": ["site", "layout", "model_version"],
        "simulation_configuration": {"software": None, "corsika_configuration": ["all"]},
    }
    mock_generate_overwrite_files.assert_called_once()
    mock_generate_parameter_submission_scripts.assert_called_once()


def test_set_nested_value():
    """Test setting nested values in dictionaries."""
    data = {}
    app._set_nested_value(data, ["changes", "LSTN-02", "asum_threshold"], 10)
    assert data["changes"]["LSTN-02"]["asum_threshold"]["value"] == 10
    assert "version" in data["changes"]["LSTN-02"]["asum_threshold"]


def test_format_value_for_filename():
    """Test value formatting for filenames."""
    assert app._format_value_for_filename(10.0) == "10"
    assert app._format_value_for_filename(10.5) == "10.5"
    assert app._format_value_for_filename(10.123) == "10.123"
    assert app._format_value_for_filename(10.1000) == "10.1"


def test_generate_overwrite_files(tmp_path):
    """Test generation of overwrite YAML files."""
    template_file = tmp_path / "template.yaml"
    template_file.write_text("""model_version: "7.0.0"
model_update: "patch_update"
description: "Template"
changes:
  LSTN-02:
    min_photons:
      version: "2.0.0"
      value: 0
""")

    args_dict = {
        "parameter_values": [10, 20],
        "overwrite_template": str(template_file),
        "parameter_path": "changes.LSTN-02.asum_threshold",
        "parameter_name": "threshold",
    }

    parameter_files = app._generate_overwrite_files(args_dict, tmp_path)

    assert len(parameter_files) == 2
    assert (tmp_path / "overwrite_threshold_10.yaml").exists()
    assert (tmp_path / "overwrite_threshold_20.yaml").exists()


def test_generate_overwrite_files_with_default_name(tmp_path):
    """Test that parameter name defaults to last part of path."""
    template_file = tmp_path / "template.yaml"
    template_file.write_text("""model_version: "7.0.0"
description: "Template"
""")

    args_dict = {
        "parameter_values": [1.0, 2.0],
        "overwrite_template": str(template_file),
        "parameter_path": "changes.OBS-North.nsb_scaling_factor",
        "parameter_name": None,
    }

    parameter_files = app._generate_overwrite_files(args_dict, tmp_path)

    assert len(parameter_files) == 2
    assert (tmp_path / "overwrite_nsb_scaling_factor_1.yaml").exists()
    assert (tmp_path / "overwrite_nsb_scaling_factor_2.yaml").exists()


def test_generate_overwrite_files_missing_template(tmp_path):
    """Test that missing template file raises FileNotFoundError."""
    args_dict = {
        "parameter_values": [10],
        "overwrite_template": "nonexistent.yaml",
        "parameter_path": "changes.LSTN-02.asum_threshold",
        "parameter_name": "threshold",
    }

    with pytest.raises(FileNotFoundError):
        app._generate_overwrite_files(args_dict, tmp_path)


@patch(
    "simtools.applications.simulate_prod_htcondor_parameter_scan_generator.htcondor_script_generator._get_submit_file"
)
@patch(
    "simtools.applications.simulate_prod_htcondor_parameter_scan_generator.htcondor_script_generator._get_submit_script"
)
@patch(
    "simtools.applications.simulate_prod_htcondor_parameter_scan_generator.htcondor_script_generator._resolve_apptainer_images"
)
def test_generate_parameter_submission_scripts(
    mock_resolve_images, mock_get_submit_script, mock_get_submit_file, tmp_path
):
    """Test generation of submission scripts."""
    mock_resolve_images.return_value = {"default": Path("/path/to/image.sif")}
    mock_get_submit_script.return_value = "#!/usr/bin/env bash\nsimtools-simulate-prod"
    mock_get_submit_file.return_value = "universe = container\nqueue from params.txt\n"

    overwrite_file = tmp_path / "overwrite_threshold_10.yaml"
    overwrite_file.write_text("test: value")

    args_dict = {
        "output_path": str(tmp_path),
        "apptainer_image": "/path/to/image.sif",
        "label": "test_label",
        "number_of_runs": 5,
        "priority": 5,
        "htcondor_log_path": None,
        "parameter_path": "changes.LSTN-02.asum_threshold",
        "parameter_name": "threshold",
    }

    parameter_files = {10: overwrite_file}

    app._generate_parameter_submission_scripts(args_dict, parameter_files)

    assert (tmp_path / "simulate_prod_test_label_10.submit.sh").exists()
    assert (tmp_path / "simulate_prod_test_label_10.submit.condor").exists()
    assert (tmp_path / "htcondor_logs" / "log").exists()
    assert (tmp_path / "htcondor_logs" / "error").exists()
    assert (tmp_path / "htcondor_logs" / "output").exists()


def test_setup_directories(tmp_path):
    """Test directory setup."""
    args_dict = {
        "output_path": str(tmp_path / "output"),
        "htcondor_log_path": None,
    }

    work_dir, log_dir, error_dir, output_dir = app._setup_directories(args_dict)

    assert work_dir.exists()
    assert log_dir.exists()
    assert error_dir.exists()
    assert output_dir.exists()
    assert log_dir == work_dir / "htcondor_logs" / "log"
    assert error_dir == work_dir / "htcondor_logs" / "error"
    assert output_dir == work_dir / "htcondor_logs" / "output"
