from pathlib import Path
from unittest.mock import call, patch

import pytest
import yaml

from simtools.job_execution import bias_curve_submissions


def _base_args(tmp_test_directory):
    return {
        "site": "North",
        "model_version": "7.0.0",
        "array_layout_name": "LSTN-01",
        "simulation_software": "corsika_sim_telarray",
        "azimuth_angle": 0.0,
        "zenith_angle": 20.0,
        "showers_per_run": 10000,
        "core_scatter": "20 1900 m",
        "view_cone": "0 deg 5 deg",
        "number_of_runs": 10,
        "corsika_le_interaction": "urqmd",
        "corsika_he_interaction": "epos",
        "apptainer_image": str(Path(tmp_test_directory) / "simtools.sif"),
        "output_path": str(Path(tmp_test_directory) / "bias_curves"),
        "htcondor_output_path": "htcondor_submit",
        "label": "test_label",
        "priority": 1,
        "telescopes": ["LSTN-01"],
    }


@patch("simtools.job_execution.bias_curve_submissions._generate_curve_submissions")
@patch(
    "simtools.job_execution.bias_curve_submissions._resolve_telescopes_from_layout",
    return_value=["LSTN-01"],
)
def test_generate_bias_curve_submissions_uses_fixed_curve_definitions(
    mock_resolve_telescopes,
    mock_generate_curve_submissions,
    tmp_test_directory,
):
    args = _base_args(tmp_test_directory)
    args.pop("telescopes")

    bias_curve_submissions.generate_bias_curve_submissions(args)

    output_root = Path(args["output_path"]).expanduser().resolve()
    assert args["telescopes"] == ["LSTN-01"]
    mock_resolve_telescopes.assert_called_once_with(args)
    assert mock_generate_curve_submissions.call_args_list == [
        call(
            curve_name="nsb",
            curve_definition=bias_curve_submissions._CURVE_DEFINITIONS["nsb"],
            args=args,
            output_root=output_root,
        ),
        call(
            curve_name="proton",
            curve_definition=bias_curve_submissions._CURVE_DEFINITIONS["proton"],
            args=args,
            output_root=output_root,
        ),
    ]


def test_resolve_telescopes_from_layout_enforces_single_telescope(tmp_test_directory):
    args = _base_args(tmp_test_directory)

    with patch("simtools.job_execution.bias_curve_submissions.SiteModel") as mock_site_model:
        mock_site_model.return_value.get_array_elements_for_layout.return_value = [
            "LSTN-01",
            "LSTN-02",
        ]

        with pytest.raises(ValueError, match="single-telescope layouts"):
            bias_curve_submissions._resolve_telescopes_from_layout(args)


@patch("simtools.job_execution.bias_curve_submissions.simtools_runner.run_applications")
def test_generate_curve_submissions_writes_workflow_with_curve_specific_energy_range(
    mock_run_applications,
    tmp_test_directory,
):
    args = _base_args(tmp_test_directory)
    output_root = Path(args["output_path"]).expanduser().resolve()
    curve_directory = output_root / "nsb"

    bias_curve_submissions._generate_curve_submissions(
        "nsb",
        bias_curve_submissions._CURVE_DEFINITIONS["nsb"],
        args,
        output_root,
    )

    workflow_file = curve_directory / "nsb_workflow.yml"
    assert workflow_file.exists()

    workflow = yaml.safe_load(workflow_file.read_text(encoding="utf-8"))
    applications = workflow["applications"]

    generate_grid_step = applications[0]
    generate_grid_config = generate_grid_step["configuration"]

    assert generate_grid_step["application"] == "simtools-production-generate-grid"
    assert generate_grid_config["primary"] == "gamma"
    assert generate_grid_config["energy_range"] == "20 MeV 25 MeV"
    assert generate_grid_config["label"] == "test_label_nsb"
    assert generate_grid_config["output_file"] == str(curve_directory / "base_grid.ecsv")

    assert applications[1]["application"] == "simtools-generate-parameter-scan-grid"
    assert applications[2]["application"] == "simtools-simulate-prod-htcondor-generator"

    mock_run_applications.assert_called_once()
    runner_args = mock_run_applications.call_args.args[0]
    assert runner_args["config_file"] == str(workflow_file)
    assert runner_args["steps"] is None
    assert runner_args["ignore_runtime_environment"] is True


@patch("simtools.job_execution.bias_curve_submissions.simtools_runner.run_applications")
@pytest.mark.parametrize(
    ("curve_name", "expected_primary", "expected_energy_range"),
    [
        ("nsb", "gamma", "20 MeV 25 MeV"),
        ("proton", "proton", "800 GeV 2000 GeV"),
    ],
)
def test_workflow_uses_curve_specific_primary_and_energy_range(
    mock_run_applications,
    curve_name,
    expected_primary,
    expected_energy_range,
    tmp_test_directory,
):
    args = _base_args(tmp_test_directory)
    output_root = Path(args["output_path"]).expanduser().resolve()
    curve_directory = output_root / curve_name

    bias_curve_submissions._generate_curve_submissions(
        curve_name,
        bias_curve_submissions._CURVE_DEFINITIONS[curve_name],
        args,
        output_root,
    )

    workflow_file = curve_directory / f"{curve_name}_workflow.yml"
    workflow = yaml.safe_load(workflow_file.read_text(encoding="utf-8"))

    generate_grid_config = workflow["applications"][0]["configuration"]

    assert generate_grid_config["primary"] == expected_primary
    assert generate_grid_config["energy_range"] == expected_energy_range
    assert generate_grid_config["label"] == f"test_label_{curve_name}"
    assert generate_grid_config["output_file"] == str(curve_directory / "base_grid.ecsv")

    mock_run_applications.assert_called_once()


@patch("simtools.job_execution.bias_curve_submissions.simtools_runner.run_applications")
def test_nsb_scan_config_contains_nsb_fields_and_trigger_scan(
    mock_run_applications,
    tmp_test_directory,
):
    args = _base_args(tmp_test_directory)
    output_root = Path(args["output_path"]).expanduser().resolve()
    curve_directory = output_root / "nsb"

    bias_curve_submissions._generate_curve_submissions(
        "nsb",
        bias_curve_submissions._CURVE_DEFINITIONS["nsb"],
        args,
        output_root,
    )

    scan_config_file = curve_directory / "scan_config.yml"
    assert scan_config_file.exists()

    scan_config_text = scan_config_file.read_text(encoding="utf-8")

    assert "LSTN-01" in scan_config_text
    assert "asum_threshold" in scan_config_text
    assert "220" in scan_config_text
    assert "300" in scan_config_text
    assert "min_photons" in scan_config_text
    assert "min_photoelectrons" in scan_config_text
    assert "nsb_scaling_factor" in scan_config_text
    assert "value: 2" in scan_config_text

    mock_run_applications.assert_called_once()


@patch("simtools.job_execution.bias_curve_submissions.simtools_runner.run_applications")
def test_proton_scan_config_contains_only_trigger_scan(
    mock_run_applications,
    tmp_test_directory,
):
    args = _base_args(tmp_test_directory)
    output_root = Path(args["output_path"]).expanduser().resolve()
    curve_directory = output_root / "proton"

    bias_curve_submissions._generate_curve_submissions(
        "proton",
        bias_curve_submissions._CURVE_DEFINITIONS["proton"],
        args,
        output_root,
    )

    scan_config_file = curve_directory / "scan_config.yml"
    assert scan_config_file.exists()

    scan_config_text = scan_config_file.read_text(encoding="utf-8")

    assert "LSTN-01" in scan_config_text
    assert "asum_threshold" in scan_config_text
    assert "220" in scan_config_text
    assert "300" in scan_config_text
    assert "min_photons" not in scan_config_text
    assert "min_photoelectrons" not in scan_config_text
    assert "nsb_scaling_factor" not in scan_config_text

    mock_run_applications.assert_called_once()


@pytest.mark.parametrize(
    ("telescope", "expected_param", "expected_values"),
    [
        ("LSTN-01", "asum_threshold", [220, 230, 240, 250, 260, 270, 280, 290, 300]),
        ("MSTN-01", "dsum_threshold", [22, 23, 24, 25, 26, 27, 28, 29, 30]),
    ],
)
def test_threshold_scan_is_chosen_from_telescope_type(
    telescope,
    expected_param,
    expected_values,
):
    assert bias_curve_submissions._threshold_param_name(telescope) == expected_param
    assert bias_curve_submissions._threshold_values_for_telescope(telescope) == expected_values


def test_threshold_param_name_rejects_empty_telescope_name():
    with pytest.raises(ValueError, match="empty telescope name"):
        bias_curve_submissions._threshold_param_name("")
