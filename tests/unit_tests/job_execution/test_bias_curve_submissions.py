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


def _command_value(command, option):
    """Return the value following an option in a command list."""
    return command[command.index(option) + 1]


def _run_generation_without_external_side_effects(curve_name, args, output_root):
    """Run one curve-generation step while mocking external execution."""
    with (
        patch("simtools.job_execution.bias_curve_submissions._run_command") as mock_run_command,
        patch("simtools.job_execution.bias_curve_submissions._build_scan_grid") as mock_build_scan,
        patch(
            "simtools.job_execution.bias_curve_submissions."
            "htcondor_script_generator.generate_submission_script"
        ) as mock_generate_submission_script,
    ):
        bias_curve_submissions._generate_curve_submissions(
            curve_name,
            bias_curve_submissions._CURVE_DEFINITIONS[curve_name],
            args,
            output_root,
        )

    return mock_run_command, mock_build_scan, mock_generate_submission_script


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
    assert args["telescope"] == "LSTN-01"
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


def test_generate_curve_submissions_runs_grid_generation_and_htcondor_generation(
    tmp_test_directory,
):
    args = _base_args(tmp_test_directory)
    args["telescope"] = "LSTN-01"
    output_root = Path(args["output_path"]).expanduser().resolve()
    curve_directory = output_root / "nsb"

    mock_run_command, mock_build_scan, mock_generate_submission_script = (
        _run_generation_without_external_side_effects("nsb", args, output_root)
    )

    base_grid_file = curve_directory / "base_grid.ecsv"
    scan_grid_file = curve_directory / "scan_grid.ecsv"

    mock_run_command.assert_called_once()
    command, working_directory = mock_run_command.call_args.args

    assert working_directory == output_root
    assert command[0] == "simtools-production-generate-grid"
    assert _command_value(command, "--primary") == "gamma"
    assert _command_value(command, "--energy_range") == "20 MeV 25 MeV"
    assert _command_value(command, "--label") == "nsb"
    assert _command_value(command, "--output_file") == str(base_grid_file)

    mock_build_scan.assert_called_once()
    build_scan_kwargs = mock_build_scan.call_args.kwargs
    assert build_scan_kwargs["base_grid_file"] == base_grid_file
    assert build_scan_kwargs["scan_grid_file"] == scan_grid_file
    assert build_scan_kwargs["telescope"] == "LSTN-01"
    assert len(build_scan_kwargs["overwrite_files_and_labels"]) == len(
        bias_curve_submissions._ASUM_THRESHOLDS
    )

    mock_generate_submission_script.assert_called_once()
    htcondor_args = mock_generate_submission_script.call_args.args[0]
    assert htcondor_args["label"] == "nsb"
    assert htcondor_args["job_grid_file"] == str(scan_grid_file)
    assert htcondor_args["output_path"] == str(curve_directory / "htcondor_submit")
    assert htcondor_args["simulation_output"] == str(output_root)
    assert htcondor_args["telescope"] == "LSTN-01"


@pytest.mark.parametrize(
    ("curve_name", "expected_primary", "expected_energy_range"),
    [
        ("nsb", "gamma", "20 MeV 25 MeV"),
        ("proton", "proton", "2 GeV 2000 GeV"),
    ],
)
def test_curve_generation_uses_curve_specific_primary_and_energy_range(
    curve_name,
    expected_primary,
    expected_energy_range,
    tmp_test_directory,
):
    args = _base_args(tmp_test_directory)
    output_root = Path(args["output_path"]).expanduser().resolve()
    curve_directory = output_root / curve_name

    mock_run_command, mock_build_scan, mock_generate_submission_script = (
        _run_generation_without_external_side_effects(curve_name, args, output_root)
    )

    command, working_directory = mock_run_command.call_args.args

    assert working_directory == output_root
    assert _command_value(command, "--primary") == expected_primary
    assert _command_value(command, "--energy_range") == expected_energy_range
    assert _command_value(command, "--label") == curve_name
    assert _command_value(command, "--output_file") == str(curve_directory / "base_grid.ecsv")

    mock_build_scan.assert_called_once()
    mock_generate_submission_script.assert_called_once()


def test_nsb_overwrite_files_contain_nsb_fields_and_trigger_scan(tmp_test_directory):
    args = _base_args(tmp_test_directory)
    output_root = Path(args["output_path"]).expanduser().resolve()
    curve_directory = output_root / "nsb"

    mock_run_command, mock_build_scan, mock_generate_submission_script = (
        _run_generation_without_external_side_effects("nsb", args, output_root)
    )

    overwrite_file = curve_directory / "overwrite_files" / "overwrite_asum220.yaml"
    assert overwrite_file.exists()

    overwrite = yaml.safe_load(overwrite_file.read_text(encoding="utf-8"))

    assert overwrite["model_version"] == args["model_version"]
    assert overwrite["description"] == "Tune for NSB telescope trigger scan"

    telescope_changes = overwrite["changes"]["LSTN-01"]
    assert telescope_changes["asum_threshold"]["value"] == 220
    assert telescope_changes["min_photons"]["value"] == 0
    assert telescope_changes["min_photoelectrons"]["value"] == 0

    site_changes = overwrite["changes"]["OBS-North"]
    assert site_changes["nsb_scaling_factor"]["value"] == 2

    build_scan_kwargs = mock_build_scan.call_args.kwargs
    labels = [label for _, label in build_scan_kwargs["overwrite_files_and_labels"]]
    assert labels == [f"asum{threshold}" for threshold in bias_curve_submissions._ASUM_THRESHOLDS]

    mock_run_command.assert_called_once()
    mock_generate_submission_script.assert_called_once()


def test_proton_overwrite_files_contain_only_trigger_scan(tmp_test_directory):
    args = _base_args(tmp_test_directory)
    output_root = Path(args["output_path"]).expanduser().resolve()
    curve_directory = output_root / "proton"

    mock_run_command, mock_build_scan, mock_generate_submission_script = (
        _run_generation_without_external_side_effects("proton", args, output_root)
    )

    overwrite_file = curve_directory / "overwrite_files" / "overwrite_asum220.yaml"
    assert overwrite_file.exists()

    overwrite = yaml.safe_load(overwrite_file.read_text(encoding="utf-8"))

    assert overwrite["model_version"] == args["model_version"]
    assert overwrite["description"] == "Tune for proton telescope trigger scan"

    changes = overwrite["changes"]
    assert set(changes.keys()) == {"LSTN-01"}
    assert changes["LSTN-01"] == {
        "asum_threshold": {
            "version": bias_curve_submissions._PARAMETER_VERSION,
            "value": 220,
        }
    }

    build_scan_kwargs = mock_build_scan.call_args.kwargs
    labels = [label for _, label in build_scan_kwargs["overwrite_files_and_labels"]]
    assert labels == [f"asum{threshold}" for threshold in bias_curve_submissions._ASUM_THRESHOLDS]

    mock_run_command.assert_called_once()
    mock_generate_submission_script.assert_called_once()


@pytest.mark.parametrize(
    ("telescope", "expected_param", "expected_values"),
    [
        (
            "LSTN-01",
            "asum_threshold",
            [220, 230, 240, 250, 260, 270, 280, 290, 300, 320, 340, 360],
        ),
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
