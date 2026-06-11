from pathlib import Path
from unittest.mock import call, patch

import pytest

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


@patch("simtools.job_execution.bias_curve_submissions._build_htcondor_args")
@patch(
    "simtools.job_execution.bias_curve_submissions.htcondor_script_generator.generate_submission_script"
)
@patch("simtools.job_execution.bias_curve_submissions._build_scan_grid")
@patch("simtools.job_execution.bias_curve_submissions._generate_overwrite_files")
@patch("simtools.job_execution.bias_curve_submissions._run_command")
def test_generate_curve_submissions_forwards_curve_specific_energy_range(
    mock_run_command,
    mock_generate_overwrite_files,
    mock_build_scan_grid,
    mock_generate_submission_script,
    mock_build_htcondor_args,
    tmp_test_directory,
):
    args = _base_args(tmp_test_directory)
    output_root = Path(args["output_path"]).expanduser().resolve()
    curve_directory = output_root / "nsb"
    scan_grid_file = curve_directory / "scan_grid.ecsv"

    mock_generate_overwrite_files.return_value = [("overwrite.yaml", "asum_threshold_220")]
    mock_build_htcondor_args.return_value = {"output_path": "htcondor_submit"}

    bias_curve_submissions._generate_curve_submissions(
        "nsb",
        bias_curve_submissions._CURVE_DEFINITIONS["nsb"],
        args,
        output_root,
    )

    command = mock_run_command.call_args.args[0]
    assert command[command.index("--primary") + 1] == "gamma"
    assert command[command.index("--energy_range") + 1] == "20 MeV 25 MeV"
    assert command[command.index("--label") + 1] == "test_label_nsb"
    assert str(curve_directory / "base_grid.ecsv") in command

    mock_generate_overwrite_files.assert_called_once_with(
        curve_name="nsb",
        telescopes=["LSTN-01"],
        args=args,
        curve_directory=curve_directory,
        label="test_label_nsb",
    )
    mock_build_scan_grid.assert_called_once()
    mock_build_htcondor_args.assert_called_once_with(
        label="test_label_nsb",
        curve_directory=curve_directory,
        scan_grid_file=scan_grid_file,
        args=args,
    )
    mock_generate_submission_script.assert_called_once_with({"output_path": "htcondor_submit"})


@pytest.mark.parametrize(
    ("telescope", "threshold", "threshold_param"),
    [
        ("LSTN-01", 220, "asum_threshold"),
        ("MSTN-01", 22, "dsum_threshold"),
    ],
)
def test_build_proton_overwrite_sets_only_trigger_threshold(
    telescope,
    threshold,
    threshold_param,
):
    overwrite = bias_curve_submissions._build_proton_overwrite(
        telescopes=[telescope],
        threshold=threshold,
        site="North",
        model_version="7.0.0",
    )

    telescope_changes = overwrite["changes"][telescope]
    obs_changes = overwrite["changes"]["OBS-North"]

    assert telescope_changes == {
        threshold_param: {
            "version": "2.0.0",
            "value": threshold,
        }
    }
    assert "nsb_scaling_factor" not in obs_changes
    assert "min_photons" not in telescope_changes
    assert "min_photoelectrons" not in telescope_changes


@pytest.mark.parametrize(
    ("telescope", "threshold", "threshold_param"),
    [
        ("LSTN-01", 220, "asum_threshold"),
        ("MSTN-01", 22, "dsum_threshold"),
    ],
)
def test_build_nsb_overwrite_sets_nsb_fields_and_trigger_threshold(
    telescope,
    threshold,
    threshold_param,
):
    overwrite = bias_curve_submissions._build_nsb_overwrite(
        telescopes=[telescope],
        threshold=threshold,
        site="North",
        model_version="7.0.0",
    )

    telescope_changes = overwrite["changes"][telescope]
    obs_changes = overwrite["changes"]["OBS-North"]

    assert telescope_changes["min_photons"]["value"] == 0
    assert telescope_changes["min_photoelectrons"]["value"] == 0
    assert telescope_changes[threshold_param]["value"] == threshold
    assert obs_changes["nsb_scaling_factor"]["value"] == 2


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
