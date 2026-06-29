from pathlib import Path
from unittest.mock import patch

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
        "output_path": str(Path(tmp_test_directory) / "bias_curves"),
    }


@patch("simtools.job_execution.bias_curve_submissions._generate_curve_submissions")
@patch(
    "simtools.job_execution.bias_curve_submissions._resolve_telescope_from_layout",
    return_value="LSTN-01",
)
def test_generate_bias_curve_submissions_uses_configured_curve_definitions_without_mutating_args(
    mock_resolve_telescope,
    mock_generate_curve_submissions,
    tmp_test_directory,
):
    args = _base_args(tmp_test_directory)
    original_args = dict(args)

    bias_curve_submissions.generate_bias_curve_submissions(args)

    assert args == original_args
    mock_resolve_telescope.assert_called_once_with(args)
    assert [
        item.kwargs["curve_name"] for item in mock_generate_curve_submissions.call_args_list
    ] == [
        "nsb",
        "proton",
    ]
    assert all(
        item.kwargs["args"]["telescope"] == "LSTN-01"
        for item in mock_generate_curve_submissions.call_args_list
    )
    assert (
        mock_generate_curve_submissions.call_args_list[0].kwargs["curve_definition"]
        == bias_curve_submissions._CURVE_DEFINITIONS["nsb"]
    )


def test_resolve_telescopes_from_layout_returns_single_valid_telescope(tmp_test_directory):
    args = _base_args(tmp_test_directory)

    with patch("simtools.job_execution.bias_curve_submissions.SiteModel") as mock_site_model:
        mock_site_model.return_value.get_array_elements_for_layout.return_value = ["LSTN-01"]

        assert bias_curve_submissions._resolve_telescope_from_layout(args) == "LSTN-01"


def test_resolve_telescopes_from_layout_enforces_single_telescope(tmp_test_directory):
    args = _base_args(tmp_test_directory)

    with patch("simtools.job_execution.bias_curve_submissions.SiteModel") as mock_site_model:
        mock_site_model.return_value.get_array_elements_for_layout.return_value = [
            "LSTN-01",
            "LSTN-02",
        ]

        with pytest.raises(ValueError, match="single-telescope layouts"):
            bias_curve_submissions._resolve_telescope_from_layout(args)


def test_resolve_telescopes_from_layout_rejects_invalid_telescope(tmp_test_directory):
    args = _base_args(tmp_test_directory)

    with patch("simtools.job_execution.bias_curve_submissions.SiteModel") as mock_site_model:
        mock_site_model.return_value.get_array_elements_for_layout.return_value = ["Invalid-LST"]

        with pytest.raises(ValueError, match="resolved to invalid telescope"):
            bias_curve_submissions._resolve_telescope_from_layout(args)


@pytest.mark.parametrize(
    ("telescope", "expected_param", "expected_values"),
    [
        ("LSTN-01", "asum_threshold", bias_curve_submissions._ASUM_THRESHOLDS),
        ("MSTN-01", "dsum_threshold", bias_curve_submissions._DSUM_THRESHOLDS),
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


def test_threshold_param_name_rejects_unknown_telescope_type():
    with pytest.raises(ValueError, match="Supported telescope types"):
        bias_curve_submissions._threshold_param_name("XSTN-01")


def test_threshold_values_rejects_unsupported_threshold_parameter(monkeypatch):
    monkeypatch.setattr(
        bias_curve_submissions,
        "_threshold_param_name",
        lambda telescope: "unknown_threshold",
    )

    with pytest.raises(ValueError, match="Unsupported threshold parameter"):
        bias_curve_submissions._threshold_values_for_telescope("LSTN-01")


@pytest.mark.parametrize(
    ("threshold_param", "expected_prefix"),
    [
        ("asum_threshold", "asum"),
        ("dsum_threshold", "dsum"),
    ],
)
def test_threshold_label_prefix_uses_short_parameter_name(threshold_param, expected_prefix):
    assert bias_curve_submissions._threshold_label_prefix(threshold_param) == expected_prefix


def test_parameter_scan_entry_uses_compact_threshold_label(tmp_test_directory):
    entry = bias_curve_submissions._parameter_scan_entry("LSTN-01")

    assert entry == {
        "name": "asum_threshold",
        "path": "changes.LSTN-01.asum_threshold",
        "version": bias_curve_submissions._PARAMETER_VERSION,
        "values": bias_curve_submissions._ASUM_THRESHOLDS,
        "label": "asum",
        "label_separator": "",
    }


def test_scan_config_contains_nsb_base_overwrite_and_trigger_scan(tmp_test_directory):
    args = _base_args(tmp_test_directory)

    scan_config = bias_curve_submissions._scan_config("nsb", "LSTN-01", args)

    assert scan_config["label"] == "nsb"
    overwrite = scan_config["parameter_scan"]["overwrite"]
    assert overwrite["model_version"] == args["model_version"]
    assert overwrite["description"] == "Tune for NSB telescope trigger scan"

    telescope_changes = overwrite["changes"]["LSTN-01"]
    assert telescope_changes["min_photons"]["value"] == 0
    assert telescope_changes["min_photoelectrons"]["value"] == 0
    assert "asum_threshold" not in telescope_changes

    assert overwrite["changes"]["OBS-North"]["nsb_scaling_factor"]["value"] == 2
    assert scan_config["parameter_scan"]["parameters"] == [
        bias_curve_submissions._parameter_scan_entry("LSTN-01")
    ]
    assert scan_config["parameter_scan"]["job_grid_updates"] == {"telescope": "LSTN-01"}


def test_scan_config_contains_proton_base_overwrite_and_trigger_scan(tmp_test_directory):
    args = _base_args(tmp_test_directory)

    scan_config = bias_curve_submissions._scan_config("proton", "LSTN-01", args)

    assert scan_config["label"] == "proton"
    overwrite = scan_config["parameter_scan"]["overwrite"]
    assert overwrite["model_version"] == args["model_version"]
    assert overwrite["description"] == "Tune for proton telescope trigger scan"
    assert overwrite["changes"] == {
        "LSTN-01": {},
        "OBS-North": {
            "nsb_scaling_factor": {
                "version": bias_curve_submissions._PARAMETER_VERSION,
                "value": 2,
            }
        },
    }
    assert scan_config["parameter_scan"]["parameters"] == [
        bias_curve_submissions._parameter_scan_entry("LSTN-01")
    ]


def test_base_overwrite_rejects_unknown_curve(tmp_test_directory):
    args = _base_args(tmp_test_directory)

    with pytest.raises(ValueError, match="Unsupported curve name"):
        bias_curve_submissions._base_overwrite("unknown", "LSTN-01", args)


@pytest.mark.parametrize(
    ("curve_name", "expected_primary", "expected_energy_range"),
    [
        ("nsb", "gamma", "20 MeV 25 MeV"),
        ("proton", "proton", "2 GeV 2000 GeV"),
    ],
)
def test_production_grid_configuration_uses_curve_specific_primary_and_energy_range(
    curve_name,
    expected_primary,
    expected_energy_range,
    tmp_test_directory,
):
    args = _base_args(tmp_test_directory)
    args["telescope"] = "LSTN-01"
    base_grid_file = Path(tmp_test_directory) / "base_grid.ecsv"

    curve_definition = bias_curve_submissions._CURVE_DEFINITIONS[curve_name]
    configuration = bias_curve_submissions._production_grid_configuration(
        args,
        curve_definition,
        base_grid_file,
        curve_name,
    )

    assert configuration["primary"] == expected_primary
    assert configuration["energy_range"] == expected_energy_range
    assert configuration["label"] == curve_name
    assert configuration["output_file"] == str(base_grid_file)
    assert configuration["telescope"] == "LSTN-01"


def test_workflow_config_runs_expected_applications_in_order(tmp_test_directory):
    args = _base_args(tmp_test_directory)
    output_root = Path(args["output_path"]).expanduser().resolve()
    curve_directory = output_root / "nsb"
    base_grid_file = curve_directory / "base_grid.ecsv"
    scan_config_file = curve_directory / "scan_config.yaml"
    scan_grid_file = curve_directory / "scan_grid.ecsv"

    workflow_config = bias_curve_submissions._workflow_config(
        curve_name="nsb",
        curve_definition=bias_curve_submissions._CURVE_DEFINITIONS["nsb"],
        args=args,
        base_grid_file=base_grid_file,
        scan_config_file=scan_config_file,
        scan_grid_file=scan_grid_file,
    )

    applications = workflow_config["applications"]
    assert [app["application"] for app in applications] == [
        "simtools-production-generate-grid",
        "simtools-generate-parameter-scan-grid",
    ]

    assert applications[0]["configuration"]["output_file"] == str(base_grid_file)
    assert applications[1]["configuration"] == {
        "job_grid_file": str(base_grid_file),
        "scan_config": str(scan_config_file),
        "output_file": str(scan_grid_file),
    }


def test_run_workflow_uses_simtools_runner(tmp_test_directory):
    args = _base_args(tmp_test_directory)
    workflow_file = Path(tmp_test_directory) / "workflow.yaml"

    with patch(
        "simtools.job_execution.bias_curve_submissions.simtools_runner.run_applications"
    ) as mock_run_applications:
        bias_curve_submissions._run_workflow(workflow_file, args)

    mock_run_applications.assert_called_once_with(
        {
            "config_file": str(workflow_file),
            "steps": None,
            "activity_id": args.get("activity_id"),
            "ignore_runtime_environment": True,
        }
    )


def test_generate_curve_submissions_writes_configs_and_runs_workflow(tmp_test_directory):
    args = _base_args(tmp_test_directory)
    args["telescope"] = "LSTN-01"
    output_root = Path(args["output_path"]).expanduser().resolve()
    curve_directory = output_root / "nsb"
    io_handler = bias_curve_submissions.IOHandler()
    io_handler.set_paths(output_path=output_root)

    with patch(
        "simtools.job_execution.bias_curve_submissions.simtools_runner.run_applications"
    ) as mock_run_applications:
        bias_curve_submissions._generate_curve_submissions(
            curve_name="nsb",
            curve_definition=bias_curve_submissions._CURVE_DEFINITIONS["nsb"],
            args=args,
            io_handler=io_handler,
        )

    scan_config_file = curve_directory / "scan_config.yaml"
    workflow_file = curve_directory / "workflow.yaml"

    assert scan_config_file.exists()
    assert workflow_file.exists()

    scan_config = yaml.safe_load(scan_config_file.read_text(encoding="utf-8"))
    assert scan_config["label"] == "nsb"
    assert scan_config["parameter_scan"]["parameters"][0]["label"] == "asum"
    assert scan_config["parameter_scan"]["parameters"][0]["label_separator"] == ""
    assert scan_config["parameter_scan"]["job_grid_updates"] == {"telescope": "LSTN-01"}

    workflow_config = yaml.safe_load(workflow_file.read_text(encoding="utf-8"))
    assert [app["application"] for app in workflow_config["applications"]] == [
        "simtools-production-generate-grid",
        "simtools-generate-parameter-scan-grid",
    ]

    mock_run_applications.assert_called_once_with(
        {
            "config_file": str(workflow_file),
            "steps": None,
            "activity_id": args.get("activity_id"),
            "ignore_runtime_environment": True,
        }
    )


def test_validate_required_args_rejects_missing_required_argument(tmp_test_directory):
    args = _base_args(tmp_test_directory)
    args["site"] = ""

    with pytest.raises(ValueError, match="Missing required argument: --site"):
        bias_curve_submissions._validate_required_args(args)
