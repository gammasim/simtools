from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml
from astropy.table import Table
from jsonschema.exceptions import ValidationError

from simtools.io.io_handler import IOHandler
from simtools.production_configuration import parameter_scans

_MODULE = "simtools.production_configuration.parameter_scans"
_OVERWRITE_BASE = {
    "model_version": "7.0.0",
    "model_update": "patch_update",
    "model_version_history": ["7.0.0"],
    "description": "Parameter scan",
    "changes": {},
}


def _base_args(tmp_test_directory):
    return {
        "site": "North",
        "model_version": "7.0.0",
        "array_layout_name": "LSTN-01",
        "threshold_parameter": "asum_threshold",
        "simulation_software": "corsika_sim_telarray",
        "azimuth_angle": 0.0,
        "zenith_angle": 20.0,
        "showers_per_run": 10000,
        "core_scatter": "20 1900 m",
        "view_cone": "0 deg 5 deg",
        "number_of_runs": 10,
        "corsika_le_interaction": "urqmd",
        "corsika_he_interaction": "epos",
        "nsb_energy_range": "20 MeV 25 MeV",
        "proton_energy_range": "2 GeV 2000 GeV",
        "nsb_scaling_factor": 2,
        "trigger_thresholds": None,
        "output_path": str(Path(tmp_test_directory) / "bias_curves"),
    }


def _run_expansion(tmp_test_directory, scan_config):
    """Write a scan config, expand a mocked base grid, and return (rows, metadata)."""
    scan_config_path = Path(tmp_test_directory) / "scan_config.yml"
    scan_config_path.write_text(yaml.safe_dump(scan_config), encoding="utf-8")
    with (
        patch(f"{_MODULE}.read_job_grid", return_value=([{"primary": "gamma"}], {"site": "North"})),
        patch(f"{_MODULE}.serialize_job_grid") as mock_serialize,
    ):
        parameter_scans.expand_job_grid_with_scan(
            Path(tmp_test_directory) / "base_grid.ecsv",
            scan_config_path,
            Path(tmp_test_directory) / "scan_grid.ecsv",
        )
    return mock_serialize.call_args.args[0], mock_serialize.call_args.kwargs["metadata"]


def _scan_config(parameter_scan):
    return {"label": "scan", "parameter_scan": parameter_scan}


@pytest.mark.parametrize(
    ("value", "expected"),
    [("20 MeV/25 MeV", "20MeV-25MeV"), (220.0, "220"), (2.5, "2.5")],
)
def test_format_value_for_name(value, expected):
    assert parameter_scans._format_value_for_name(value) == expected


def test_set_nested_parameter_creates_missing_path_and_sets_version():
    data = {}
    parameter_scans._set_nested_parameter(
        data, ["changes", "LSTN-01", "asum_threshold"], 220.0, version="2.0.0"
    )
    assert data == {"changes": {"LSTN-01": {"asum_threshold": {"version": "2.0.0", "value": 220}}}}


def test_set_nested_parameter_replaces_non_dict_leaf():
    data = {"changes": {"LSTN-01": {"asum_threshold": 0}}}
    parameter_scans._set_nested_parameter(data, ["changes", "LSTN-01", "asum_threshold"], 230)
    assert data["changes"]["LSTN-01"]["asum_threshold"] == {"value": 230}


def test_set_nested_parameter_rejects_non_dict_intermediate():
    with pytest.raises(TypeError, match="is not a dict"):
        parameter_scans._set_nested_parameter(
            {"changes": "not-a-dict"}, ["changes", "LSTN-01", "asum_threshold"], 220
        )


def test_expand_job_grid_with_scan_uses_sanitized_default_label(tmp_test_directory):
    scan_config = _scan_config(
        {
            "overwrite": {**_OVERWRITE_BASE, "changes": {}},
            "parameters": [
                {
                    "name": "threshold value",
                    "path": "changes.LSTN-01.asum_threshold",
                    "values": ["20 MeV/25 MeV"],
                }
            ],
        }
    )
    rows, metadata = _run_expansion(tmp_test_directory, scan_config)

    assert rows[0]["model_parameter_set"] == "thresholdvalue_20MeV-25MeV"
    assert rows[0]["scan_label"] == "thresholdvalue_20MeV-25MeV"
    assert metadata["model_parameter_sets"] == {
        "thresholdvalue_20MeV-25MeV": {"LSTN-01": {"asum_threshold": {"value": "20 MeV/25 MeV"}}}
    }


def test_expand_job_grid_with_scan_uses_explicit_label_version_and_job_grid_updates(
    tmp_test_directory,
):
    scan_config = _scan_config(
        {
            "overwrite": {**_OVERWRITE_BASE, "changes": {"LSTN-01": {}}},
            "job_grid_updates": {"array_layout_name": "LSTN-01"},
            "parameters": [
                {
                    "name": "asum_threshold",
                    "path": "changes.LSTN-01.asum_threshold",
                    "version": "2.0.0",
                    "values": [220],
                    "label": "asum",
                    "label_separator": "",
                }
            ],
        }
    )
    rows, metadata = _run_expansion(tmp_test_directory, scan_config)

    assert rows[0]["model_parameter_set"] == "asum220"
    assert rows[0]["scan_label"] == "asum220"
    assert rows[0]["array_layout_name"] == "LSTN-01"
    assert metadata["model_parameter_sets"]["asum220"]["LSTN-01"]["asum_threshold"] == {
        "version": "2.0.0",
        "value": 220,
    }


def test_expand_job_grid_with_scan_builds_cartesian_product(tmp_test_directory):
    scan_config = _scan_config(
        {
            "overwrite": _OVERWRITE_BASE,
            "parameters": [
                {"name": "a", "path": "changes.X.a", "values": [1, 2], "label": "a"},
                {"name": "b", "path": "changes.X.b", "values": 3, "label": "b"},
            ],
        }
    )
    rows, metadata = _run_expansion(tmp_test_directory, scan_config)

    assert sorted(metadata["model_parameter_sets"]) == ["a_1_b_3", "a_2_b_3"]
    assert len(rows) == 2


def test_expand_job_grid_with_scan_validates_configuration(tmp_test_directory):
    scan_config_path = Path(tmp_test_directory) / "scan_config.yml"
    scan_config_path.write_text(
        yaml.safe_dump({"parameter_scan": {"overwrite": {}, "parameters": []}}),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        parameter_scans.expand_job_grid_with_scan(
            Path(tmp_test_directory) / "base_grid.ecsv",
            scan_config_path,
            Path(tmp_test_directory) / "scan_grid.ecsv",
        )


@pytest.mark.parametrize(
    ("meta", "expected_keys"),
    [
        (
            {
                "model_parameter_sets": {"set1": {"param": 1}},
                "site": "South",
                "simulation_software": "sim_telarray",
                "job_grid_summary": {"total": 10},
                "job_grid_format_version": "1.0",
                "unwanted_key": "should_be_removed",
            },
            {
                "model_parameter_sets",
                "site",
                "simulation_software",
                "job_grid_summary",
                "job_grid_format_version",
            },
        ),
        ({"some_other_key": "value", "random_key": 456}, set()),
    ],
)
def test_clean_scan_grid_metadata_keeps_only_essential_keys(
    tmp_test_directory, meta, expected_keys
):
    output_file = Path(tmp_test_directory) / "scan_grid.ecsv"
    table = Table([[1, 2], [3, 4]], names=["col1", "col2"])
    table.meta = meta
    table.write(output_file, format="ascii.ecsv", overwrite=True)

    parameter_scans._clean_scan_grid_metadata(str(output_file))

    assert set(Table.read(output_file, format="ascii.ecsv").meta) == expected_keys


@patch(f"{_MODULE}.generate_job_grid")
@patch(f"{_MODULE}.expand_job_grid_with_scan")
def test_generate_scan_grids_generates_both_curves_without_mutating_args(
    mock_expand, mock_generate_job_grid, tmp_test_directory
):
    args = _base_args(tmp_test_directory)
    original_args = dict(args)
    io_handler = IOHandler()
    io_handler.set_paths(output_path=Path(args["output_path"]).resolve())

    with (
        patch(f"{_MODULE}.resolve_array_layout_name", return_value="LSTN-01") as mock_layout,
        patch(f"{_MODULE}._threshold_param_name", return_value="asum_threshold") as mock_param,
    ):
        parameter_scans.generate_scan_grids(args, io_handler)

    assert args == original_args
    mock_layout.assert_called_once_with("LSTN-01", "7.0.0")
    mock_param.assert_called_once_with(args, "LSTN-01")
    assert [call.args[0]["label"] for call in mock_generate_job_grid.call_args_list] == [
        "nsb",
        "proton",
    ]
    primaries = [call.args[0]["primary"] for call in mock_generate_job_grid.call_args_list]
    assert primaries == ["gamma", "proton"]
    assert mock_expand.call_count == 2


def test_generate_scan_grid_writes_scan_config_and_expands_grid(tmp_test_directory):
    args = _base_args(tmp_test_directory)
    output_root = Path(args["output_path"]).expanduser().resolve()
    io_handler = IOHandler()
    io_handler.set_paths(output_path=output_root)
    curve_directory = output_root / "nsb"

    with (
        patch(f"{_MODULE}.generate_job_grid") as mock_generate_job_grid,
        patch(f"{_MODULE}.expand_job_grid_with_scan") as mock_expand_job_grid,
    ):
        parameter_scans._generate_scan_grid(
            "nsb",
            ("gamma", "nsb_energy_range"),
            {**args, "threshold_parameter": "asum_threshold"},
            "LSTN-01",
            io_handler,
        )

    scan_config_file = curve_directory / "scan_config.yaml"
    base_grid_file = curve_directory / "base_grid.ecsv"
    scan_grid_file = curve_directory / "scan_grid.ecsv"
    scan_config = yaml.safe_load(scan_config_file.read_text(encoding="utf-8"))
    assert scan_config["label"] == "nsb"
    assert scan_config["parameter_scan"]["parameters"][0]["label"] == "asum"
    assert scan_config["parameter_scan"]["parameters"][0]["label_separator"] == ""
    assert not (curve_directory / "workflow.yaml").exists()

    production_config = mock_generate_job_grid.call_args.args[0]
    assert production_config["primary"] == "gamma"
    assert production_config["energy_range"] == "20 MeV 25 MeV"
    assert production_config["label"] == "nsb"
    assert "telescope" not in production_config
    mock_expand_job_grid.assert_called_once_with(base_grid_file, scan_config_file, scan_grid_file)


@pytest.mark.parametrize(
    ("default_trigger", "expected_param"),
    [
        ("AnalogSum", "asum_threshold"),
        ("DigitalSum", "dsum_threshold"),
        ("Majority", "discriminator_threshold"),
    ],
)
def test_threshold_parameter_is_chosen_from_telescope_model(
    default_trigger, expected_param, tmp_test_directory
):
    args = _base_args(tmp_test_directory)
    with patch(f"{_MODULE}.TelescopeModel") as model:
        model.return_value.get_parameter_value.return_value = default_trigger
        assert parameter_scans._threshold_param_name(args, "LSTN-01") == expected_param
    model.assert_called_once_with(site="North", telescope_name="LSTN-01", model_version="7.0.0")


@pytest.mark.parametrize(
    ("threshold_param", "expected_values"),
    [
        (
            "asum_threshold",
            [220, 230, 240, 250, 260, 270, 280, 290, 300, 320, 340, 360],
        ),
        ("dsum_threshold", [22, 23, 24, 25, 26, 27, 28, 29, 30]),
        ("discriminator_threshold", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    ],
)
def test_threshold_values_use_parameter_defaults(threshold_param, expected_values):
    assert parameter_scans._threshold_values(threshold_param) == expected_values


def test_threshold_values_expand_configured_thresholds():
    assert parameter_scans._threshold_values("asum_threshold", [225, 3, 10]) == [225, 235, 245]


@pytest.mark.parametrize(
    ("trigger_thresholds", "error"),
    [
        ([225, 2], "must contain minimum threshold"),
        ([225, 0, 10], "positive integer"),
        ([225, 2.5, 10], "positive integer"),
        ([225, 2, 0], "step size must be positive"),
    ],
)
def test_invalid_configured_thresholds_are_rejected(trigger_thresholds, error):
    with pytest.raises(ValueError, match=error):
        parameter_scans._threshold_values("asum_threshold", trigger_thresholds)


@pytest.mark.parametrize("curve_name", ["nsb", "proton"])
def test_scan_config_contains_curve_overwrite_and_trigger_scan(tmp_test_directory, curve_name):
    args = {**_base_args(tmp_test_directory), "threshold_parameter": "asum_threshold"}
    scan_config = parameter_scans._scan_config(curve_name, "LSTN-01", args)

    assert scan_config["label"] == curve_name
    overwrite = scan_config["parameter_scan"]["overwrite"]
    assert (
        overwrite["description"]
        == f"Tune for {'NSB' if curve_name == 'nsb' else 'proton'} telescope trigger scan"
    )
    assert overwrite["changes"]["OBS-North"] == {"nsb_scaling_factor": {"value": 2}}
    if curve_name == "nsb":
        assert overwrite["changes"]["LSTN-01"] == {
            "min_photons": {"value": 0},
            "min_photoelectrons": {"value": 0},
        }
    else:
        assert overwrite["changes"]["LSTN-01"] == {}
    assert scan_config["parameter_scan"]["parameters"] == [
        {
            "name": "asum_threshold",
            "path": "changes.LSTN-01.asum_threshold",
            "values": [220, 230, 240, 250, 260, 270, 280, 290, 300, 320, 340, 360],
            "label": "asum",
            "label_separator": "",
        }
    ]
    assert "job_grid_updates" not in scan_config["parameter_scan"]


def test_validate_required_args_rejects_missing_required_argument(tmp_test_directory):
    args = _base_args(tmp_test_directory)
    args["site"] = ""
    with pytest.raises(ValueError, match="Missing required argument: --site"):
        parameter_scans.generate_scan_grids(args, Mock())
