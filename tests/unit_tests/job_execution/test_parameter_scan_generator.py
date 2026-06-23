from pathlib import Path
from unittest import mock

import pytest
import yaml

from simtools.job_execution import parameter_scan_generator


def test_format_value_for_name_sanitizes_spaces_and_slashes():
    assert parameter_scan_generator._format_value_for_name("20 MeV/25 MeV") == "20MeV-25MeV"
    assert parameter_scan_generator._format_value_for_name(220.0) == "220"


def test_set_nested_parameter_creates_missing_path_and_sets_version():
    data = {}

    parameter_scan_generator._set_nested_parameter(
        data,
        ["changes", "LSTN-01", "asum_threshold"],
        220.0,
        version="2.0.0",
    )

    assert data == {
        "changes": {
            "LSTN-01": {
                "asum_threshold": {
                    "version": "2.0.0",
                    "value": 220,
                }
            }
        }
    }


def test_set_nested_parameter_replaces_non_dict_leaf():
    data = {"changes": {"LSTN-01": {"asum_threshold": 0}}}

    parameter_scan_generator._set_nested_parameter(
        data,
        ["changes", "LSTN-01", "asum_threshold"],
        230,
    )

    assert data["changes"]["LSTN-01"]["asum_threshold"] == {"value": 230}


def test_set_nested_parameter_rejects_non_dict_intermediate():
    data = {"changes": "not-a-dict"}

    with pytest.raises(TypeError, match="intermediate key 'changes' is not a dictionary"):
        parameter_scan_generator._set_nested_parameter(
            data,
            ["changes", "LSTN-01", "asum_threshold"],
            220,
        )


def test_parse_parameter_scan_config_requires_overwrite():
    with pytest.raises(KeyError, match="requires 'overwrite'"):
        parameter_scan_generator._parse_parameter_scan_config(
            {
                "parameters": [
                    {
                        "name": "asum_threshold",
                        "path": "changes.LSTN-01.asum_threshold",
                        "values": [220],
                    }
                ]
            }
        )


def test_parse_parameter_scan_config_requires_overwrite_dict():
    with pytest.raises(TypeError, match="must be a dictionary"):
        parameter_scan_generator._parse_parameter_scan_config(
            {
                "overwrite": "not-a-dict",
                "parameters": [
                    {
                        "name": "asum_threshold",
                        "path": "changes.LSTN-01.asum_threshold",
                        "values": [220],
                    }
                ],
            }
        )


def test_parse_parameter_scan_config_adds_label_fields():
    params, overwrite = parameter_scan_generator._parse_parameter_scan_config(
        {
            "overwrite": {"changes": {}},
            "parameters": [
                {
                    "name": "asum_threshold",
                    "path": "changes.LSTN-01.asum_threshold",
                    "values": [220],
                    "label": "asum",
                    "label_separator": "",
                }
            ],
        }
    )

    assert overwrite == {"changes": {}}
    assert params == [
        {
            "name": "asum_threshold",
            "path": "changes.LSTN-01.asum_threshold",
            "values": [220],
            "version": None,
            "label": "asum",
            "label_separator": "",
        }
    ]


def test_generate_parameter_combinations_uses_compact_labels():
    combinations = parameter_scan_generator._generate_parameter_combinations(
        [
            {
                "name": "asum_threshold",
                "path": "changes.LSTN-01.asum_threshold",
                "values": [220, 230],
                "version": "2.0.0",
                "label": "asum",
                "label_separator": "",
            }
        ]
    )

    assert combinations == [
        {
            "combo": {
                "asum_threshold": {
                    "path": "changes.LSTN-01.asum_threshold",
                    "value": 220,
                    "version": "2.0.0",
                }
            },
            "name": "asum220",
        },
        {
            "combo": {
                "asum_threshold": {
                    "path": "changes.LSTN-01.asum_threshold",
                    "value": 230,
                    "version": "2.0.0",
                }
            },
            "name": "asum230",
        },
    ]


@mock.patch("simtools.job_execution.parameter_scan_generator.serialize_job_grid")
@mock.patch("simtools.job_execution.parameter_scan_generator.read_job_grid")
def test_expand_job_grid_with_scan_uses_sanitized_default_label_and_description(
    mock_read_grid,
    mock_serialize_grid,
    tmp_test_directory,
):
    mock_read_grid.return_value = ([{"primary": "gamma"}], {"site": "North"})

    scan_config = {
        "parameter_scan": {
            "overwrite": {
                "changes": {},
            },
            "parameters": [
                {
                    "name": "threshold value",
                    "path": "changes.LSTN-01.asum_threshold",
                    "values": ["20 MeV/25 MeV"],
                }
            ],
        }
    }
    scan_config_path = Path(tmp_test_directory) / "scan_config.yml"
    scan_config_path.write_text(yaml.safe_dump(scan_config), encoding="utf-8")

    output_file = Path(tmp_test_directory) / "scan_grid.ecsv"

    parameter_scan_generator.expand_job_grid_with_scan(
        Path(tmp_test_directory) / "base_grid.ecsv",
        scan_config_path,
        output_file,
    )

    expanded_rows = mock_serialize_grid.call_args.args[0]
    overwrite_file = Path(expanded_rows[0]["overwrite_model_parameters"])

    assert overwrite_file.name == "overwrite_scan_thresholdvalue_20MeV-25MeV.yaml"
    assert expanded_rows[0]["scan_label"] == "thresholdvalue_20MeV-25MeV"

    overwrite = yaml.safe_load(overwrite_file.read_text(encoding="utf-8"))
    assert overwrite["description"] == "Parameter scan - threshold value=20 MeV/25 MeV"


@mock.patch("simtools.job_execution.parameter_scan_generator.serialize_job_grid")
@mock.patch("simtools.job_execution.parameter_scan_generator.read_job_grid")
def test_expand_job_grid_with_scan_uses_explicit_compact_label(
    mock_read_grid,
    mock_serialize_grid,
    tmp_test_directory,
):
    mock_read_grid.return_value = ([{"primary": "gamma"}], {"site": "North"})

    scan_config = {
        "label": "nsb",
        "parameter_scan": {
            "overwrite": {
                "changes": {"LSTN-01": {}},
            },
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
        },
    }
    scan_config_path = Path(tmp_test_directory) / "scan_config.yml"
    scan_config_path.write_text(yaml.safe_dump(scan_config), encoding="utf-8")

    output_file = Path(tmp_test_directory) / "scan_grid.ecsv"

    parameter_scan_generator.expand_job_grid_with_scan(
        Path(tmp_test_directory) / "base_grid.ecsv",
        scan_config_path,
        output_file,
    )

    expanded_rows = mock_serialize_grid.call_args.args[0]
    overwrite_file = Path(expanded_rows[0]["overwrite_model_parameters"])

    assert overwrite_file.name == "overwrite_nsb_asum220.yaml"
    assert expanded_rows[0]["scan_label"] == "asum220"

    overwrite = yaml.safe_load(overwrite_file.read_text(encoding="utf-8"))
    assert overwrite["changes"]["LSTN-01"]["asum_threshold"] == {
        "version": "2.0.0",
        "value": 220,
    }
