from pathlib import Path
from unittest import mock

import astropy.units as u
import pytest
import yaml
from jsonschema import ValidationError

from simtools.job_execution import htcondor_script_generator, parameter_scan_generator
from simtools.production_configuration.job_grid_io import serialize_job_grid


def _overwrite_base():
    return {
        "model_version": "7.0.0",
        "model_update": "patch_update",
        "model_version_history": ["7.0.0"],
        "description": "Parameter scan",
        "changes": {},
    }


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
    params, overwrite, job_grid_updates = parameter_scan_generator._parse_parameter_scan_config(
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
    assert job_grid_updates == {}
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


def test_parse_parameter_scan_config_wraps_scalar_value():
    params, _, _ = parameter_scan_generator._parse_parameter_scan_config(
        {
            "overwrite": {"changes": {}},
            "parameters": [
                {
                    "name": "asum_threshold",
                    "path": "changes.LSTN-01.asum_threshold",
                    "values": 220,
                }
            ],
        }
    )

    assert params[0]["values"] == [220]


def test_parse_parameter_scan_config_rejects_non_dict_job_grid_updates():
    with pytest.raises(TypeError, match=r"job_grid_updates.*must be a dictionary"):
        parameter_scan_generator._parse_parameter_scan_config(
            {
                "overwrite": {"changes": {}},
                "job_grid_updates": "not-a-dict",
                "parameters": [
                    {
                        "name": "asum_threshold",
                        "path": "changes.LSTN-01.asum_threshold",
                        "values": [220],
                    }
                ],
            }
        )


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
        "label": "scan / path",
        "parameter_scan": {
            "overwrite": _overwrite_base(),
            "parameters": [
                {
                    "name": "threshold value",
                    "path": "changes.LSTN-01.asum_threshold",
                    "values": ["20 MeV/25 MeV"],
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

    assert overwrite_file.name == "overwrite_scan-path_thresholdvalue_20MeV-25MeV.yaml"
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
            "overwrite": {**_overwrite_base(), "changes": {"LSTN-01": {}}},
            "job_grid_updates": {"telescope": "LSTN-01"},
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
    assert expanded_rows[0]["telescope"] == "LSTN-01"

    overwrite = yaml.safe_load(overwrite_file.read_text(encoding="utf-8"))
    assert overwrite["changes"]["LSTN-01"]["asum_threshold"] == {
        "version": "2.0.0",
        "value": 220,
    }


def test_expand_job_grid_with_scan_validates_configuration(tmp_test_directory):
    scan_config_path = Path(tmp_test_directory) / "scan_config.yml"
    scan_config_path.write_text(
        yaml.safe_dump({"parameter_scan": {"overwrite": {}, "parameters": []}}),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        parameter_scan_generator.expand_job_grid_with_scan(
            Path(tmp_test_directory) / "base_grid.ecsv",
            scan_config_path,
            Path(tmp_test_directory) / "scan_grid.ecsv",
        )


def test_scan_grid_is_consumed_by_htcondor_generator(tmp_test_directory):
    base_grid_file = Path(tmp_test_directory) / "base_grid.ecsv"
    scan_directory = Path(tmp_test_directory) / "scan output"
    scan_config_path = Path(tmp_test_directory) / "scan_config.yml"
    scan_grid_file = scan_directory / "scan_grid.ecsv"
    submit_directory = Path(tmp_test_directory) / "submit"
    image_file = Path(tmp_test_directory) / "simtools.sif"
    image_file.touch()

    serialize_job_grid(
        [
            {
                "run_number": 1,
                "primary": "gamma",
                "azimuth_angle": 0 * u.deg,
                "zenith_angle": 20 * u.deg,
                "energy_min": 20 * u.MeV,
                "energy_max": 25 * u.MeV,
                "cores_per_shower": 1,
                "core_scatter_max": 100 * u.m,
                "view_cone_min": 0 * u.deg,
                "view_cone_max": 5 * u.deg,
                "showers_per_run": 10,
                "nsb_rate": 1.0,
                "model_version": "7.0.0",
                "array_layout_name": "LSTN-01",
                "corsika_le_interaction": "urqmd",
                "corsika_he_interaction": "epos",
            }
        ],
        base_grid_file,
        metadata={"site": "North", "simulation_software": "corsika_sim_telarray"},
    )
    scan_config_path.write_text(
        yaml.safe_dump(
            {
                "label": "nsb",
                "parameter_scan": {
                    "overwrite": {**_overwrite_base(), "changes": {"LSTN-01": {}}},
                    "parameters": [
                        {
                            "name": "asum_threshold",
                            "path": "changes.LSTN-01.asum_threshold",
                            "version": "2.0.0",
                            "values": 220,
                            "label": "asum",
                            "label_separator": "",
                        }
                    ],
                    "job_grid_updates": {"telescope": "LSTN-01"},
                },
            }
        ),
        encoding="utf-8",
    )

    parameter_scan_generator.expand_job_grid_with_scan(
        base_grid_file, scan_config_path, scan_grid_file
    )
    htcondor_script_generator.generate_submission_script(
        {
            "output_path": submit_directory,
            "apptainer_image": str(image_file),
            "priority": 1,
            "job_grid_file": scan_grid_file,
            "label": "bias",
            "log_level": "INFO",
        }
    )

    params = (submit_directory / "simulate_prod.submit.params.txt").read_text(encoding="utf-8")
    submit_file = (submit_directory / "simulate_prod.submit.condor").read_text(encoding="utf-8")
    script = (submit_directory / "simulate_prod.submit.sh").read_text(encoding="utf-8")
    assert str(scan_directory / "overwrite_nsb_asum220.yaml") in params
    assert "'$(overwrite_model_parameters)'" in submit_file
    assert 'job_label="${job_label}_${scan_label}"' in script
    assert "--overwrite_model_parameters" in script
