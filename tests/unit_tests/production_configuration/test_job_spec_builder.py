from unittest import mock

import astropy.units as u
import pytest

from simtools.production_configuration.job_spec_builder import (
    build_job_specs,
    calculate_log_energy_midpoint,
    calculate_scaled_nshow,
    get_nshow_scaling_reference_energy,
    normalize_energy_ranges,
    resolve_array_layout_name,
)


@pytest.fixture
def args_dict():
    return {
        "azimuth_angle": 45 * u.deg,
        "zenith_angle": 20 * u.deg,
        "energy_range": [1 * u.GeV, 10 * u.GeV],
        "core_scatter": [0, 100 * u.m],
        "model_version": "v1.0",
        "array_layout_name": "test_layout",
        "primary": "gamma",
        "nshow": 1000,
        "run_number": 1,
        "number_of_runs": 10,
        "corsika_le_interaction": "urqmd",
        "corsika_he_interaction": "epos",
    }


def test_normalize_energy_ranges_expands_list_of_pairs():
    energy_ranges = normalize_energy_ranges(
        [
            (30 * u.GeV, 30 * u.GeV),
            (300 * u.GeV, 300 * u.GeV),
        ]
    )

    assert energy_ranges == [
        (30 * u.GeV, 30 * u.GeV),
        (300 * u.GeV, 300 * u.GeV),
    ]


def test_calculate_log_energy_midpoint():
    midpoint_energy = calculate_log_energy_midpoint((1 * u.GeV, 100 * u.GeV))

    assert midpoint_energy.to_value(u.GeV) == pytest.approx(10.0)


def test_get_nshow_scaling_reference_energy_uses_first_range():
    reference_energy = get_nshow_scaling_reference_energy(
        [(10 * u.GeV, 10 * u.GeV), (100 * u.GeV, 100 * u.GeV)]
    )

    assert reference_energy.to_value(u.GeV) == pytest.approx(10.0)


def test_calculate_scaled_nshow_returns_baseline_without_power_index():
    scaled_nshow = calculate_scaled_nshow((10 * u.GeV, 100 * u.GeV), 50)

    assert scaled_nshow == 50


def test_calculate_scaled_nshow_scales_against_reference_energy():
    scaled_nshow = calculate_scaled_nshow(
        (100 * u.GeV, 100 * u.GeV),
        100,
        nshow_power_index=-1.0,
        reference_energy=10 * u.GeV,
    )

    assert scaled_nshow == 10


def test_calculate_scaled_nshow_uses_ceil_for_fractional_result():
    scaled_nshow = calculate_scaled_nshow(
        (100 * u.GeV, 100 * u.GeV),
        10,
        nshow_power_index=-2.0,
        reference_energy=10 * u.GeV,
    )

    assert scaled_nshow == 1


def test_build_job_specs_expands_model_version_list(args_dict):
    args_dict["model_version"] = ["6.3.0", "7.0.0"]

    job_specs = build_job_specs(args_dict, ["7.0.0"])
    model_versions = {job_spec["model_version"] for job_spec in job_specs}

    assert model_versions == {"6.3.0", "7.0.0"}
    assert len(job_specs) == 2 * args_dict["number_of_runs"]


def test_build_job_specs_scales_nshow_by_energy_range(args_dict):
    args_dict["number_of_runs"] = 1
    args_dict["nshow"] = 100
    args_dict["nshow_power_index"] = -1.0
    args_dict["energy_range"] = [
        (10 * u.GeV, 10 * u.GeV),
        (100 * u.GeV, 100 * u.GeV),
    ]

    job_specs = build_job_specs(args_dict, ["7.0.0"])

    assert [job_spec["nshow"] for job_spec in job_specs] == [100, 10]


def test_build_job_specs_uses_default_interaction_models_when_missing(args_dict):
    args_dict.pop("corsika_le_interaction")
    args_dict.pop("corsika_he_interaction")

    job_specs = build_job_specs(args_dict, ["7.0.0"])

    assert {job_spec["corsika_le_interaction"] for job_spec in job_specs} == {"urqmd"}
    assert {job_spec["corsika_he_interaction"] for job_spec in job_specs} == {"epos"}


def test_build_job_specs_increments_run_number(args_dict):
    args_dict["number_of_runs"] = 2
    args_dict["run_number"] = 10
    args_dict["model_version"] = ["6.3.0", "7.0.0"]

    job_specs = build_job_specs(args_dict, ["7.0.0"])
    run_numbers = [job_spec["run_number"] for job_spec in job_specs]

    assert run_numbers == [10, 11, 12, 13]


@mock.patch(
    "simtools.production_configuration.job_spec_builder.get_energy_range_for_zenith_angle",
    return_value=None,
)
def test_build_job_specs_skips_entries_when_energy_range_is_none(mock_energy_range, args_dict):
    args_dict["corsika_limits"] = "limits.ecsv"
    args_dict["number_of_runs"] = 1

    job_specs = build_job_specs(args_dict, ["7.0.0"])

    assert job_specs == []
    mock_energy_range.assert_called_once()


@mock.patch(
    "simtools.production_configuration.job_spec_builder.get_nshow_for_energy_range_and_zenith_angle",
    return_value=777,
)
def test_build_job_specs_uses_dummy_nshow_when_corsika_limits_set(mock_nshow, args_dict):
    args_dict["corsika_limits"] = "limits.ecsv"
    args_dict["number_of_runs"] = 1

    job_specs = build_job_specs(args_dict, ["7.0.0"])

    assert len(job_specs) == 1
    assert job_specs[0]["nshow"] == 777
    mock_nshow.assert_called_once()


def test_resolve_array_layout_name_resolves_stringified_by_version_layout():
    array_layout_name = str(
        {
            "by_version": {
                "<7.0.0": "alpha",
                ">=7.0.0": "CTAO-North-Alpha",
            }
        }
    )

    assert resolve_array_layout_name(array_layout_name, "7.0.0") == "CTAO-North-Alpha"
