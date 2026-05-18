from unittest import mock

import astropy.units as u
import pytest
from astropy.tests.helper import assert_quantity_allclose
from simtools.production_configuration.production_grid import CorsikaLimitsLookup

from simtools.production_configuration.job_spec_builder import (
    build_job_specs,
    calculate_log_energy_midpoint,
    calculate_scaled_nshow,
    get_core_scatter_max_for_zenith_angle,
    get_energy_range_for_zenith_angle,
    normalize_energy_ranges,
    normalize_grid_axes,
    normalize_to_list,
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


@pytest.fixture
def corsika_limits():
    return "tests/resources/corsika_simulation_limits/merged_corsika_limits_for_test.ecsv"


def test_normalize_energy_ranges_expands_list_of_pairs():
    energy_ranges = normalize_energy_ranges(
        [
            (30 * u.GeV, 30 * u.GeV),
            (300 * u.GeV, 300 * u.GeV),
        ]
    )

    assert len(energy_ranges) == 2
    for actual, expected in zip(
        energy_ranges,
        [
            (30 * u.GeV, 30 * u.GeV),
            (300 * u.GeV, 300 * u.GeV),
        ],
    ):
        assert_quantity_allclose(actual[0], expected[0])
        assert_quantity_allclose(actual[1], expected[1])


def test_normalize_to_list_converts_tuple_values():
    assert normalize_to_list((1, 2)) == [1, 2]


def test_normalize_grid_axes_uses_defaults_and_none_for_missing_axes():
    grid_axes = normalize_grid_axes({"primary": "gamma"})

    assert grid_axes["primary"] == ["gamma"]
    assert grid_axes["azimuth_angle"] == [None]
    assert grid_axes["zenith_angle"] == [None]
    assert grid_axes["model_version"] == [None]
    assert grid_axes["corsika_le_interaction"] == ["urqmd"]
    assert grid_axes["corsika_he_interaction"] == ["epos"]


def test_normalize_energy_ranges_accepts_single_tuple_pair():
    energy_ranges = normalize_energy_ranges((30 * u.GeV, 300 * u.GeV))
    assert len(energy_ranges) == 1
    assert_quantity_allclose(energy_ranges[0][0], 30 * u.GeV)
    assert_quantity_allclose(energy_ranges[0][1], 300 * u.GeV)


def test_normalize_energy_ranges_raises_for_invalid_shape():
    with pytest.raises(ValueError, match="energy_range must be one pair"):
        normalize_energy_ranges([30 * u.GeV, 300])


def test_calculate_log_energy_midpoint():
    midpoint_energy = calculate_log_energy_midpoint((1 * u.GeV, 100 * u.GeV))

    assert midpoint_energy.to_value(u.GeV) == pytest.approx(10.0)


def test_calculate_log_energy_midpoint_raises_for_non_quantity_values():
    with pytest.raises(TypeError, match="energy_range_pair must contain astropy Quantity values"):
        calculate_log_energy_midpoint((1, 100 * u.GeV))


def test_calculate_log_energy_midpoint_raises_for_non_positive_values():
    with pytest.raises(ValueError, match="Energy range values must be strictly positive"):
        calculate_log_energy_midpoint((0 * u.GeV, 100 * u.GeV))


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


def test_calculate_scaled_nshow_raises_for_invalid_baseline():
    with pytest.raises(ValueError, match="baseline_nshow must be a positive integer"):
        calculate_scaled_nshow((10 * u.GeV, 100 * u.GeV), 0)


def test_calculate_scaled_nshow_requires_reference_energy_when_scaled():
    with pytest.raises(ValueError, match="reference_energy is required"):
        calculate_scaled_nshow((10 * u.GeV, 100 * u.GeV), 10, nshow_power_index=-1.0)


def test_calculate_scaled_nshow_raises_when_scaled_result_drops_below_one():
    with pytest.raises(ValueError, match="Scaled nshow must be at least 1"):
        calculate_scaled_nshow(
            (10 * u.GeV, 10 * u.GeV),
            1,
            nshow_power_index=1.0,
            reference_energy=-100 * u.GeV,
        )


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
    args_dict["nshow_reference_energy"] = 10 * u.GeV
    args_dict["energy_range"] = [
        (10 * u.GeV, 10 * u.GeV),
        (100 * u.GeV, 100 * u.GeV),
    ]

    job_specs = build_job_specs(args_dict, ["7.0.0"])
    nshow_values = [job_spec["nshow"] for job_spec in job_specs]
    assert nshow_values[0] == pytest.approx(100)
    assert nshow_values[1] == pytest.approx(10)


def test_build_job_specs_requires_reference_energy_for_nshow_scaling(args_dict):
    args_dict["number_of_runs"] = 1
    args_dict["nshow_power_index"] = -1.0
    args_dict["nshow_reference_energy"] = None

    with pytest.raises(ValueError, match="reference_energy is required"):
        build_job_specs(args_dict, ["7.0.0"])


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
    "simtools.production_configuration.job_spec_builder.calculate_scaled_nshow",
    return_value=777,
)
def test_build_job_specs_uses_dummy_nshow_when_corsika_limits_set(
    mock_nshow, args_dict, corsika_limits
):
    args_dict["corsika_limits"] = corsika_limits
    args_dict["telescope_ids"] = ["LSTN-01"]
    args_dict["number_of_runs"] = 1

    job_specs = build_job_specs(args_dict, ["7.0.0"])

    assert len(job_specs) == 1
    assert job_specs[0]["nshow"] == 777
    mock_nshow.assert_called_once()


def test_get_energy_range_for_zenith_angle_clips_to_lookup_threshold(corsika_limits):
    lookup = CorsikaLimitsLookup(corsika_limits, telescope_ids=["LSTN-01"])

    selected_energy_range = get_energy_range_for_zenith_angle(
        20 * u.deg,
        (1 * u.GeV, 10 * u.GeV),
        lookup,
        azimuth_angle=0 * u.deg,
    )

    assert_quantity_allclose(selected_energy_range[0], 7 * u.GeV)
    assert_quantity_allclose(selected_energy_range[1], 10 * u.GeV)


def test_get_energy_range_for_zenith_angle_returns_none_if_threshold_exceeds_max(corsika_limits):
    lookup = CorsikaLimitsLookup(corsika_limits, telescope_ids=["LSTN-01"])

    selected_energy_range = get_energy_range_for_zenith_angle(
        20 * u.deg,
        (1 * u.GeV, 5 * u.GeV),
        lookup,
        azimuth_angle=0 * u.deg,
    )

    assert selected_energy_range is None


def test_get_core_scatter_max_for_zenith_angle_clips_to_lookup_radius(corsika_limits):
    lookup = CorsikaLimitsLookup(corsika_limits, telescope_ids=["LSTN-01"])

    selected_core_scatter_max = get_core_scatter_max_for_zenith_angle(
        20 * u.deg,
        (10, 1000 * u.m),
        lookup,
        azimuth_angle=0 * u.deg,
    )

    assert_quantity_allclose(selected_core_scatter_max, 925 * u.m)


def test_build_job_specs_reads_limits_from_corsika_limits_file(args_dict, corsika_limits):
    args_dict["number_of_runs"] = 1
    args_dict["azimuth_angle"] = 0 * u.deg
    args_dict["energy_range"] = (1 * u.GeV, 10 * u.GeV)
    args_dict["core_scatter"] = (10, 1000 * u.m)
    args_dict["corsika_limits"] = corsika_limits
    args_dict["telescope_ids"] = ["LSTN-01"]

    job_specs = build_job_specs(args_dict, ["7.0.0"])

    assert len(job_specs) == 1
    assert_quantity_allclose(job_specs[0]["energy_min"], 7 * u.GeV)
    assert_quantity_allclose(job_specs[0]["energy_max"], 10 * u.GeV)
    assert_quantity_allclose(job_specs[0]["core_scatter_max"], 925 * u.m)
