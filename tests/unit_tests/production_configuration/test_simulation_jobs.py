from unittest.mock import Mock, patch

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.tests.helper import assert_quantity_allclose

from simtools.production_configuration.corsika_limits_lookup import CorsikaLimitsLookup
from simtools.production_configuration.simulation_jobs import (
    _build_rows_for_point,
    _clip_energy_range_from_threshold,
    _clip_max_quantity,
    _generate_observation_grids_per_layout,
    _generate_observation_points_from_axes,
    _iter_compact_axis_specs,
    _normalize_axis_spec_tokens,
    _parse_axis_range_tokens,
    _parse_axis_spec,
    _resolve_coordinate_system,
    _resolve_shower_params,
    _scale_total_showers,
    build_axes_dict_from_cli_args,
    build_job_grid_metadata,
    build_observing_location,
    build_production_grid_engine,
    build_simulation_jobs,
    calculate_log_energy_midpoint,
    calculate_scaled_showers_per_run,
    calculate_zenith_scaled_showers_per_run,
    get_core_scatter_max_for_zenith_angle,
    get_energy_range_for_zenith_angle,
    get_viewcone_max_for_zenith_angle,
    normalize_energy_ranges,
    normalize_grid_axes,
    resolve_single_model_version,
    resolve_time_of_observation,
)


def test_resolve_single_model_version_uses_first_list_entry():
    assert resolve_single_model_version(["7.0.0", "7.1.0"]) == "7.0.0"
    assert resolve_single_model_version("7.0.0") == "7.0.0"


def test_resolve_time_of_observation_returns_none_for_horizontal_without_input():
    args_dict = {
        "axis": [["azimuth", "0", "deg", "1", "deg", "2"], ["zenith", "0", "deg", "1", "deg", "2"]]
    }
    assert resolve_time_of_observation(None, args_dict) is None


def test_build_axes_dict_from_cli_args_builds_horizontal_grid():
    axes = build_axes_dict_from_cli_args(
        {
            "axis": [
                ["azimuth", "310", "deg", "20", "deg", "3", "linear"],
                ["zenith", "30", "deg", "40", "deg", "2", "linear"],
                ["nsb", "4", "MHz", "5", "MHz", "2", "linear"],
                ["offset", "0", "deg", "10", "deg", "2", "linear"],
            ]
        }
    )

    assert "azimuth" in axes
    assert "zenith_angle" in axes
    assert "ra" not in axes
    assert "dec" not in axes


def test_build_axes_dict_from_cli_args_accepts_compact_axis_definitions():
    axes = build_axes_dict_from_cli_args(
        {
            "axis": [
                ["azimuth", "310", "deg", "20", "deg", "3", "linear"],
                ["zenith", "30", "deg", "40", "deg", "2"],
                ["nsb", "4", "MHz", "5", "MHz", "2"],
                ["offset", "0", "deg", "10", "deg", "2"],
            ]
        }
    )

    assert axes == {
        "nsb_level": {"range": [4.0, 5.0], "binning": 2, "scaling": "linear", "units": "MHz"},
        "offset": {"range": [0.0, 10.0], "binning": 2, "scaling": "linear", "units": "deg"},
        "azimuth": {"range": [310.0, 20.0], "binning": 3, "scaling": "linear", "units": "deg"},
        "zenith_angle": {
            "range": [30.0, 40.0],
            "binning": 2,
            "scaling": "linear",
            "units": "deg",
        },
    }


def test_build_axes_dict_from_cli_args_accepts_config_wrapped_axis_strings():
    axes = build_axes_dict_from_cli_args(
        {
            "axis": [
                ["azimuth 310 deg 20 deg 3 linear"],
                ["zenith 30 deg 40 deg 2"],
                ["nsb 4 MHz 5 MHz 2"],
                ["offset 0 deg 10 deg 2"],
            ]
        }
    )

    assert axes["azimuth"]["range"] == [310.0, 20.0]
    assert axes["zenith_angle"]["binning"] == 2
    assert axes["nsb_level"]["units"] == "MHz"


def test_build_axes_dict_from_cli_args_accepts_merged_config_axis_list():
    axes = build_axes_dict_from_cli_args(
        {
            "axis": [
                [
                    "azimuth 310 deg 20 deg 3 linear",
                    "zenith 30 deg 40 deg 2",
                    "nsb 4 MHz 5 MHz 2",
                    "offset 0 deg 10 deg 2",
                ]
            ]
        }
    )

    assert axes["azimuth"]["range"] == [310.0, 20.0]
    assert axes["zenith_angle"]["binning"] == 2
    assert axes["offset"]["range"] == [0.0, 10.0]


def test_build_axes_dict_from_cli_args_rejects_multiple_direction_coordinate_systems():
    with pytest.raises(ValueError, match="Cannot define both azimuth/zenith and ra/dec axes"):
        build_axes_dict_from_cli_args(
            {
                "axis": [
                    ["azimuth", "310", "deg", "20", "deg", "3"],
                    ["zenith", "30", "deg", "40", "deg", "2"],
                    ["ra", "0", "deg", "360", "deg", "36"],
                    ["dec", "-90", "deg", "90", "deg", "18"],
                    ["nsb", "4", "MHz", "5", "MHz", "2"],
                    ["offset", "0", "deg", "10", "deg", "2"],
                ]
            }
        )


def test_resolve_time_of_observation_raises_for_radec_without_input():
    args_dict = {
        "axis": [["ra", "0", "deg", "1", "deg", "2"], ["dec", "0", "deg", "1", "deg", "2"]]
    }
    with pytest.raises(ValueError, match="time_of_observation"):
        resolve_time_of_observation(None, args_dict)


def test_build_job_grid_metadata_includes_job_context():
    metadata = build_job_grid_metadata(
        {
            "site": "North",
            "simulation_software": "corsika_sim_telarray",
            "axis": [
                ["ra", "0", "deg", "1", "deg", "2"],
                ["dec", "0", "deg", "1", "deg", "2"],
            ],
            "time_of_observation": "2017-09-16 00:00:00",
            "corsika_limits": "limits.ecsv",
        }
    )

    assert metadata["site"] == "North"
    assert metadata["simulation_software"] == "corsika_sim_telarray"
    assert metadata["coordinate_system"] == "ra_dec"
    assert metadata["time_of_observation_utc"].startswith("2017-09-16T00:00:00")
    assert metadata["corsika_limits"] == "limits.ecsv"


@patch("simtools.production_configuration.simulation_jobs.SiteModel")
def test_build_observing_location_uses_site_model(mock_site_model):
    model = mock_site_model.return_value
    model.get_parameter_value_with_unit.side_effect = [1 * u.deg, 2 * u.deg, 3 * u.m]

    location = build_observing_location("North", ["7.0.0"])

    assert isinstance(location, EarthLocation)
    mock_site_model.assert_called_once_with(model_version="7.0.0", site="North")


@patch("simtools.production_configuration.simulation_jobs.ProductionGridEngine")
def test_build_production_grid_engine_resolves_layout_name(
    mock_production_grid_engine,
):
    args_dict = {
        "array_layout_name": {"by_version": {"<7.0.0": "alpha", ">=7.0.0": "beta"}},
        "model_version": ["7.0.0"],
        "time_of_observation": None,
        "corsika_limits": "limits.ecsv",
        "axis": [
            ["azimuth", "310", "deg", "20", "deg", "3", "linear"],
            ["zenith", "30", "deg", "40", "deg", "2", "linear"],
            ["nsb", "4", "MHz", "5", "MHz", "2", "linear"],
            ["offset", "0", "deg", "10", "deg", "2", "linear"],
        ],
    }

    build_production_grid_engine(args_dict)

    mock_production_grid_engine.assert_called_once_with(
        axes={
            "nsb_level": {"range": [4.0, 5.0], "binning": 2, "scaling": "linear", "units": "MHz"},
            "offset": {"range": [0.0, 10.0], "binning": 2, "scaling": "linear", "units": "deg"},
            "azimuth": {
                "range": [310.0, 20.0],
                "binning": 3,
                "scaling": "linear",
                "units": "deg",
            },
            "zenith_angle": {
                "range": [30.0, 40.0],
                "binning": 2,
                "scaling": "linear",
                "units": "deg",
            },
        },
        coordinate_system="horizontal",
        observing_location=None,
        time_of_observation=None,
        lookup_table="limits.ecsv",
        array_layout_name="beta",
    )


@patch("simtools.production_configuration.simulation_jobs.build_observing_location")
@patch("simtools.production_configuration.simulation_jobs.ProductionGridEngine")
def test_build_production_grid_engine_builds_observing_location_for_radec(
    mock_production_grid_engine,
    mock_build_observing_location,
):
    location = EarthLocation(lat=1 * u.deg, lon=2 * u.deg, height=3 * u.m)
    mock_build_observing_location.return_value = location

    build_production_grid_engine(
        {
            "site": "North",
            "array_layout_name": "alpha",
            "model_version": ["7.0.0"],
            "time_of_observation": "2017-09-16 00:00:00",
            "corsika_limits": None,
            "axis": [
                ["ra", "0", "deg", "360", "deg", "36", "linear"],
                ["dec", "-90", "deg", "90", "deg", "18", "linear"],
                ["nsb", "4", "MHz", "4", "MHz", "1", "linear"],
                ["offset", "0", "deg", "10", "deg", "2", "linear"],
            ],
        }
    )

    mock_build_observing_location.assert_called_once_with(site="North", model_version=["7.0.0"])
    assert mock_production_grid_engine.call_args.kwargs["observing_location"] == location


def test_normalize_grid_axes_applies_defaults():
    normalized = normalize_grid_axes(
        {
            "primary": "gamma",
            "azimuth_angle": 180 * u.deg,
            "zenith_angle": 20 * u.deg,
            "model_version": "7.0.0",
        }
    )

    assert normalized["primary"] == ["gamma"]
    assert normalized["corsika_le_interaction"] == ["urqmd"]
    assert normalized["corsika_he_interaction"] == ["epos"]


@pytest.mark.parametrize(
    ("energy_range", "expected_length"),
    [
        ((30 * u.GeV, 100 * u.GeV), 1),
        ([[30 * u.GeV, 100 * u.GeV], [100 * u.GeV, 1 * u.TeV]], 2),
    ],
)
def test_normalize_energy_ranges_accepts_supported_shapes(energy_range, expected_length):
    assert len(normalize_energy_ranges(energy_range)) == expected_length


def test_normalize_energy_ranges_rejects_invalid_shape():
    with pytest.raises(ValueError, match="energy_range"):
        normalize_energy_ranges([30 * u.GeV])


def test_normalize_energy_ranges_accepts_quantity_pair_list():
    normalized = normalize_energy_ranges([30 * u.GeV, 100 * u.GeV])

    assert normalized == [(30 * u.GeV, 100 * u.GeV)]


def test_get_energy_range_for_zenith_angle_without_lookup_returns_configured_pair():
    energy_range = (30 * u.GeV, 100 * u.GeV)

    assert get_energy_range_for_zenith_angle(20 * u.deg, energy_range, None) == energy_range


@patch.object(CorsikaLimitsLookup, "__init__", return_value=None)
@patch.object(
    CorsikaLimitsLookup,
    "interpolate_point",
    return_value={
        "lower_energy_threshold": 0.2,
        "upper_scatter_radius": 100.0,
        "viewcone_radius": 2.0,
    },
)
def test_get_energy_range_for_zenith_angle_wraps_lookup_path_and_skips_step(
    mock_interpolate_point,
    mock_init,
):

    energy_range = get_energy_range_for_zenith_angle(
        20 * u.deg,
        (30 * u.GeV, 100 * u.GeV),
        "limits.ecsv",
    )

    assert energy_range is None
    mock_init.assert_called_once_with("limits.ecsv")
    mock_interpolate_point.assert_called_once()


@patch.object(CorsikaLimitsLookup, "__init__", return_value=None)
@patch.object(
    CorsikaLimitsLookup,
    "interpolate_point",
    return_value={
        "lower_energy_threshold": 0.01,
        "upper_scatter_radius": 100.0,
        "viewcone_radius": 2.0,
    },
)
def test_get_energy_range_for_zenith_angle_keeps_range_below_threshold(
    mock_interpolate_point,
    mock_init,
):

    energy_range = get_energy_range_for_zenith_angle(
        20 * u.deg,
        (30 * u.GeV, 100 * u.GeV),
        "limits.ecsv",
    )

    assert energy_range == (30 * u.GeV, 100 * u.GeV)
    mock_init.assert_called_once_with("limits.ecsv")
    mock_interpolate_point.assert_called_once()


def test_get_energy_range_for_zenith_angle_clips_threshold():
    corsika_limits = CorsikaLimitsLookup(
        "tests/resources/corsika_simulation_limits/corsika_limits_for_test.ecsv"
    )
    corsika_limits.interpolate_point = Mock(
        return_value={
            "lower_energy_threshold": 0.05,
            "upper_scatter_radius": 100.0,
            "viewcone_radius": 2.0,
        }
    )

    energy_range = get_energy_range_for_zenith_angle(
        20 * u.deg,
        (30 * u.GeV, 100 * u.GeV),
        corsika_limits,
    )

    assert_quantity_allclose(energy_range[0], 50 * u.GeV)
    assert_quantity_allclose(energy_range[1], 100 * u.GeV)


def test_get_core_scatter_max_for_zenith_angle_clips_value():
    corsika_limits = CorsikaLimitsLookup(
        "tests/resources/corsika_simulation_limits/corsika_limits_for_test.ecsv"
    )
    corsika_limits.interpolate_point = Mock(
        return_value={
            "lower_energy_threshold": 0.01,
            "upper_scatter_radius": 100.0,
            "viewcone_radius": 2.0,
        }
    )

    scatter_max = get_core_scatter_max_for_zenith_angle(
        20 * u.deg,
        [10, 150 * u.m],
        corsika_limits,
    )

    assert scatter_max == 100 * u.m


def test_get_core_scatter_max_for_zenith_angle_without_lookup_uses_configured_max():
    assert get_core_scatter_max_for_zenith_angle(20 * u.deg, [10, 150 * u.m], None) == 150 * u.m


@patch.object(CorsikaLimitsLookup, "__init__", return_value=None)
@patch.object(
    CorsikaLimitsLookup,
    "interpolate_point",
    return_value={
        "lower_energy_threshold": 0.01,
        "upper_scatter_radius": 100.0,
        "viewcone_radius": 2.0,
    },
)
def test_get_core_scatter_max_for_zenith_angle_uses_lookup_path(mock_interpolate_point, mock_init):
    scatter_max = get_core_scatter_max_for_zenith_angle(
        20 * u.deg,
        [10, 150 * u.m],
        "limits.ecsv",
    )

    assert scatter_max == 100 * u.m
    mock_init.assert_called_once_with("limits.ecsv")
    mock_interpolate_point.assert_called_once()


def test_get_viewcone_max_for_zenith_angle_without_lookup_uses_configured_max():
    assert get_viewcone_max_for_zenith_angle(20 * u.deg, [0 * u.deg, 5 * u.deg], None) == 5 * u.deg


@patch.object(CorsikaLimitsLookup, "__init__", return_value=None)
@patch.object(
    CorsikaLimitsLookup,
    "interpolate_point",
    return_value={
        "lower_energy_threshold": 0.01,
        "upper_scatter_radius": 100.0,
        "viewcone_radius": 2.0,
    },
)
def test_get_viewcone_max_for_zenith_angle_uses_lookup_path(mock_interpolate_point, mock_init):

    viewcone_max = get_viewcone_max_for_zenith_angle(
        20 * u.deg,
        [0 * u.deg, 5 * u.deg],
        "limits.ecsv",
    )

    assert viewcone_max == 2 * u.deg
    mock_init.assert_called_once_with("limits.ecsv")
    mock_interpolate_point.assert_called_once()


def test_calculate_log_energy_midpoint_returns_geometric_mean():
    midpoint = calculate_log_energy_midpoint((10 * u.GeV, 1 * u.TeV))

    assert_quantity_allclose(midpoint, 100 * u.GeV)


def test_calculate_log_energy_midpoint_raises_for_non_quantities():
    with pytest.raises(TypeError, match="Quantity"):
        calculate_log_energy_midpoint((10, 100 * u.GeV))


def test_calculate_log_energy_midpoint_raises_for_non_positive_energy():
    with pytest.raises(ValueError, match="strictly positive"):
        calculate_log_energy_midpoint((0 * u.GeV, 100 * u.GeV))


def test_calculate_scaled_showers_per_run_returns_baseline_without_power_law():
    assert calculate_scaled_showers_per_run((30 * u.GeV, 100 * u.GeV), 5) == 5


def test_calculate_scaled_showers_per_run_raises_for_non_positive_baseline():
    with pytest.raises(ValueError, match="positive integer"):
        calculate_scaled_showers_per_run((30 * u.GeV, 100 * u.GeV), 0)


def test_calculate_scaled_showers_per_run_raises_for_invalid_power_law_tuple():
    with pytest.raises(ValueError, match="exactly two values"):
        calculate_scaled_showers_per_run(
            (30 * u.GeV, 100 * u.GeV), 5, showers_per_run_power_law=(1.0,)
        )


def test_calculate_scaled_showers_per_run_scales_from_midpoint_energy():
    scaled_showers_per_run = calculate_scaled_showers_per_run(
        (10 * u.GeV, 1 * u.TeV),
        5,
        showers_per_run_power_law=(1.0, 10 * u.GeV),
    )

    assert scaled_showers_per_run == 50


@patch("simtools.production_configuration.simulation_jobs.np.ceil", return_value=0)
def test_calculate_scaled_showers_per_run_raises_when_scaled_value_is_below_one(mock_ceil):
    with pytest.raises(ValueError, match="at least 1"):
        calculate_scaled_showers_per_run(
            (10 * u.GeV, 1 * u.TeV),
            5,
            showers_per_run_power_law=(1.0, 10 * u.GeV),
        )
    mock_ceil.assert_called_once()


def test_calculate_zenith_scaled_showers_per_run_returns_baseline_for_fixed_mode():
    assert calculate_zenith_scaled_showers_per_run(20 * u.deg, 1000, "fixed") == 1000


def test_calculate_zenith_scaled_showers_per_run_scales_with_cosine():
    expected = int(np.ceil(1000 * np.round(np.cos(np.radians(60)), decimals=12)))
    assert calculate_zenith_scaled_showers_per_run(60 * u.deg, 1000, "inverse_cosine") == expected


def test_calculate_zenith_scaled_showers_per_run_keeps_baseline_at_zenith_0():
    assert calculate_zenith_scaled_showers_per_run(0 * u.deg, 1000, "inverse_cosine") == 1000


def test_calculate_zenith_scaled_showers_per_run_raises_for_non_positive_baseline():
    with pytest.raises(ValueError, match="positive integer"):
        calculate_zenith_scaled_showers_per_run(20 * u.deg, 0, "inverse_cosine")


def test_calculate_zenith_scaled_showers_per_run_raises_at_zenith_90():
    with pytest.raises(ValueError, match="at least 1"):
        calculate_zenith_scaled_showers_per_run(90 * u.deg, 1000, "inverse_cosine")


def test_calculate_zenith_scaled_showers_per_run_raises_near_zenith_90():
    # Rounding makes this tiny cosine effectively zero, which must trigger validation.
    with pytest.raises(ValueError, match="at least 1"):
        calculate_zenith_scaled_showers_per_run(89.999999999999 * u.deg, 1000, "inverse_cosine")


def test_clip_energy_range_from_threshold_returns_none_above_max():
    assert (
        _clip_energy_range_from_threshold(
            (30 * u.GeV, 100 * u.GeV),
            200 * u.GeV,
        )
        is None
    )


def test_clip_energy_range_from_threshold_returns_original_without_threshold():
    energy_range = (30 * u.GeV, 100 * u.GeV)

    assert _clip_energy_range_from_threshold(energy_range, None) == energy_range


def test_clip_max_quantity_returns_configured_value_without_lookup():
    assert _clip_max_quantity(5 * u.deg, None) == 5 * u.deg


def test_resolve_shower_params_converts_showers_per_run_power_law():
    (
        showers_per_run,
        power_law,
        showers_per_run_zenith_scaling,
        total_showers,
        total_showers_scaling,
    ) = _resolve_shower_params(
        {
            "showers_per_run": 5,
            "showers_per_run_power_law": ["1.0", "100", "GeV"],
        }
    )

    assert showers_per_run == 5
    assert power_law[0] == pytest.approx(1.0)
    assert power_law[1] == 100 * u.GeV
    assert showers_per_run_zenith_scaling == "fixed"
    assert total_showers is None
    assert total_showers_scaling == "fixed"


def test_resolve_shower_params_accepts_power_law_as_compact_string():
    (
        showers_per_run,
        power_law,
        showers_per_run_zenith_scaling,
        total_showers,
        total_showers_scaling,
    ) = _resolve_shower_params(
        {
            "showers_per_run": 5,
            "showers_per_run_power_law": "1.0 100 GeV",
        }
    )

    assert showers_per_run == 5
    assert power_law[0] == pytest.approx(1.0)
    assert power_law[1] == 100 * u.GeV
    assert showers_per_run_zenith_scaling == "fixed"
    assert total_showers is None
    assert total_showers_scaling == "fixed"


def test_resolve_shower_params_raises_for_invalid_power_law_shape():
    with pytest.raises(ValueError, match="must be provided as"):
        _resolve_shower_params(
            {
                "showers_per_run": 5,
                "showers_per_run_power_law": ["1.0", "100 GeV"],
            }
        )


def test_resolve_shower_params_accepts_showers_per_run_zenith_scaling():
    (
        _showers_per_run,
        _power_law,
        showers_per_run_zenith_scaling,
        _total_showers,
        _total_showers_scaling,
    ) = _resolve_shower_params(
        {
            "showers_per_run": 5,
            "showers_per_run_zenith_scaling": "inverse_cosine",
        }
    )
    assert showers_per_run_zenith_scaling == "inverse_cosine"


def test_resolve_shower_params_raises_for_invalid_showers_per_run_zenith_scaling():
    with pytest.raises(ValueError, match="showers_per_run_zenith_scaling must be one of"):
        _resolve_shower_params(
            {
                "showers_per_run": 5,
                "showers_per_run_zenith_scaling": "bad_mode",
            }
        )


def test_build_rows_for_point_skips_energy_ranges_below_threshold():
    rows = _build_rows_for_point(
        point_base={"primary": "gamma", "zenith_angle": 20 * u.deg},
        energy_ranges=[(30 * u.GeV, 40 * u.GeV), (50 * u.GeV, 100 * u.GeV)],
        lower_energy_threshold=45 * u.GeV,
        showers_per_run=5,
        showers_per_run_power_law=None,
        number_of_runs=2,
        total_showers=None,
        total_showers_scaling="fixed",
        run_number=10,
    )

    assert [row["run_number"] for row in rows] == [10, 11]
    assert all(row["energy_min"] == 50 * u.GeV for row in rows)
    assert all(row["showers_per_run"] == 5 for row in rows)


def test_build_rows_for_point_rounds_total_showers_up_with_warning(caplog):
    caplog.set_level("WARNING")

    rows = _build_rows_for_point(
        point_base={"primary": "gamma", "zenith_angle": 20 * u.deg},
        energy_ranges=[(30 * u.GeV, 100 * u.GeV)],
        lower_energy_threshold=None,
        showers_per_run=1000,
        showers_per_run_power_law=None,
        number_of_runs=2,
        total_showers=2500,
        total_showers_scaling="fixed",
        run_number=1,
    )

    assert [row["showers_per_run"] for row in rows] == [1000, 1000, 1000]
    assert [row["run_number"] for row in rows] == [1, 2, 3]
    assert "adjusting to 3000 to keep equal showers per run" in caplog.text


def test_build_rows_for_point_scales_total_showers_with_zenith_scaled():
    rows = _build_rows_for_point(
        point_base={"primary": "gamma", "zenith_angle": 60 * u.deg},
        energy_ranges=[(30 * u.GeV, 100 * u.GeV)],
        lower_energy_threshold=None,
        showers_per_run=200,
        showers_per_run_power_law=None,
        number_of_runs=1,
        total_showers=2500,
        total_showers_scaling="zenith_scaled",
        run_number=1,
    )

    assert [row["showers_per_run"] for row in rows] == [200, 200]
    assert [row["run_number"] for row in rows] == [1, 2]


def test_build_rows_for_point_uses_custom_zenith_angle_scaling_factor():
    rows = _build_rows_for_point(
        point_base={"primary": "gamma", "zenith_angle": 60 * u.deg},
        energy_ranges=[(30 * u.GeV, 100 * u.GeV)],
        lower_energy_threshold=None,
        showers_per_run=1000,
        showers_per_run_power_law=None,
        number_of_runs=1,
        total_showers=2500,
        total_showers_scaling="zenith_scaled",
        run_number=1,
        zenith_angle_scaling_factor=0.0,
    )

    assert [row["showers_per_run"] for row in rows] == [1000, 1000, 1000]
    assert [row["run_number"] for row in rows] == [1, 2, 3]


def test_build_rows_for_point_scales_showers_per_run_with_zenith():
    rows = _build_rows_for_point(
        point_base={"primary": "gamma", "zenith_angle": 60 * u.deg},
        energy_ranges=[(30 * u.GeV, 100 * u.GeV)],
        lower_energy_threshold=None,
        showers_per_run=1000,
        showers_per_run_power_law=None,
        showers_per_run_zenith_scaling="inverse_cosine",
        number_of_runs=2,
        total_showers=None,
        total_showers_scaling="fixed",
        run_number=1,
    )

    assert [row["showers_per_run"] for row in rows] == [500, 500]


def test_generate_observation_points_from_axes_adds_lookup_limits():
    corsika_limits = Mock()
    corsika_limits.interpolate_point.return_value = {
        "lower_energy_threshold": 0.05,
        "upper_scatter_radius": 150.0,
        "viewcone_radius": 3.0,
    }

    points = _generate_observation_points_from_axes([180 * u.deg], [20 * u.deg], corsika_limits)

    assert len(points) == 1
    assert_quantity_allclose(points[0]["lower_energy_threshold"], 0.05 * u.TeV)
    assert_quantity_allclose(points[0]["scatter_radius"], 150 * u.m)
    assert_quantity_allclose(points[0]["viewcone_radius"], 3 * u.deg)


@patch("simtools.production_configuration.simulation_jobs._generate_observation_points_from_axes")
@patch("simtools.production_configuration.simulation_jobs.CorsikaLimitsLookup")
def test_generate_observation_grids_per_layout_uses_layout_specific_lookup(
    mock_corsika_limits_lookup,
    mock_generate_observation_points_from_axes,
):
    mock_generate_observation_points_from_axes.return_value = [{"azimuth": 0 * u.deg}]
    args_dict = {
        "array_layout_name": {"by_version": {"<7.0.0": "alpha", ">=7.0.0": "beta"}},
        "corsika_limits": "limits.ecsv",
    }
    grid_axes = {
        "model_version": ["6.3.0", "7.0.0"],
        "azimuth_angle": [0 * u.deg],
        "zenith_angle": [20 * u.deg],
    }

    observation_grids, resolved_layout_names = _generate_observation_grids_per_layout(
        args_dict, grid_axes
    )

    assert set(observation_grids) == {"alpha", "beta"}
    assert resolved_layout_names == {"6.3.0": "alpha", "7.0.0": "beta"}
    mock_corsika_limits_lookup.assert_any_call("limits.ecsv", array_layout_name="alpha")
    mock_corsika_limits_lookup.assert_any_call("limits.ecsv", array_layout_name="beta")


@patch("simtools.production_configuration.simulation_jobs.build_production_grid_engine")
def test_generate_observation_grids_per_layout_uses_shared_axes_and_skips_duplicate_layouts(
    mock_build_production_grid_engine,
):
    mock_build_production_grid_engine.return_value.generate_simulation_grid.return_value = [
        {"azimuth": 0 * u.deg}
    ]

    observation_grids, resolved_layout_names = _generate_observation_grids_per_layout(
        {
            "axis": [
                ["azimuth", "310", "deg", "20", "deg", "3"],
                ["zenith", "20", "deg", "40", "deg", "2"],
                ["nsb", "4", "MHz", "5", "MHz", "2"],
                ["offset", "0", "deg", "10", "deg", "2"],
            ],
            "array_layout_name": {"by_version": {"<7.0.0": "alpha", ">=7.0.0": "alpha"}},
        },
        {
            "model_version": ["6.3.0", "7.0.0"],
            "azimuth_angle": [0 * u.deg],
            "zenith_angle": [20 * u.deg],
        },
    )

    assert observation_grids == {"alpha": [{"azimuth": 0 * u.deg}]}
    assert resolved_layout_names == {"6.3.0": "alpha", "7.0.0": "alpha"}
    mock_build_production_grid_engine.assert_called_once()


@patch("simtools.production_configuration.simulation_jobs._generate_observation_grids_per_layout")
def test_build_simulation_jobs_expands_runs_from_observation_grid(
    mock_generate_observation_grids_per_layout,
):
    mock_generate_observation_grids_per_layout.return_value = (
        {
            "alpha": [
                {
                    "azimuth": 180 * u.deg,
                    "zenith_angle": 20 * u.deg,
                    "lower_energy_threshold": 40 * u.GeV,
                    "scatter_radius": 100 * u.m,
                    "viewcone_radius": 2 * u.deg,
                }
            ]
        },
        {"6.3.0": "alpha"},
    )
    rows = build_simulation_jobs(
        {
            "primary": ["gamma"],
            "azimuth_angle": [180 * u.deg],
            "zenith_angle": [20 * u.deg],
            "model_version": ["6.3.0"],
            "corsika_le_interaction": "urqmd",
            "corsika_he_interaction": "epos",
            "energy_range": [(30 * u.GeV, 100 * u.GeV)],
            "core_scatter": [10, 200 * u.m],
            "view_cone": [0 * u.deg, 5 * u.deg],
            "showers_per_run": 5,
            "number_of_runs": 2,
            "run_number": 11,
            "array_layout_name": "alpha",
        }
    )

    assert [row["run_number"] for row in rows] == [11, 12]
    assert rows[0]["array_layout_name"] == "alpha"
    assert rows[0]["core_scatter_max"] == 100 * u.m
    assert rows[0]["view_cone_max"] == 2 * u.deg
    assert rows[0]["showers_per_run"] == 5


@patch("simtools.production_configuration.simulation_jobs._generate_observation_grids_per_layout")
def test_build_simulation_jobs_raises_for_total_showers_and_number_of_runs(
    mock_generate_observation_grids_per_layout,
):
    mock_generate_observation_grids_per_layout.return_value = ({"alpha": []}, {"6.3.0": "alpha"})

    with pytest.raises(ValueError, match="total_showers and number_of_runs"):
        build_simulation_jobs(
            {
                "primary": ["gamma"],
                "azimuth_angle": [180 * u.deg],
                "zenith_angle": [20 * u.deg],
                "model_version": ["6.3.0"],
                "corsika_le_interaction": "urqmd",
                "corsika_he_interaction": "epos",
                "energy_range": [(30 * u.GeV, 100 * u.GeV)],
                "core_scatter": [10, 200 * u.m],
                "view_cone": [0 * u.deg, 5 * u.deg],
                "showers_per_run": 5,
                "number_of_runs": 2,
                "total_showers": 100,
                "run_number": 11,
                "array_layout_name": "alpha",
            }
        )


def test_parse_axis_range_tokens_with_single_token():
    result = _parse_axis_range_tokens(["30 deg .. 40 deg"])

    assert len(result) == 2


def test_parse_axis_range_tokens_with_two_tokens():
    lo, hi = _parse_axis_range_tokens(["30 deg", "40 deg"])

    assert_quantity_allclose(lo, 30 * u.deg)
    assert_quantity_allclose(hi, 40 * u.deg)


def test_parse_axis_range_tokens_raises_for_invalid_token_count():
    with pytest.raises(ValueError, match="exactly two quantities"):
        _parse_axis_range_tokens(["30", "deg", "40"])


def test_normalize_axis_spec_tokens_raises_for_invalid_type():
    with pytest.raises(TypeError, match="strings or lists"):
        _normalize_axis_spec_tokens(42)


def test_parse_axis_spec_raises_for_too_few_tokens():
    with pytest.raises(ValueError, match="at least an axis name"):
        _parse_axis_spec("zenith 2")


def test_parse_axis_spec_raises_for_unknown_axis():
    with pytest.raises(ValueError, match="Unknown axis"):
        _parse_axis_spec("badaxis 0 deg 90 deg 3")


def test_parse_axis_spec_raises_for_missing_range():
    # Only name + binning + scaling => range_tokens is empty
    with pytest.raises(ValueError, match="missing its range"):
        _parse_axis_spec("zenith 3 linear")


def test_parse_axis_spec_raises_for_non_integer_binning():
    with pytest.raises(ValueError, match="binning must be an integer"):
        _parse_axis_spec("zenith 0 deg 90 deg foo")


def test_iter_compact_axis_specs_accepts_string_axis():
    specs = list(_iter_compact_axis_specs({"axis": "azimuth 310 deg 20 deg 3 linear"}))

    assert specs == ["azimuth 310 deg 20 deg 3 linear"]


def test_resolve_coordinate_system_returns_none_without_recognised_axes():
    assert _resolve_coordinate_system({"nsb": {}, "offset": {}}) is None


def test_build_axes_dict_from_cli_args_raises_without_direction_axes():
    with pytest.raises(ValueError, match="azimuth/zenith or both ra/dec"):
        build_axes_dict_from_cli_args(
            {
                "axis": [
                    ["nsb", "4", "MHz", "5", "MHz", "2"],
                    ["offset", "0", "deg", "10", "deg", "2"],
                ]
            }
        )


def test_build_axes_dict_from_cli_args_raises_for_missing_required_axes():
    with pytest.raises(ValueError, match="Missing required shared axis"):
        build_axes_dict_from_cli_args(
            {
                "axis": [
                    ["azimuth", "310", "deg", "20", "deg", "3"],
                    ["zenith", "30", "deg", "40", "deg", "2"],
                ]
            }
        )


@patch("simtools.production_configuration.simulation_jobs._resolve_coordinate_system_from_args")
@patch("simtools.production_configuration.simulation_jobs.build_axes_dict_from_cli_args")
def test_build_production_grid_engine_raises_for_unknown_coordinate_system(
    mock_build_axes, mock_resolve_cs
):
    mock_build_axes.return_value = {}
    mock_resolve_cs.return_value = "unknown"

    with pytest.raises(ValueError, match="azimuth/zenith or both ra/dec"):
        build_production_grid_engine({})


def test_scale_total_showers_raises_for_unknown_mode():
    with pytest.raises(ValueError, match="Unknown total_showers_scaling mode"):
        _scale_total_showers(1000, 20 * u.deg, "bad_mode")


def test_build_rows_for_point_skips_when_effective_total_showers_is_zero():
    rows = _build_rows_for_point(
        point_base={"primary": "gamma", "zenith_angle": 20 * u.deg},
        energy_ranges=[(30 * u.GeV, 100 * u.GeV)],
        lower_energy_threshold=None,
        showers_per_run=1000,
        showers_per_run_power_law=None,
        number_of_runs=1,
        total_showers=0,
        total_showers_scaling="fixed",
        run_number=1,
    )

    assert rows == []
