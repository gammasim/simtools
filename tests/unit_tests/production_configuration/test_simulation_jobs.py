from unittest.mock import Mock, patch

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
    _resolve_nshow_params,
    build_job_grid_metadata,
    build_observing_location,
    build_production_grid_engine,
    build_simulation_jobs,
    calculate_log_energy_midpoint,
    calculate_scaled_nshow,
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
    args_dict = {"azimuth_range": [0, 1], "zenith_range": [0, 1]}
    assert resolve_time_of_observation(None, args_dict) is None


def test_resolve_time_of_observation_raises_for_radec_without_input():
    args_dict = {"ra_range": [0, 1], "dec_range": [0, 1]}
    with pytest.raises(ValueError, match="time_of_observation"):
        resolve_time_of_observation(None, args_dict)


def test_build_job_grid_metadata_includes_job_context():
    metadata = build_job_grid_metadata(
        {
            "site": "North",
            "simulation_software": "corsika_sim_telarray",
            "ra_range": [0, 1],
            "dec_range": [0, 1],
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
        "azimuth_range": [310 * u.deg, 20 * u.deg],
        "azimuth_binning": 3,
        "azimuth_scaling": "linear",
        "zenith_range": [30 * u.deg, 40 * u.deg],
        "zenith_binning": 2,
        "zenith_scaling": "linear",
        "nsb_range": [4 * u.MHz, 5 * u.MHz],
        "nsb_binning": 2,
        "nsb_scaling": "linear",
        "offset_range": [0 * u.deg, 10 * u.deg],
        "offset_binning": 2,
        "offset_scaling": "linear",
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
            "ra_range": [0 * u.deg, 360 * u.deg],
            "ra_binning": 36,
            "ra_scaling": "linear",
            "dec_range": [-90 * u.deg, 90 * u.deg],
            "dec_binning": 18,
            "dec_scaling": "linear",
            "nsb_range": [4 * u.MHz, 4 * u.MHz],
            "nsb_binning": 1,
            "nsb_scaling": "linear",
            "offset_range": [0 * u.deg, 10 * u.deg],
            "offset_binning": 2,
            "offset_scaling": "linear",
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
        "tests/resources/corsika_simulation_limits/merged_corsika_limits_for_test.ecsv"
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
        "tests/resources/corsika_simulation_limits/merged_corsika_limits_for_test.ecsv"
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


def test_calculate_scaled_nshow_returns_baseline_without_power_index():
    assert calculate_scaled_nshow((30 * u.GeV, 100 * u.GeV), 5) == 5


def test_calculate_scaled_nshow_raises_for_non_positive_baseline():
    with pytest.raises(ValueError, match="positive integer"):
        calculate_scaled_nshow((30 * u.GeV, 100 * u.GeV), 0)


def test_calculate_scaled_nshow_raises_for_missing_reference_energy():
    with pytest.raises(ValueError, match="reference_energy"):
        calculate_scaled_nshow((30 * u.GeV, 100 * u.GeV), 5, nshow_power_index=1.0)


def test_calculate_scaled_nshow_scales_from_midpoint_energy():
    scaled_nshow = calculate_scaled_nshow(
        (10 * u.GeV, 1 * u.TeV),
        5,
        nshow_power_index=1.0,
        reference_energy=10 * u.GeV,
    )

    assert scaled_nshow == 50


@patch("simtools.production_configuration.simulation_jobs.np.ceil", return_value=0)
def test_calculate_scaled_nshow_raises_when_scaled_value_is_below_one(mock_ceil):
    with pytest.raises(ValueError, match="at least 1"):
        calculate_scaled_nshow(
            (10 * u.GeV, 1 * u.TeV),
            5,
            nshow_power_index=1.0,
            reference_energy=10 * u.GeV,
        )
    mock_ceil.assert_called_once()


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


def test_resolve_nshow_params_converts_reference_energy():
    nshow, power_index, reference_energy = _resolve_nshow_params(
        {"nshow": 5, "nshow_power_index": 1.0, "nshow_reference_energy": "100 GeV"}
    )

    assert nshow == 5
    assert power_index == pytest.approx(1.0)
    assert reference_energy == 100 * u.GeV


def test_build_rows_for_point_skips_energy_ranges_below_threshold():
    rows = _build_rows_for_point(
        point_base={"primary": "gamma"},
        energy_ranges=[(30 * u.GeV, 40 * u.GeV), (50 * u.GeV, 100 * u.GeV)],
        lower_energy_threshold=45 * u.GeV,
        nshow=5,
        nshow_power_index=None,
        reference_energy=None,
        number_of_runs=2,
        run_number=10,
    )

    assert [row["run_number"] for row in rows] == [10, 11]
    assert all(row["energy_min"] == 50 * u.GeV for row in rows)


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

    observation_grids = _generate_observation_grids_per_layout(args_dict, grid_axes)

    assert set(observation_grids) == {"alpha", "beta"}
    mock_corsika_limits_lookup.assert_any_call("limits.ecsv", array_layout_name="alpha")
    mock_corsika_limits_lookup.assert_any_call("limits.ecsv", array_layout_name="beta")


@patch("simtools.production_configuration.simulation_jobs.build_production_grid_engine")
def test_generate_observation_grids_per_layout_uses_shared_axes_and_skips_duplicate_layouts(
    mock_build_production_grid_engine,
):
    mock_build_production_grid_engine.return_value.generate_simulation_grid.return_value = [
        {"azimuth": 0 * u.deg}
    ]

    observation_grids = _generate_observation_grids_per_layout(
        {
            "azimuth_range": [310 * u.deg, 20 * u.deg],
            "array_layout_name": {"by_version": {"<7.0.0": "alpha", ">=7.0.0": "alpha"}},
        },
        {
            "model_version": ["6.3.0", "7.0.0"],
            "azimuth_angle": [0 * u.deg],
            "zenith_angle": [20 * u.deg],
        },
    )

    assert observation_grids == {"alpha": [{"azimuth": 0 * u.deg}]}
    mock_build_production_grid_engine.assert_called_once()


@patch("simtools.production_configuration.simulation_jobs._generate_observation_grids_per_layout")
def test_build_simulation_jobs_expands_runs_from_observation_grid(
    mock_generate_observation_grids_per_layout,
):
    mock_generate_observation_grids_per_layout.return_value = {
        "alpha": [
            {
                "azimuth": 180 * u.deg,
                "zenith_angle": 20 * u.deg,
                "lower_energy_threshold": 40 * u.GeV,
                "scatter_radius": 100 * u.m,
                "viewcone_radius": 2 * u.deg,
            }
        ]
    }
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
            "nshow": 5,
            "number_of_runs": 2,
            "run_number": 11,
            "array_layout_name": "alpha",
        }
    )

    assert [row["run_number"] for row in rows] == [11, 12]
    assert rows[0]["array_layout_name"] == "alpha"
    assert rows[0]["core_scatter_max"] == 100 * u.m
    assert rows[0]["view_cone_max"] == 2 * u.deg
