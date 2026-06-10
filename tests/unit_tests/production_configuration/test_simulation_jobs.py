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
    _clip_energy_range_to_configured_bounds,
    _clip_max_quantity,
    _format_quantity_summary,
    _generate_observation_grids_per_layout,
    _generate_observation_points_from_axes,
    _iter_compact_axis_specs,
    _normalize_axis_spec_tokens,
    _parse_axis_range_tokens,
    _parse_axis_spec,
    _resolve_coordinate_system,
    _resolve_energy_max_scaling,
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
    scale_energy_max_for_zenith_angle,
)


def _base_simulation_jobs_args():
    return {
        "site": "North",
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
        "number_of_runs": 1,
        "run_number": 11,
        "array_layout_name": "alpha",
    }


def _observation_grid_return(point):
    return ({"6.3.0": [point]}, {"6.3.0": "alpha"})


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
                ["offset", "0", "deg", "10", "deg", "2"],
            ]
        }
    )

    assert axes == {
        "offset": {"range": [0.0, 10.0], "binning": 2, "scaling": "linear", "units": "deg"},
        "azimuth": {"range": [310.0, 20.0], "binning": 3, "scaling": "linear", "units": "deg"},
        "zenith_angle": {
            "range": [30.0, 40.0],
            "binning": 2,
            "scaling": "linear",
            "units": "deg",
        },
    }


def test_build_axes_dict_from_cli_args_derives_horizontal_binning_from_density():
    axes = build_axes_dict_from_cli_args(
        {
            "direction_grid_density": 1.0,
            "axis": [
                ["azimuth", "310", "deg", "20", "deg", "3", "linear"],
                ["zenith", "30", "deg", "40", "deg", "2"],
                ["offset", "0", "deg", "10", "deg", "2"],
            ],
        }
    )

    assert axes["azimuth"]["binning"] == 41
    assert axes["zenith_angle"]["binning"] == 10


def test_build_axes_dict_from_cli_args_accepts_density_with_unit():
    axes = build_axes_dict_from_cli_args(
        {
            "direction_grid_density": "1.0 1/deg^2",
            "axis": [
                ["azimuth", "310", "deg", "20", "deg", "3", "linear"],
                ["zenith", "30", "deg", "40", "deg", "2"],
                ["offset", "0", "deg", "10", "deg", "2"],
            ],
        }
    )

    assert axes["azimuth"]["direction_grid_density"] == pytest.approx(1.0)


def test_build_axes_dict_from_cli_args_accepts_density_as_single_token_list():
    axes = build_axes_dict_from_cli_args(
        {
            "direction_grid_density": ["1.0"],
            "axis": [
                ["azimuth", "310", "deg", "20", "deg", "3", "linear"],
                ["zenith", "30", "deg", "40", "deg", "2"],
                ["offset", "0", "deg", "10", "deg", "2"],
            ],
        }
    )

    assert axes["azimuth"]["direction_grid_density"] == pytest.approx(1.0)


def test_build_axes_dict_from_cli_args_derives_radec_binning_from_density():
    axes = build_axes_dict_from_cli_args(
        {
            "direction_grid_density": 1.0,
            "axis": [
                ["ra", "0", "deg", "360", "deg", "36", "linear"],
                ["dec", "-90", "deg", "90", "deg", "18", "linear"],
                ["offset", "0", "deg", "10", "deg", "2", "linear"],
            ],
        }
    )

    assert axes["ra"]["binning"] == 230
    assert axes["dec"]["binning"] == 180
    assert axes["ra"]["direction_grid_density"] == pytest.approx(1.0)


def test_build_axes_dict_from_cli_args_reduces_ra_binning_towards_dec_poles():
    axes = build_axes_dict_from_cli_args(
        {
            "direction_grid_density": 1.0,
            "axis": [
                ["ra", "0", "deg", "360", "deg", "36", "linear"],
                ["dec", "80", "deg", "90", "deg", "10", "linear"],
                ["offset", "0", "deg", "10", "deg", "2", "linear"],
            ],
        }
    )

    assert axes["ra"]["binning"] == 32
    assert axes["dec"]["binning"] == 10
    assert axes["ra"]["direction_grid_density"] == pytest.approx(1.0)


def test_build_axes_dict_from_cli_args_sets_radec_density_metadata_for_adaptive_grid():
    axes = build_axes_dict_from_cli_args(
        {
            "direction_grid_density": 0.5,
            "axis": [
                ["ra", "0", "deg", "360", "deg", "36", "linear"],
                ["dec", "-30", "deg", "30", "deg", "6", "linear"],
                ["offset", "0", "deg", "10", "deg", "2", "linear"],
            ],
        }
    )

    assert axes["ra"]["direction_grid_density"] == pytest.approx(0.5)


def test_build_axes_dict_from_cli_args_keeps_horizontal_constraints_in_radec_mode():
    axes = build_axes_dict_from_cli_args(
        {
            "direction_grid_density": 0.25,
            "local_zenith_range": ["0", "deg", "70", "deg"],
            "local_azimuth_range": ["300", "deg", "60", "deg"],
            "axis": [
                ["ra", "0", "deg", "360", "deg", "36", "linear"],
                ["dec", "-40", "deg", "80", "deg", "10", "linear"],
                ["offset", "0", "deg", "10", "deg", "2", "linear"],
            ],
        }
    )

    assert "ra" in axes
    assert "dec" in axes
    assert "zenith_angle" not in axes
    assert "azimuth" not in axes
    assert axes["ra"]["local_zenith_range"] == pytest.approx([0.0, 70.0])
    assert axes["ra"]["local_azimuth_range"] == pytest.approx([300.0, 60.0])


def test_build_axes_dict_from_cli_args_reduces_az_binning_towards_zenith_pole():
    axes = build_axes_dict_from_cli_args(
        {
            "direction_grid_density": 1.0,
            "axis": [
                ["azimuth", "0", "deg", "180", "deg", "36", "linear"],
                ["zenith", "0", "deg", "10", "deg", "10", "linear"],
                ["offset", "0", "deg", "10", "deg", "2", "linear"],
            ],
        }
    )

    assert axes["azimuth"]["binning"] == 16
    assert axes["zenith_angle"]["binning"] == 10


def test_build_axes_dict_from_cli_args_uses_directed_azimuth_span_for_density():
    axes = build_axes_dict_from_cli_args(
        {
            "direction_grid_density": 1.0,
            "axis": [
                ["azimuth", "0", "deg", "240", "deg", "36", "linear"],
                ["zenith", "0", "deg", "70", "deg", "2", "linear"],
                ["offset", "0", "deg", "10", "deg", "2", "linear"],
            ],
        }
    )

    assert axes["azimuth"]["binning"] == 130
    assert axes["zenith_angle"]["binning"] == 70


def test_build_axes_dict_from_cli_args_sets_horizontal_density_metadata_for_adaptive_grid():
    axes = build_axes_dict_from_cli_args(
        {
            "direction_grid_density": 1.0,
            "axis": [
                ["azimuth", "0", "deg", "240", "deg", "36", "linear"],
                ["zenith", "0", "deg", "70", "deg", "2", "linear"],
                ["offset", "0", "deg", "10", "deg", "2", "linear"],
            ],
        }
    )

    assert axes["azimuth"]["direction_grid_density"] == pytest.approx(1.0)


def test_build_axes_dict_from_cli_args_uses_full_circle_span_for_azimuth_density():
    axes = build_axes_dict_from_cli_args(
        {
            "direction_grid_density": 1.0,
            "axis": [
                ["azimuth", "0", "deg", "360", "deg", "36", "linear"],
                ["zenith", "0", "deg", "70", "deg", "2", "linear"],
                ["offset", "0", "deg", "10", "deg", "2", "linear"],
            ],
        }
    )

    assert axes["azimuth"]["binning"] == 194
    assert axes["zenith_angle"]["binning"] == 70


def test_build_axes_dict_from_cli_args_rejects_non_positive_density():
    with pytest.raises(ValueError, match="direction_grid_density must be strictly positive"):
        build_axes_dict_from_cli_args(
            {
                "direction_grid_density": 0,
                "axis": [
                    ["azimuth", "310", "deg", "20", "deg", "3"],
                    ["zenith", "30", "deg", "40", "deg", "2"],
                    ["offset", "0", "deg", "10", "deg", "2"],
                ],
            }
        )


def test_build_axes_dict_from_cli_args_accepts_config_wrapped_axis_strings():
    axes = build_axes_dict_from_cli_args(
        {
            "axis": [
                ["azimuth 310 deg 20 deg 3 linear"],
                ["zenith 30 deg 40 deg 2"],
                ["offset 0 deg 10 deg 2"],
            ]
        }
    )

    assert axes["azimuth"]["range"] == [310.0, 20.0]
    assert axes["zenith_angle"]["binning"] == 2


def test_build_axes_dict_from_cli_args_accepts_merged_config_axis_list():
    axes = build_axes_dict_from_cli_args(
        {
            "axis": [
                [
                    "azimuth 310 deg 20 deg 3 linear",
                    "zenith 30 deg 40 deg 2",
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
                    ["offset", "0", "deg", "10", "deg", "2"],
                ]
            }
        )


def test_build_axes_dict_from_cli_args_rejects_invalid_density_unit():
    with pytest.raises(ValueError, match="direction_grid_density must be a float or quantity"):
        build_axes_dict_from_cli_args(
            {
                "direction_grid_density": "1.0 1/s",
                "axis": [
                    ["ra", "0", "deg", "360", "deg", "36"],
                    ["dec", "-90", "deg", "90", "deg", "18"],
                    ["offset", "0", "deg", "10", "deg", "2"],
                ],
            }
        )


def test_build_axes_dict_from_cli_args_rejects_invalid_density_type():
    with pytest.raises(TypeError, match="direction_grid_density must be a number"):
        build_axes_dict_from_cli_args(
            {
                "direction_grid_density": {"value": 1.0},
                "axis": [
                    ["ra", "0", "deg", "360", "deg", "36"],
                    ["dec", "-90", "deg", "90", "deg", "18"],
                    ["offset", "0", "deg", "10", "deg", "2"],
                ],
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
            "direction_grid_density": 0.25,
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
    assert metadata["direction_grid_density"] == pytest.approx(0.25)
    assert metadata["direction_grid_density_unit"] == "1/deg^2"
    assert metadata["time_of_observation_utc"].startswith("2017-09-16T00:00:00")
    assert metadata["corsika_limits"] == "limits.ecsv"


def test_build_job_grid_metadata_raises_for_radec_without_site():
    with pytest.raises(ValueError, match="site is required"):
        build_job_grid_metadata(
            {
                "site": None,
                "simulation_software": "corsika_sim_telarray",
                "axis": [
                    ["ra", "0", "deg", "1", "deg", "2"],
                    ["dec", "0", "deg", "1", "deg", "2"],
                ],
                "time_of_observation": "2017-09-16 00:00:00",
                "corsika_limits": "limits.ecsv",
            }
        )


@patch("simtools.production_configuration.simulation_jobs.SiteModel")
def test_build_observing_location_uses_site_model(mock_site_model):
    model = mock_site_model.return_value
    model.get_parameter_value_with_unit.side_effect = [1 * u.deg, 2 * u.deg, 3 * u.m]

    location = build_observing_location("North", ["7.0.0"])

    assert isinstance(location, EarthLocation)
    mock_site_model.assert_called_once_with(model_version="7.0.0", site="North")


@patch("simtools.production_configuration.simulation_jobs.SiteModel")
@patch("simtools.production_configuration.simulation_jobs.ProductionGridEngine")
def test_build_production_grid_engine_resolves_layout_name(
    mock_production_grid_engine,
    mock_site_model,
):
    mock_site_model.return_value.get_nsb_integrated_flux.return_value = 0.42
    args_dict = {
        "site": "North",
        "array_layout_name": {"by_version": {"<7.0.0": "alpha", ">=7.0.0": "beta"}},
        "model_version": ["7.0.0"],
        "time_of_observation": None,
        "corsika_limits": "limits.ecsv",
        "axis": [
            ["azimuth", "310", "deg", "20", "deg", "3", "linear"],
            ["zenith", "30", "deg", "40", "deg", "2", "linear"],
            ["offset", "0", "deg", "10", "deg", "2", "linear"],
        ],
    }

    build_production_grid_engine(args_dict)

    mock_production_grid_engine.assert_called_once_with(
        axes={
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
        lookup_nsb_rate=0.42,
    )
    mock_site_model.assert_called_once_with(model_version="7.0.0", site="North")


@patch("simtools.production_configuration.simulation_jobs.build_observing_location")
@patch("simtools.production_configuration.simulation_jobs.SiteModel")
@patch("simtools.production_configuration.simulation_jobs.ProductionGridEngine")
def test_build_production_grid_engine_builds_observing_location_for_radec(
    mock_production_grid_engine,
    mock_site_model,
    mock_build_observing_location,
):
    location = EarthLocation(lat=1 * u.deg, lon=2 * u.deg, height=3 * u.m)
    mock_build_observing_location.return_value = location
    mock_site_model.return_value.get_nsb_integrated_flux.return_value = 0.24

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
                ["offset", "0", "deg", "10", "deg", "2", "linear"],
            ],
        }
    )

    mock_build_observing_location.assert_called_once_with(site="North", model_version="7.0.0")
    mock_site_model.assert_called_once_with(model_version="7.0.0", site="North")
    assert mock_production_grid_engine.call_args.kwargs["observing_location"] == location


def test_build_production_grid_engine_raises_without_site_for_nsb_rate():
    with pytest.raises(ValueError, match="site and model_version are required"):
        build_production_grid_engine(
            {
                "model_version": ["7.0.0"],
                "time_of_observation": None,
                "corsika_limits": None,
                "axis": [
                    ["azimuth", "310", "deg", "20", "deg", "3", "linear"],
                    ["zenith", "30", "deg", "40", "deg", "2", "linear"],
                    ["offset", "0", "deg", "10", "deg", "2", "linear"],
                ],
            }
        )


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
        "lower_energy_limit": 0.2,
        "upper_radius_limit": 100.0,
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
        "lower_energy_limit": 0.01,
        "upper_radius_limit": 100.0,
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
            "lower_energy_limit": 0.05,
            "upper_radius_limit": 100.0,
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
            "lower_energy_limit": 0.01,
            "upper_radius_limit": 100.0,
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
        "lower_energy_limit": 0.01,
        "upper_radius_limit": 100.0,
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
        "lower_energy_limit": 0.01,
        "upper_radius_limit": 100.0,
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
    assert calculate_zenith_scaled_showers_per_run(60 * u.deg, 1000, "cosine_zenith") == expected


def test_calculate_zenith_scaled_showers_per_run_keeps_baseline_at_zenith_0():
    assert calculate_zenith_scaled_showers_per_run(0 * u.deg, 1000, "cosine_zenith") == 1000


def test_calculate_zenith_scaled_showers_per_run_raises_for_non_positive_baseline():
    with pytest.raises(ValueError, match="positive integer"):
        calculate_zenith_scaled_showers_per_run(20 * u.deg, 0, "cosine_zenith")


def test_calculate_zenith_scaled_showers_per_run_raises_at_zenith_90():
    with pytest.raises(ValueError, match="at least 1"):
        calculate_zenith_scaled_showers_per_run(90 * u.deg, 1000, "cosine_zenith")


def test_calculate_zenith_scaled_showers_per_run_raises_near_zenith_90():
    # Rounding makes this tiny cosine effectively zero, which must trigger validation.
    with pytest.raises(ValueError, match="at least 1"):
        calculate_zenith_scaled_showers_per_run(89.999999999999 * u.deg, 1000, "cosine_zenith")


def test_calculate_zenith_scaled_showers_per_run_raises_for_invalid_mode():
    with pytest.raises(ValueError, match="Unknown showers_per_run_scaling mode"):
        calculate_zenith_scaled_showers_per_run(20 * u.deg, 1000, "invalid_mode")


def test_scale_energy_max_for_zenith_angle_returns_original_without_scaling_index():
    energy_range = (30 * u.GeV, 100 * u.GeV)

    assert scale_energy_max_for_zenith_angle(60 * u.deg, energy_range, None) == energy_range


def test_scale_energy_max_for_zenith_angle_scales_max_energy():
    scaled = scale_energy_max_for_zenith_angle(
        60 * u.deg,
        (30 * u.GeV, 100 * u.GeV),
        (-2.0, 100 * u.GeV),
    )

    assert_quantity_allclose(scaled[0], 30 * u.GeV)
    assert_quantity_allclose(scaled[1], 400 * u.GeV)


def test_scale_energy_max_for_zenith_angle_returns_none_if_scaled_max_below_min():
    assert (
        scale_energy_max_for_zenith_angle(
            60 * u.deg,
            (30 * u.GeV, 100 * u.GeV),
            (2.0, 100 * u.GeV),
        )
        is None
    )


def test_scale_energy_max_for_zenith_angle_raises_for_negative_index_at_zenith_90():
    with pytest.raises(ValueError, match="energy_max_scaling"):
        scale_energy_max_for_zenith_angle(90 * u.deg, (30 * u.GeV, 100 * u.GeV), (-2.0, None))


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


def test_clip_energy_range_to_configured_bounds_caps_selected_max():
    clipped = _clip_energy_range_to_configured_bounds(
        (50 * u.GeV, 300 * u.GeV),
        (30 * u.GeV, 200 * u.GeV),
    )

    assert_quantity_allclose(clipped[0], 50 * u.GeV)
    assert_quantity_allclose(clipped[1], 200 * u.GeV)


def test_clip_max_quantity_returns_configured_value_without_lookup():
    assert _clip_max_quantity(5 * u.deg, None) == 5 * u.deg


def test_format_quantity_summary_returns_single_value_for_constant_series():
    summary = _format_quantity_summary(u.Quantity([200, 200, 200], u.TeV))

    assert summary == "200 TeV"


def test_format_quantity_summary_returns_range_with_explicit_unit():
    summary = _format_quantity_summary(u.Quantity([30 * u.GeV, 1 * u.TeV]))

    assert summary == "[30, 1000] GeV"


def test_resolve_shower_params_converts_showers_per_run_power_law():
    (
        showers_per_run,
        power_law,
        showers_per_run_scaling,
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
    assert showers_per_run_scaling == "fixed"
    assert total_showers is None
    assert total_showers_scaling == "fixed"


def test_resolve_shower_params_accepts_power_law_as_compact_string():
    (
        showers_per_run,
        power_law,
        showers_per_run_scaling,
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
    assert showers_per_run_scaling == "fixed"
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


@pytest.mark.parametrize(
    "energy_max_scaling",
    [
        ["-2.5", "300", "TeV"],
        "-2.5 300 TeV",
        ["-2.5 300 TeV"],
    ],
)
def test_resolve_energy_max_scaling_parses_new_parameter(energy_max_scaling):
    scaling = _resolve_energy_max_scaling({"energy_max_scaling": energy_max_scaling})

    assert scaling[0] == pytest.approx(-2.5)
    assert_quantity_allclose(scaling[1], 300 * u.TeV)


def test_resolve_energy_max_scaling_accepts_legacy_index():
    scaling = _resolve_energy_max_scaling({"energy_max_scaling_index": -2.0})

    assert scaling == (-2.0, None)


def test_resolve_shower_params_accepts_showers_per_run_scaling():
    (
        _showers_per_run,
        _power_law,
        showers_per_run_scaling,
        _total_showers,
        _total_showers_scaling,
    ) = _resolve_shower_params(
        {
            "showers_per_run": 5,
            "showers_per_run_scaling": "cosine_zenith",
        }
    )
    assert showers_per_run_scaling == "cosine_zenith"


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
        showers_per_run_scaling="cosine_zenith",
        number_of_runs=2,
        total_showers=None,
        total_showers_scaling="fixed",
        run_number=1,
    )

    assert [row["showers_per_run"] for row in rows] == [500, 500]


def test_build_rows_for_point_scales_energy_max_with_zenith_and_clips_to_configured_range():
    rows = _build_rows_for_point(
        point_base={"primary": "gamma", "zenith_angle": 60 * u.deg},
        energy_ranges=[(30 * u.GeV, 100 * u.GeV)],
        lower_energy_threshold=None,
        showers_per_run=1000,
        showers_per_run_power_law=None,
        number_of_runs=1,
        total_showers=None,
        total_showers_scaling="fixed",
        run_number=1,
        energy_max_scaling=(-2.0, 100 * u.GeV),
    )

    assert len(rows) == 1
    assert_quantity_allclose(rows[0]["energy_min"], 30 * u.GeV)
    assert_quantity_allclose(rows[0]["energy_max"], 100 * u.GeV)


def test_build_rows_for_point_skips_when_threshold_exceeds_configured_energy_max():
    rows = _build_rows_for_point(
        point_base={"primary": "gamma", "zenith_angle": 60 * u.deg},
        energy_ranges=[(30 * u.GeV, 100 * u.GeV)],
        lower_energy_threshold=150 * u.GeV,
        showers_per_run=1000,
        showers_per_run_power_law=None,
        number_of_runs=1,
        total_showers=None,
        total_showers_scaling="fixed",
        run_number=1,
        energy_max_scaling=(-2.0, 100 * u.GeV),
    )

    assert rows == []


def test_generate_observation_points_from_axes_adds_lookup_limits():
    corsika_limits = Mock()
    corsika_limits.interpolate_point.return_value = {
        "lower_energy_limit": 0.05,
        "upper_radius_limit": 150.0,
        "viewcone_radius": 3.0,
    }
    corsika_limits.lookup_field_units = {
        "lower_energy_limit": u.TeV,
        "upper_radius_limit": u.m,
        "viewcone_radius": u.deg,
    }

    points = _generate_observation_points_from_axes(
        [180 * u.deg],
        [20 * u.deg],
        corsika_limits,
        nsb_rate=0.37,
    )

    assert len(points) == 1
    corsika_limits.interpolate_point.assert_called_once_with(
        20 * u.deg,
        180 * u.deg,
        nsb=0.37,
    )
    assert points[0]["nsb_rate"] == pytest.approx(0.37)
    assert_quantity_allclose(points[0]["lower_energy_limit"], 0.05 * u.TeV)
    assert_quantity_allclose(points[0]["upper_radius_limit"], 150 * u.m)
    assert_quantity_allclose(points[0]["viewcone_radius"], 3 * u.deg)


@patch("simtools.production_configuration.simulation_jobs.SiteModel")
@patch("simtools.production_configuration.simulation_jobs.CorsikaLimitsLookup")
@patch("simtools.production_configuration.simulation_jobs._generate_observation_points_from_axes")
def test_generate_observation_grids_per_layout_uses_layout_specific_lookup(
    mock_generate_observation_points_from_axes,
    mock_corsika_limits_lookup,
    mock_site_model,
):
    mock_generate_observation_points_from_axes.return_value = [{"azimuth": 0 * u.deg}]
    mock_site_model.return_value.get_nsb_integrated_flux.side_effect = [0.2, 0.4]
    args_dict = {
        "site": "North",
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

    assert set(observation_grids) == {"6.3.0", "7.0.0"}
    assert resolved_layout_names == {"6.3.0": "alpha", "7.0.0": "beta"}
    assert mock_site_model.call_count == 2
    mock_corsika_limits_lookup.assert_any_call("limits.ecsv", array_layout_name="alpha")
    mock_corsika_limits_lookup.assert_any_call("limits.ecsv", array_layout_name="beta")
    mock_generate_observation_points_from_axes.assert_any_call(
        azimuth_values=[0 * u.deg],
        zenith_values=[20 * u.deg],
        corsika_limits=mock_corsika_limits_lookup.return_value,
        nsb_rate=0.2,
    )


@patch("simtools.production_configuration.simulation_jobs.SiteModel")
@patch("simtools.production_configuration.simulation_jobs.build_production_grid_engine")
def test_generate_observation_grids_per_layout_uses_shared_axes_and_skips_duplicate_layouts(
    mock_build_production_grid_engine,
    mock_site_model,
):
    mock_build_production_grid_engine.return_value.generate_simulation_grid.return_value = [
        {"azimuth": 0 * u.deg}
    ]
    mock_site_model.return_value.get_nsb_integrated_flux.return_value = 0.31

    observation_grids, resolved_layout_names = _generate_observation_grids_per_layout(
        {
            "site": "North",
            "axis": [
                ["azimuth", "310", "deg", "20", "deg", "3"],
                ["zenith", "20", "deg", "40", "deg", "2"],
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

    assert observation_grids == {
        "6.3.0": [{"azimuth": 0 * u.deg}],
        "7.0.0": [{"azimuth": 0 * u.deg}],
    }
    assert resolved_layout_names == {"6.3.0": "alpha", "7.0.0": "alpha"}
    mock_build_production_grid_engine.assert_called_once()
    assert mock_site_model.call_count == 2


@patch("simtools.production_configuration.simulation_jobs._resolve_nsb_rate")
@patch("simtools.production_configuration.simulation_jobs.build_production_grid_engine")
def test_generate_observation_grids_per_layout_does_not_reuse_different_nsb_rates(
    mock_build_production_grid_engine,
    mock_resolve_nsb_rate,
):
    mock_build_production_grid_engine.return_value.generate_simulation_grid.return_value = [
        {"azimuth": 0 * u.deg}
    ]
    mock_resolve_nsb_rate.side_effect = [0.2, 0.4]

    _generate_observation_grids_per_layout(
        {
            "site": "North",
            "axis": [
                ["azimuth", "310", "deg", "20", "deg", "3"],
                ["zenith", "20", "deg", "40", "deg", "2"],
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

    assert mock_build_production_grid_engine.call_count == 2
    called_model_versions = [
        call.kwargs["model_version"] for call in mock_build_production_grid_engine.call_args_list
    ]
    assert called_model_versions == ["6.3.0", "7.0.0"]


@patch("simtools.production_configuration.simulation_jobs._generate_observation_grids_per_layout")
def test_build_simulation_jobs_expands_runs_from_observation_grid(
    mock_generate_observation_grids_per_layout,
):
    mock_generate_observation_grids_per_layout.return_value = _observation_grid_return(
        {
            "azimuth": 180 * u.deg,
            "zenith_angle": 20 * u.deg,
            "ra": 123 * u.deg,
            "dec": -45 * u.deg,
            "lower_energy_limit": 40 * u.GeV,
            "upper_radius_limit": 100 * u.m,
            "viewcone_radius": 2 * u.deg,
        }
    )
    args_dict = _base_simulation_jobs_args()
    args_dict["number_of_runs"] = 2
    rows = build_simulation_jobs(args_dict)

    assert [row["run_number"] for row in rows] == [11, 12]
    assert rows[0]["array_layout_name"] == "alpha"
    assert rows[0]["core_scatter_max"] == 100 * u.m
    assert rows[0]["view_cone_min"] == 0 * u.deg
    assert rows[0]["view_cone_max"] == 2 * u.deg
    assert rows[0]["showers_per_run"] == 5
    assert rows[0]["ra"] == 123 * u.deg
    assert rows[0]["dec"] == -45 * u.deg


@patch("simtools.production_configuration.simulation_jobs.SiteModel")
@patch("simtools.production_configuration.simulation_jobs._generate_observation_grids_per_layout")
def test_build_simulation_jobs_sets_nsb_rate_from_site_model(
    mock_generate_observation_grids_per_layout,
    mock_site_model,
):
    mock_generate_observation_grids_per_layout.return_value = _observation_grid_return(
        {
            "azimuth": 180 * u.deg,
            "zenith_angle": 20 * u.deg,
            "lower_energy_limit": 40 * u.GeV,
            "upper_radius_limit": 100 * u.m,
            "viewcone_radius": 2 * u.deg,
        }
    )
    mock_site_model.return_value.get_nsb_integrated_flux.return_value = 0.37

    args_dict = _base_simulation_jobs_args()
    args_dict["site"] = "North"

    rows = build_simulation_jobs(args_dict)

    assert rows[0]["nsb_rate"] == pytest.approx(0.37)
    mock_site_model.assert_called_once_with(model_version="6.3.0", site="North")


@patch("simtools.production_configuration.simulation_jobs._generate_observation_grids_per_layout")
def test_build_simulation_jobs_clips_core_and_viewcone_max_by_configured_limits(
    mock_generate_observation_grids_per_layout,
):
    mock_generate_observation_grids_per_layout.return_value = _observation_grid_return(
        {
            "azimuth": 180 * u.deg,
            "zenith_angle": 20 * u.deg,
            "lower_energy_limit": 40 * u.GeV,
            "upper_radius_limit": 400 * u.m,
            "viewcone_radius": 10 * u.deg,
        }
    )
    args_dict = _base_simulation_jobs_args()
    args_dict["view_cone"] = [3 * u.deg, 5 * u.deg]
    rows = build_simulation_jobs(args_dict)

    assert rows[0]["core_scatter_max"] == 200 * u.m
    assert rows[0]["view_cone_min"] == 3 * u.deg
    assert rows[0]["view_cone_max"] == 5 * u.deg


@patch("simtools.production_configuration.simulation_jobs._generate_observation_grids_per_layout")
def test_build_simulation_jobs_uses_interpolated_energy_min_when_threshold_key_missing(
    mock_generate_observation_grids_per_layout,
):
    mock_generate_observation_grids_per_layout.return_value = _observation_grid_return(
        {
            "azimuth": 180 * u.deg,
            "zenith_angle": 20 * u.deg,
            "br_energy_min": 50 * u.GeV,
            "upper_radius_limit": 100 * u.m,
            "viewcone_radius": 2 * u.deg,
        }
    )
    rows = build_simulation_jobs(_base_simulation_jobs_args())

    assert len(rows) == 1
    assert rows[0]["energy_min"] == 50 * u.GeV
    assert rows[0]["energy_max"] == 100 * u.GeV


@patch("simtools.production_configuration.simulation_jobs._generate_observation_grids_per_layout")
def test_build_simulation_jobs_clips_viewcone_min_to_lookup_limited_max(
    mock_generate_observation_grids_per_layout,
):
    mock_generate_observation_grids_per_layout.return_value = _observation_grid_return(
        {
            "azimuth": 180 * u.deg,
            "zenith_angle": 20 * u.deg,
            "lower_energy_limit": 40 * u.GeV,
            "upper_radius_limit": 100 * u.m,
            "viewcone_radius": 2 * u.deg,
        }
    )
    args_dict = _base_simulation_jobs_args()
    args_dict["view_cone"] = [3 * u.deg, 5 * u.deg]
    rows = build_simulation_jobs(args_dict)

    assert rows[0]["view_cone_min"] == 2 * u.deg
    assert rows[0]["view_cone_max"] == 2 * u.deg


@patch("simtools.production_configuration.simulation_jobs._generate_observation_grids_per_layout")
def test_build_simulation_jobs_raises_for_total_showers_and_number_of_runs(
    mock_generate_observation_grids_per_layout,
):
    mock_generate_observation_grids_per_layout.return_value = ({"alpha": []}, {"6.3.0": "alpha"})
    args_dict = _base_simulation_jobs_args()
    args_dict["number_of_runs"] = 2
    args_dict["total_showers"] = 100

    with pytest.raises(ValueError, match="total_showers and number_of_runs"):
        build_simulation_jobs(args_dict)


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
    assert _resolve_coordinate_system({"offset": {}}) is None


def test_build_axes_dict_from_cli_args_raises_without_direction_axes():
    with pytest.raises(ValueError, match="azimuth/zenith or both ra/dec"):
        build_axes_dict_from_cli_args(
            {
                "axis": [
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
