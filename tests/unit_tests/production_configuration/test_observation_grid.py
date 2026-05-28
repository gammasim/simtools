from unittest.mock import Mock, patch

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.tests.helper import assert_quantity_allclose
from astropy.time import Time

from simtools.production_configuration.observation_grid import ProductionGridEngine


def test_generate_simulation_grid_keeps_horizontal_coordinates_for_radec_axes():
    axes = {
        "axes": {
            "ra": {"range": [0, 0], "binning": 1, "scaling": "linear", "units": "deg"},
            "dec": {"range": [0, 0], "binning": 1, "scaling": "linear", "units": "deg"},
        }
    }
    engine = ProductionGridEngine(
        axes=axes,
        coordinate_system="ra_dec",
        observing_location=EarthLocation(lat=28.76 * u.deg, lon=-17.89 * u.deg, height=2200 * u.m),
        time_of_observation=Time("2017-09-16 00:00:00", scale="utc"),
        lookup_table=None,
    )

    simulation_grid = engine.generate_simulation_grid()

    assert "zenith_angle" in simulation_grid[0]
    assert "azimuth" in simulation_grid[0]
    assert "ra" in simulation_grid[0]
    assert "dec" in simulation_grid[0]


def test_require_time_of_observation_raises_without_time():
    engine = ProductionGridEngine(axes={"axes": {}}, time_of_observation=None)

    with pytest.raises(ValueError, match="Observing time"):
        engine._require_time_of_observation()


def test_get_max_zenith_for_radec_mode_reads_axis_range():
    engine = ProductionGridEngine(
        axes={"axes": {"zenith_angle": {"range": [10, 60], "binning": 2, "units": "deg"}}}
    )

    assert engine._get_max_zenith_for_radec_mode() == 60


def test_get_max_zenith_for_radec_mode_raises_for_invalid_axis_definition():
    engine = ProductionGridEngine(axes={"axes": {}})

    with pytest.raises(ValueError, match="valid two-element 'range'"):
        engine._get_max_zenith_for_radec_mode()


def test_generate_target_values_supports_log_and_inverse_cos_scaling():
    engine = ProductionGridEngine(
        axes={
            "axes": {
                "energy": {"range": [1, 100], "binning": 3, "scaling": "log", "units": "TeV"},
                "zenith_angle": {
                    "range": [0, 60],
                    "binning": 3,
                    "scaling": "1/cos",
                    "units": "deg",
                },
            }
        }
    )

    assert_quantity_allclose(engine.target_values["energy"], [1, 10, 100] * u.TeV)
    assert len(engine.target_values["zenith_angle"]) == 3


def test_add_lookup_limits_to_point_uses_quantity_nsb_level():
    engine = ProductionGridEngine(axes={"axes": {}}, lookup_table=None)
    engine.lookup_table = "limits.ecsv"
    engine._interpolate_limits_for_point = Mock(
        return_value={
            "lower_energy_threshold": 0.02,
            "upper_scatter_radius": 120.0,
            "viewcone_radius": 3.5,
        }
    )
    point = {"nsb_level": 5 * u.dimensionless_unscaled}

    engine._add_lookup_limits_to_point(point, zenith=20.0, azimuth=180.0)

    engine._interpolate_limits_for_point.assert_called_once_with(
        zenith=20.0, azimuth=180.0, nsb=5.0
    )
    assert_quantity_allclose(point["lower_energy_threshold"], 0.02 * u.TeV)
    assert_quantity_allclose(point["scatter_radius"], 120 * u.m)
    assert_quantity_allclose(point["viewcone_radius"], 3.5 * u.deg)


def test_interpolate_limits_for_point_delegates_to_lookup_helper():
    engine = ProductionGridEngine(axes={"axes": {}}, lookup_table=None)
    engine._limits_lookup = Mock()
    engine._limits_lookup.interpolate_point.return_value = {"lower_energy_threshold": 0.02}

    limits = engine._interpolate_limits_for_point(20.0, 180.0, 1.0)

    assert limits == {"lower_energy_threshold": 0.02}
    engine._limits_lookup.interpolate_point.assert_called_once_with(20.0, 180.0, 1.0)


def test_prepare_lookup_table_limits_for_point_interpolation_prepares():
    engine = ProductionGridEngine(axes={"axes": {}}, lookup_table=None)
    engine._limits_lookup = Mock()

    engine._prepare_lookup_table_limits_for_point_interpolation()

    engine._limits_lookup.prepare_point_interpolators.assert_called_once_with()


def test_generate_extra_axis_combinations_returns_empty_placeholder_for_no_extra_axes():
    engine = ProductionGridEngine(
        axes={"axes": {"zenith_angle": {"range": [20, 20], "binning": 1, "units": "deg"}}}
    )

    keys, units, combinations = engine._generate_extra_axis_combinations(("zenith_angle",))

    assert keys == []
    assert units == []
    assert len(combinations) == 1


def test_generate_extra_axis_combinations_returns_mesh_for_remaining_axes():
    engine = ProductionGridEngine(
        axes={
            "axes": {
                "zenith_angle": {"range": [20, 20], "binning": 1, "units": "deg"},
                "nsb_level": {"range": [1, 2], "binning": 2, "units": "1"},
                "offset": {"range": [0, 1], "binning": 2, "units": "deg"},
            }
        }
    )

    keys, units, combinations = engine._generate_extra_axis_combinations(("zenith_angle",))

    assert keys == ["nsb_level", "offset"]
    assert units == [u.dimensionless_unscaled, u.deg]
    assert combinations.shape == (4, 2)


def test_create_circular_binning_uses_directed_span_from_start_to_end():
    engine = ProductionGridEngine(axes={"axes": {}})

    binning = engine.create_circular_binning((10, 350), 3)

    assert np.allclose(binning, [10, 180, 350])


def test_create_circular_binning_uses_clockwise_path_when_shorter():
    engine = ProductionGridEngine(axes={"axes": {}})

    binning = engine.create_circular_binning((350, 10), 3)

    assert np.allclose(binning, [350, 0, 10])


def test_create_circular_binning_covers_requested_range_0_to_240():
    engine = ProductionGridEngine(axes={"axes": {}})

    binning = engine.create_circular_binning((0, 240), 3)

    assert np.allclose(binning, [0, 120, 240])


def test_generate_horizontal_grid_uses_adaptive_azimuth_bins_per_zenith_row():
    engine = ProductionGridEngine(
        axes={
            "axes": {
                "zenith_angle": {"range": [0, 60], "binning": 61, "units": "deg"},
                "azimuth": {
                    "range": [0, 240],
                    "binning": 10,
                    "units": "deg",
                    "direction_grid_density": 0.05,
                },
                "nsb_level": {"range": [1, 1], "binning": 1, "units": "1"},
            }
        },
        coordinate_system="horizontal",
    )

    grid = engine._generate_horizontal_grid()

    assert len(grid) > 0

    azimuth_counts_by_zenith = {}
    for point in grid:
        zenith_value = round(point["zenith_angle"].to_value(u.deg), 6)
        azimuth_counts_by_zenith.setdefault(zenith_value, set()).add(
            round(point["azimuth"].to_value(u.deg), 6)
        )

    assert len(azimuth_counts_by_zenith[0.0]) == 1
    assert len(azimuth_counts_by_zenith[30.0]) > len(azimuth_counts_by_zenith[0.0])
    assert len(azimuth_counts_by_zenith[60.0]) > len(azimuth_counts_by_zenith[30.0])


def test_create_circular_binning_treats_full_circle_range_as_full_span():
    engine = ProductionGridEngine(axes={"axes": {}})

    binning = engine.create_circular_binning((0, 360), 4)

    assert np.allclose(binning, [0, 90, 180, 270])


def test_convert_altaz_to_radec_raises_without_time_of_observation():
    engine = ProductionGridEngine(axes={"axes": {}}, time_of_observation=None)

    with pytest.raises(ValueError, match="time_of_observation"):
        engine.convert_altaz_to_radec(45 * u.deg, 180 * u.deg)


def test_convert_altaz_to_radec_returns_icrs_coordinates():
    engine = ProductionGridEngine(
        axes={"axes": {}},
        observing_location=EarthLocation(lat=28.76 * u.deg, lon=-17.89 * u.deg, height=2200 * u.m),
        time_of_observation=Time("2017-09-16 00:00:00", scale="utc"),
    )

    radec = engine.convert_altaz_to_radec(45 * u.deg, 180 * u.deg)

    assert isinstance(radec, SkyCoord)
    assert radec.frame.name == "icrs"


def test_convert_coordinates_drops_horizontal_coordinates_when_requested():
    engine = ProductionGridEngine(axes={"axes": {}}, coordinate_system="ra_dec")
    engine.convert_altaz_to_radec = Mock(return_value=Mock(ra=Mock(deg=10), dec=Mock(deg=-20)))
    points = [{"zenith_angle": 20 * u.deg, "azimuth": 180 * u.deg}]

    converted = engine.convert_coordinates(points, keep_horizontal_coordinates=False)

    assert "zenith_angle" not in converted[0]
    assert "azimuth" not in converted[0]
    assert_quantity_allclose(converted[0]["ra"], 10 * u.deg)
    assert_quantity_allclose(converted[0]["dec"], -20 * u.deg)


def test_convert_coordinates_keeps_points_without_horizontal_values():
    engine = ProductionGridEngine(axes={"axes": {}}, coordinate_system="ra_dec")
    points = [{"ra": 10 * u.deg, "dec": -20 * u.deg}]

    converted = engine.convert_coordinates(points, keep_horizontal_coordinates=False)

    assert converted == points


@patch.object(ProductionGridEngine, "_prepare_lookup_table_limits_for_point_interpolation")
@patch("simtools.production_configuration.observation_grid.CorsikaLimitsLookup")
def test_init_with_radec_lookup_prepares_point_interpolation(
    mock_corsika_limits_lookup,
    mock_prepare_lookup_table_limits,
):
    ProductionGridEngine(
        axes={"axes": {"ra": {"range": [0, 0], "binning": 1, "units": "deg"}}},
        coordinate_system="ra_dec",
        time_of_observation=Time("2017-09-16 00:00:00", scale="utc"),
        lookup_table="limits.ecsv",
        array_layout_name="alpha",
    )

    mock_corsika_limits_lookup.assert_called_once_with(
        lookup_table="limits.ecsv",
        array_layout_name="alpha",
    )
    mock_prepare_lookup_table_limits.assert_called_once_with()


@patch("simtools.production_configuration.observation_grid.CorsikaLimitsLookup")
def test_init_with_horizontal_lookup_applies_interpolated_limits(mock_corsika_limits_lookup):
    mock_lookup = mock_corsika_limits_lookup.return_value
    mock_lookup.interpolate_grid_limits.return_value = {
        "lower_energy_threshold": np.array([[[0.02]]])
    }

    engine = ProductionGridEngine(
        axes={"axes": {"zenith_angle": {"range": [20, 20], "binning": 1, "units": "deg"}}},
        lookup_table="limits.ecsv",
        array_layout_name="alpha",
    )

    assert "lower_energy_threshold" in engine.interpolated_limits


def test_generate_horizontal_grid_uses_interpolated_limit_arrays():
    engine = ProductionGridEngine(
        axes={
            "axes": {
                "zenith_angle": {"range": [20, 20], "binning": 1, "units": "deg"},
                "azimuth": {"range": [0, 0], "binning": 1, "units": "deg"},
                "nsb_level": {"range": [1, 1], "binning": 1, "units": "1"},
            }
        }
    )
    engine.interpolated_limits = {
        "lower_energy_threshold": np.array([[[0.02]]]),
        "upper_scatter_radius": np.array([[[120.0]]]),
        "viewcone_radius": np.array([[[3.0]]]),
    }

    grid = engine._generate_horizontal_grid()

    assert_quantity_allclose(grid[0]["lower_energy_threshold"], 0.02 * u.TeV)
    assert_quantity_allclose(grid[0]["scatter_radius"], 120 * u.m)
    assert_quantity_allclose(grid[0]["viewcone_radius"], 3 * u.deg)


@patch("simtools.production_configuration.observation_grid.AltAz")
@patch("simtools.production_configuration.observation_grid.SkyCoord")
@patch("simtools.production_configuration.observation_grid.np.arange", return_value=np.array([0.0]))
def test_generate_radec_grid_direction_points_filters_by_max_zenith(
    mock_arange, mock_skycoord, mock_altaz
):
    engine = ProductionGridEngine(
        axes={"axes": {"zenith_angle": {"range": [0, 20], "binning": 2, "units": "deg"}}},
        observing_location=EarthLocation(lat=28.76 * u.deg, lon=-17.89 * u.deg, height=2200 * u.m),
        time_of_observation=Mock(),
    )
    engine.time_of_observation.sidereal_time.return_value = Mock(deg=30.0)
    mock_skycoord.return_value.transform_to.return_value = Mock(
        alt=np.array([80.0, 10.0]) * u.deg,
        az=Mock(deg=np.array([100.0, 200.0])),
    )

    direction_points = engine._generate_radec_grid_direction_points()

    assert len(direction_points) == 1
    assert_quantity_allclose(direction_points[0]["zenith_angle"], 10 * u.deg)
    assert_quantity_allclose(direction_points[0]["azimuth"], 100 * u.deg)
    mock_arange.assert_called_once()
    mock_altaz.assert_called_once()


def test_generate_horizontal_grid_handles_partial_interpolated_limit_arrays():
    engine = ProductionGridEngine(
        axes={
            "axes": {
                "zenith_angle": {"range": [20, 20], "binning": 1, "units": "deg"},
                "azimuth": {"range": [0, 0], "binning": 1, "units": "deg"},
                "nsb_level": {"range": [1, 1], "binning": 1, "units": "1"},
            }
        }
    )
    engine.interpolated_limits = {
        "upper_scatter_radius": np.array([[[120.0]]]),
        "viewcone_radius": np.array([[[3.0]]]),
    }

    grid = engine._generate_horizontal_grid()

    assert "lower_energy_threshold" not in grid[0]
    assert_quantity_allclose(grid[0]["scatter_radius"], 120 * u.m)
    assert_quantity_allclose(grid[0]["viewcone_radius"], 3 * u.deg)


def test_generate_horizontal_grid_with_circular_azimuth_binning_uses_correct_indices():
    engine = ProductionGridEngine(
        axes={
            "axes": {
                "zenith_angle": {"range": [20, 20], "binning": 1, "units": "deg"},
                "azimuth": {"range": [10, 350], "binning": 3, "units": "deg"},
                "nsb_level": {"range": [1, 1], "binning": 1, "units": "1"},
            }
        }
    )
    # Directed span from start to end gives [10, 180, 350]
    assert np.allclose(engine.target_values["azimuth"].value, np.array([10.0, 180.0, 350.0]))

    engine.interpolated_limits = {
        "lower_energy_threshold": np.array([[[0.1], [0.2], [0.3]]]),
    }

    grid = engine._generate_horizontal_grid()
    by_azimuth = {
        float(point["azimuth"].to_value(u.deg)): point["lower_energy_threshold"].to_value(u.TeV)
        for point in grid
    }

    assert by_azimuth[10.0] == pytest.approx(0.1)
    assert by_azimuth[180.0] == pytest.approx(0.2)
    assert by_azimuth[350.0] == pytest.approx(0.3)


def test_generate_grid_radec_mode_adds_extra_axis_quantities():
    engine = ProductionGridEngine(
        axes={"axes": {"zenith_angle": {"range": [20, 20], "binning": 1, "units": "deg"}}}
    )
    engine.coordinate_system = "ra_dec"
    engine._generate_radec_grid_direction_points = Mock(
        return_value=[{"zenith_angle": 20 * u.deg, "azimuth": 180 * u.deg}]
    )
    engine._generate_extra_axis_combinations = Mock(
        return_value=(["nsb_level"], [u.dimensionless_unscaled], [np.array([5.0])])
    )
    engine._add_lookup_limits_to_point = Mock()
    engine.convert_coordinates = Mock(return_value=[{"ra": 10 * u.deg, "dec": -20 * u.deg}])

    grid = engine._generate_grid_radec_mode()

    assert grid == [{"ra": 10 * u.deg, "dec": -20 * u.deg}]
    added_point = engine.convert_coordinates.call_args.args[0][0]
    assert_quantity_allclose(added_point["nsb_level"], 5 * u.dimensionless_unscaled)


def test_generate_grid_radec_mode_uses_direction_points_when_ra_dec_axes_missing():
    engine = ProductionGridEngine(
        axes={"axes": {"zenith_angle": {"range": [20, 20], "binning": 1}}}
    )
    engine.coordinate_system = "ra_dec"
    engine._generate_radec_grid_direction_points = Mock(
        return_value=[{"zenith_angle": 20 * u.deg, "azimuth": 180 * u.deg}]
    )
    engine._generate_extra_axis_combinations = Mock(return_value=([], [], [np.array([])]))
    engine.convert_coordinates = Mock(return_value=[{"ra": 10 * u.deg, "dec": -20 * u.deg}])

    grid = engine._generate_grid_radec_mode()

    assert grid == [{"ra": 10 * u.deg, "dec": -20 * u.deg}]
    engine.convert_coordinates.assert_called_once()


def test_generate_simulation_grid_uses_horizontal_grid_for_horizontal_mode():
    engine = ProductionGridEngine(axes={"axes": {}})
    engine.coordinate_system = "horizontal"
    engine._generate_horizontal_grid = Mock(return_value=[{"azimuth": 0 * u.deg}])

    grid = engine.generate_simulation_grid()

    assert grid == [{"azimuth": 0 * u.deg}]
