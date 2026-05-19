import numpy as np
import pytest
from astropy import units as u

from simtools.production_configuration.corsika_limits_lookup import CorsikaLimitsLookup


def test_load_matching_lookup_arrays_filters_by_array_layout_name():
    lookup = CorsikaLimitsLookup(
        "tests/resources/corsika_simulation_limits/merged_corsika_limits_for_test.ecsv",
        array_layout_name="1mst",
    )

    arrays = lookup.load_matching_lookup_arrays()

    assert len(arrays["points"]) == 7
    assert arrays["lower_energy_threshold"][0] == pytest.approx(0.022)


def test_load_matching_lookup_arrays_raises_for_unknown_array_layout():
    lookup = CorsikaLimitsLookup(
        "tests/resources/corsika_simulation_limits/merged_corsika_limits_for_test.ecsv",
        array_layout_name="does-not-exist",
    )

    with pytest.raises(ValueError, match="array_layout_name"):
        lookup.load_matching_lookup_arrays()


def test_load_matching_lookup_arrays_without_layout_returns_all_rows():
    lookup = CorsikaLimitsLookup(
        "tests/resources/corsika_simulation_limits/merged_corsika_limits_for_test.ecsv"
    )

    arrays = lookup.load_matching_lookup_arrays()

    assert len(arrays["points"]) == 14


def test_prepare_point_interpolators_builds_interpolator_state():
    lookup = CorsikaLimitsLookup(
        "tests/resources/corsika_simulation_limits/merged_corsika_limits_for_test.ecsv",
        array_layout_name="1mst",
    )

    interpolators = lookup.prepare_point_interpolators()

    assert lookup.lookup_points_for_interpolation.shape[1] == 3
    assert set(interpolators) == {
        "lower_energy_threshold",
        "upper_scatter_radius",
        "viewcone_radius",
    }


def test_interpolate_grid_limits_returns_requested_grid_shape():
    lookup = CorsikaLimitsLookup(
        "tests/resources/corsika_simulation_limits/merged_corsika_limits_for_test.ecsv",
        array_layout_name="1mst",
    )

    interpolated = lookup.interpolate_grid_limits(
        {
            "zenith_angle": [20, 40] * u.deg,
            "azimuth": [0, 180] * u.deg,
            "nsb_level": [1, 5] * u.dimensionless_unscaled,
        }
    )

    assert interpolated["lower_energy_threshold"].shape == (2, 2, 2)
    assert np.isfinite(interpolated["upper_scatter_radius"][0, 0, 0])


def test_interpolate_point_returns_interpolated_values():
    lookup = CorsikaLimitsLookup(
        "tests/resources/corsika_simulation_limits/merged_corsika_limits_for_test.ecsv",
        array_layout_name="1mst",
    )

    interpolated = lookup.interpolate_point(20 * u.deg, 0 * u.deg, nsb=1)

    assert interpolated["lower_energy_threshold"] == pytest.approx(0.022)
    assert interpolated["upper_scatter_radius"] == pytest.approx(1000.0)
    assert interpolated["viewcone_radius"] == pytest.approx(9.25)
