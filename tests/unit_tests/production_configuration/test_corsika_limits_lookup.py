import pytest
from astropy import units as u
from astropy.table import Table

from simtools.production_configuration.corsika_limits_lookup import CorsikaLimitsLookup

ARRAY_LAYOUT_NAME = "CTAO-North-LSTs"


def test_load_matching_lookup_arrays_filters_by_array_layout_name(corsika_limits_for_test_file):
    lookup = CorsikaLimitsLookup(
        corsika_limits_for_test_file,
        array_layout_name=ARRAY_LAYOUT_NAME,
    )

    arrays = lookup.load_matching_lookup_arrays()

    assert len(arrays["points"]) == 16
    assert arrays["lower_energy_threshold"][0] == pytest.approx(0.003)


def test_load_matching_lookup_arrays_raises_for_unknown_array_layout(corsika_limits_for_test_file):
    lookup = CorsikaLimitsLookup(
        corsika_limits_for_test_file,
        array_layout_name="does-not-exist",
    )

    with pytest.raises(ValueError, match="array_layout_name"):
        lookup.load_matching_lookup_arrays()


def test_load_matching_lookup_arrays_without_layout_returns_all_rows(corsika_limits_for_test_file):
    lookup = CorsikaLimitsLookup(corsika_limits_for_test_file)

    arrays = lookup.load_matching_lookup_arrays()

    assert len(arrays["points"]) == 96


def test_prepare_point_interpolators_builds_interpolator_state(corsika_limits_for_test_file):
    lookup = CorsikaLimitsLookup(
        corsika_limits_for_test_file,
        array_layout_name=ARRAY_LAYOUT_NAME,
    )

    interpolators = lookup.prepare_point_interpolators()

    assert lookup.lookup_points_for_interpolation.shape[1] == 3
    assert set(interpolators) == {
        "lower_energy_threshold",
        "upper_scatter_radius",
        "viewcone_radius",
    }


def test_interpolate_point_returns_interpolated_values(corsika_limits_for_test_file):
    lookup = CorsikaLimitsLookup(
        corsika_limits_for_test_file,
        array_layout_name=ARRAY_LAYOUT_NAME,
    )

    interpolated = lookup.interpolate_point(20 * u.deg, 0 * u.deg, nsb=0.24)

    assert interpolated["lower_energy_threshold"] == pytest.approx(0.003)
    assert interpolated["upper_scatter_radius"] == pytest.approx(1225.0)
    assert interpolated["viewcone_radius"] == pytest.approx(10.0)


def test_prepare_point_interpolators_supports_two_varying_dimensions(tmp_test_directory):
    lookup_table = Table(
        rows=[
            ("2d-array", 20.0, 0.0, 1.0, 0.01, 800.0, 6.0),
            ("2d-array", 20.0, 180.0, 1.0, 0.03, 1000.0, 8.0),
            ("2d-array", 40.0, 0.0, 1.0, 0.02, 900.0, 7.0),
            ("2d-array", 40.0, 180.0, 1.0, 0.04, 1100.0, 9.0),
        ],
        names=(
            "array_name",
            "zenith",
            "azimuth",
            "nsb_level",
            "lower_energy_limit",
            "upper_radius_limit",
            "viewcone_radius",
        ),
    )
    lookup_file = tmp_test_directory / "corsika_limits_2d.ecsv"
    lookup_table.write(lookup_file, format="ascii.ecsv", overwrite=True)

    lookup = CorsikaLimitsLookup(lookup_file, array_layout_name="2d-array")
    lookup.prepare_point_interpolators()

    assert lookup.lookup_interpolation_axes == [0, 1]

    interpolated = lookup.interpolate_point(30 * u.deg, 90 * u.deg, nsb=1)

    assert interpolated["lower_energy_threshold"] == pytest.approx(0.025)
    assert interpolated["upper_scatter_radius"] == pytest.approx(950.0)
    assert interpolated["viewcone_radius"] == pytest.approx(7.5)
