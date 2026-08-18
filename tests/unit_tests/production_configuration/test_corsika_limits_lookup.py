from itertools import product
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from astropy.table import Table
from astropy.tests.helper import assert_quantity_allclose

from simtools.constants import SCHEMA_PATH
from simtools.io import ascii_handler
from simtools.production_configuration.corsika_limits_lookup import CorsikaLimitsLookup

ARRAY_LAYOUT_NAME = "CTAO-North-LSTs"


@pytest.fixture
def corsika_limits_for_test_file(tmp_test_directory):
    schema = ascii_handler.collect_data_from_file(SCHEMA_PATH / "corsika_limits_table.schema.yml")
    table_columns = schema["data"][0]["table_columns"]
    column_names = [column["name"] for column in table_columns]
    rows = []
    for array_name in [ARRAY_LAYOUT_NAME, *[f"other-layout-{index}" for index in range(5)]]:
        for zenith, azimuth, nsb_level in product(
            (20.0, 40.0), (0.0, 180.0, 90.0, 270.0), (0.24, 0.84)
        ):
            row = {
                "production_index": 0,
                "primary_particle": "gamma",
                "array_name": array_name,
                "zenith": zenith,
                "azimuth": azimuth,
                "nsb_level": nsb_level,
                "lower_energy_limit": 0.003,
                "upper_radius_limit": 1225.0,
                "viewcone_radius": 10.0,
                "br_energy_min": 0.003,
                "br_energy_max": 330.0,
                "br_core_scatter_max": 1225.0,
                "br_viewcone_max": 10.0,
            }
            rows.append(tuple(row[column] for column in column_names))

    table = Table(rows=rows, names=column_names)
    for column in table_columns:
        if column["unit"] != "dimensionless":
            table[column["name"]].unit = column["unit"]

    lookup_file = Path(tmp_test_directory) / "corsika_limits.ecsv"
    table.write(lookup_file, format="ascii.ecsv", overwrite=True)
    return lookup_file


def test_load_matching_lookup_arrays_filters_by_array_layout_name(corsika_limits_for_test_file):
    lookup = CorsikaLimitsLookup(
        corsika_limits_for_test_file,
        array_layout_name=ARRAY_LAYOUT_NAME,
    )

    arrays = lookup.load_matching_lookup_arrays()

    assert len(arrays["points"]) == 16
    assert arrays["lower_energy_limit"][0] == pytest.approx(0.003)


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
    assert {"lower_energy_limit", "upper_radius_limit", "viewcone_radius"}.issubset(
        set(interpolators)
    )
    assert {"br_energy_min", "br_energy_max"}.issubset(set(interpolators))


def test_interpolate_grid_limits_returns_requested_grid_shape(corsika_limits_for_test_file):
    lookup = CorsikaLimitsLookup(
        corsika_limits_for_test_file,
        array_layout_name=ARRAY_LAYOUT_NAME,
    )

    interpolated = lookup.interpolate_grid_limits(
        {
            "zenith_angle": [20, 40] * u.deg,
            "azimuth": [0, 180] * u.deg,
            "nsb_level": [0.24, 0.84] * u.dimensionless_unscaled,
        }
    )

    assert interpolated["lower_energy_limit"].shape == (2, 2, 2)
    assert np.isfinite(interpolated["upper_radius_limit"][0, 0, 0])


def test_interpolate_point_returns_interpolated_values(corsika_limits_for_test_file):
    lookup = CorsikaLimitsLookup(
        corsika_limits_for_test_file,
        array_layout_name=ARRAY_LAYOUT_NAME,
    )

    interpolated = lookup.interpolate_point(20 * u.deg, 0 * u.deg, nsb=0.24)

    assert_quantity_allclose(interpolated["lower_energy_limit"], 0.003 * u.TeV)
    assert_quantity_allclose(interpolated["upper_radius_limit"], 1225.0 * u.m)
    assert_quantity_allclose(interpolated["viewcone_radius"], 10.0 * u.deg)


def _write_2d_lookup_table(tmp_test_directory, file_name):
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
    lookup_table["lower_energy_limit"].unit = "TeV"
    lookup_table["upper_radius_limit"].unit = "m"
    lookup_table["viewcone_radius"].unit = "deg"
    lookup_file = tmp_test_directory / file_name
    lookup_table.write(lookup_file, format="ascii.ecsv", overwrite=True)
    return lookup_file


def test_interpolate_point_falls_back_to_nearest_for_out_of_domain_point(tmp_test_directory):
    lookup_file = _write_2d_lookup_table(tmp_test_directory, "corsika_limits_outside_domain.ecsv")

    lookup = CorsikaLimitsLookup(lookup_file, array_layout_name="2d-array")

    interpolated = lookup.interpolate_point(10 * u.deg, 0 * u.deg, nsb=1)

    assert_quantity_allclose(interpolated["lower_energy_limit"], 0.01 * u.TeV)
    assert_quantity_allclose(interpolated["upper_radius_limit"], 800.0 * u.m)
    assert_quantity_allclose(interpolated["viewcone_radius"], 6.0 * u.deg)


def test_prepare_point_interpolators_supports_two_varying_dimensions(tmp_test_directory):
    lookup_file = _write_2d_lookup_table(tmp_test_directory, "corsika_limits_2d.ecsv")

    lookup = CorsikaLimitsLookup(lookup_file, array_layout_name="2d-array")
    lookup.prepare_point_interpolators()

    assert lookup.lookup_interpolation_axes == [0, 1]

    interpolated = lookup.interpolate_point(30 * u.deg, 90 * u.deg, nsb=1)

    assert_quantity_allclose(interpolated["lower_energy_limit"], 0.025 * u.TeV)
    assert_quantity_allclose(interpolated["upper_radius_limit"], 950.0 * u.m)
    assert_quantity_allclose(interpolated["viewcone_radius"], 7.5 * u.deg)
