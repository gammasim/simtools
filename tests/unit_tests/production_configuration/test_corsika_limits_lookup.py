import pytest

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
