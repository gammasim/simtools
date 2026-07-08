import astropy.units as u
import numpy as np
import pytest
from astropy.table import Table
from astropy.tests.helper import assert_quantity_allclose

import simtools.production_configuration.derive_corsika_limits as derive_corsika_limits
import simtools.production_configuration.production_event_data_helpers as event_data_helpers

# Constants
SIM_EVENTS_HISTOGRAMS_PATH = (
    "simtools.production_configuration.derive_corsika_limits.EventDataHistograms"
)
COMPUTE_LOWER_ENERGY_LIMIT_PATH = (
    "simtools.production_configuration.derive_corsika_limits.compute_lower_energy_limit"
)
COMPUTE_LIMITS_PATH = "simtools.production_configuration.derive_corsika_limits._compute_limits"
MOCK_FILE_PATH = "mock_file.fits"
DEFAULT_ALLOWED_LOSSES = {
    "core_distance": {"loss_fraction": 0.2, "loss_min_events": 10},
    "angular_distance": {"loss_fraction": 0.2, "loss_min_events": 10},
}


def _pool_result(
    production_index=0,
    event_data_file="pattern_*.hdf5",
    array_name="LST",
    telescope_ids=None,
    lower_energy_limit=0.5 * u.TeV,
    upper_radius_limit=400.0 * u.m,
    viewcone_radius=5.0 * u.deg,
):
    """Build a standard mocked pool result row for grid execution tests."""
    return {
        "production_index": production_index,
        "event_data_file": event_data_file,
        "array_name": array_name,
        "telescope_ids": telescope_ids or ["LSTN-01"],
        "lower_energy_limit": lower_energy_limit,
        "upper_radius_limit": upper_radius_limit,
        "viewcone_radius": viewcone_radius,
        "primary_particle": "gamma",
        "zenith": 20.0 * u.deg,
        "azimuth": 180.0 * u.deg,
        "nsb_level": 1.0,
    }


def test_write_results(mocker, mock_args_dict, mock_results, tmp_test_directory):
    """Test write_results function."""
    mock_io = mocker.patch("simtools.io.io_handler.IOHandler")
    mock_io.return_value.get_output_directory.return_value = tmp_test_directory

    mock_dump = mocker.patch("simtools.data_model.metadata_collector.MetadataCollector.dump")

    derive_corsika_limits.write_results(mock_results, mock_args_dict, DEFAULT_ALLOWED_LOSSES, 0.1)

    # Verify metadata was written
    mock_dump.assert_called_once()
    args = mock_dump.call_args[0]
    assert args[0] == mock_args_dict


def test_create_results_table(mock_results):
    """Test _create_results_table function."""
    table = derive_corsika_limits._create_results_table(mock_results, DEFAULT_ALLOWED_LOSSES, 0.1)
    table.info()

    assert isinstance(table, Table)
    assert len(table) == 1
    assert "telescope_ids" not in table.colnames
    assert "zenith" in table.colnames
    assert table["zenith"].unit == u.deg
    assert "br_energy_min" in table.colnames
    assert "br_energy_max" in table.colnames
    assert "br_core_scatter_max" in table.colnames
    assert "br_viewcone_max" in table.colnames
    assert table["br_energy_min"].unit == u.TeV
    assert table["br_energy_max"].unit == u.TeV
    assert table["br_core_scatter_max"].unit == u.m
    assert table["br_viewcone_max"].unit == u.deg
    assert (
        table["br_viewcone_max"].description
        == derive_corsika_limits.COLUMN_DESCRIPTIONS["br_viewcone_max"]
    )
    assert table.meta["loss_fraction_core_distance"] == pytest.approx(0.2)
    assert table.meta["loss_min_events_core_distance"] == 10
    assert table.meta["loss_fraction_angular_distance"] == pytest.approx(0.2)
    assert table.meta["loss_min_events_angular_distance"] == 10
    assert table.meta["energy_threshold_fraction"] == pytest.approx(0.1)
    assert isinstance(table.meta["created"], str)
    assert "description" in table.meta


def test_file_info_columns_are_read_from_schema():
    """Test file-info column mappings are read from the CORSIKA limits schema."""
    assert derive_corsika_limits.FILE_INFO_COLUMNS == {
        "primary_particle": "primary_particle",
        "zenith": "zenith",
        "azimuth": "azimuth",
        "nsb_level": "nsb_level",
        "br_energy_min": "energy_min",
        "br_energy_max": "energy_max",
        "br_core_scatter_max": "core_scatter_max",
        "br_viewcone_max": "viewcone_max",
    }


def test_load_output_table_configuration_from_schema_raises_without_data(mocker):
    """Raise if schema has no data section."""
    mocker.patch(
        "simtools.production_configuration.derive_corsika_limits.ascii_handler.collect_data_from_file",
        return_value={},
    )

    with pytest.raises(KeyError, match="No 'data' entry found"):
        derive_corsika_limits._load_output_table_configuration_from_schema("schema.yml")


def test_load_output_table_configuration_from_schema_raises_without_table_columns(mocker):
    """Raise if schema has data section but no table_columns."""
    mocker.patch(
        "simtools.production_configuration.derive_corsika_limits.ascii_handler.collect_data_from_file",
        return_value={"data": [{}]},
    )

    with pytest.raises(KeyError, match="No 'table_columns' entry found"):
        derive_corsika_limits._load_output_table_configuration_from_schema("schema.yml")


def test_round_value():
    """Test _round_value function for different key types."""

    # Test lower_energy_limit rounding
    assert derive_corsika_limits._round_value("lower_energy_limit", 1.2345) == pytest.approx(1.234)
    assert derive_corsika_limits._round_value("lower_energy_limit", 0.9876) == pytest.approx(0.987)
    assert derive_corsika_limits._round_value("lower_energy_limit", 2.0) == pytest.approx(2.0)
    assert derive_corsika_limits._round_value(
        "lower_energy_limit",
        0.0142,
        {"br_energy_min": 0.0142},
    ) == pytest.approx(0.0142)

    # Test upper_radius_limit rounding
    assert derive_corsika_limits._round_value("upper_radius_limit", 123.4) == 125
    assert derive_corsika_limits._round_value("upper_radius_limit", 100.0) == 100
    assert derive_corsika_limits._round_value("upper_radius_limit", 101.0) == 125
    assert derive_corsika_limits._round_value("upper_radius_limit", 75.0) == 75

    # Test viewcone_radius rounding
    assert derive_corsika_limits._round_value("viewcone_radius", 1.1) == pytest.approx(1.25)
    assert derive_corsika_limits._round_value("viewcone_radius", 2.0) == pytest.approx(2.0)
    assert derive_corsika_limits._round_value("viewcone_radius", 2.1) == pytest.approx(2.25)
    assert derive_corsika_limits._round_value("viewcone_radius", 0.3) == pytest.approx(0.5)

    # Test other keys (no rounding)
    assert derive_corsika_limits._round_value("other_key", 1.2345) == pytest.approx(1.2345)
    assert derive_corsika_limits._round_value("zenith", 45.678) == pytest.approx(45.678)
    assert derive_corsika_limits._round_value("unknown", "string_value") == "string_value"


def test_generate_corsika_limits_grid_requires_trigger_histogram_file(mock_args_dict):
    """Require a precomputed trigger-histogram file."""
    args = mock_args_dict.copy()
    args["trigger_histogram_file"] = None

    with pytest.raises(ValueError, match="Use --trigger_histogram_file"):
        derive_corsika_limits.generate_corsika_limits_grid(args)


def test_generate_corsika_limits_grid_from_trigger_histogram_file(
    mocker, mock_args_dict, tmp_test_directory
):
    """Use precomputed trigger histograms without resolving telescope configuration."""
    args = mock_args_dict.copy()
    args["trigger_histogram_file"] = "trigger_histograms.hdf5"
    args["array_names"] = ["alpha"]

    metadata = Table(
        rows=[
            {
                "production_index": 0,
                "event_data_file": "prod/*.hdf5",
                "array_name": "alpha",
                "telescope_ids": "LSTN-01",
            }
        ]
    )
    histograms = mocker.Mock()
    mock_load = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits.load_event_data_histograms",
        return_value=[(metadata[0], histograms)],
    )
    mock_derive = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits._derive_limits_from_histograms",
        return_value=_pool_result(array_name="alpha"),
    )
    mock_write = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits.write_results"
    )
    mocker.patch(
        "simtools.production_configuration.derive_corsika_limits.io_handler.IOHandler"
    ).return_value.get_output_directory.return_value = tmp_test_directory

    derive_corsika_limits.generate_corsika_limits_grid(args)

    mock_load.assert_called_once_with("trigger_histograms.hdf5", array_names=["alpha"])
    mock_derive.assert_called_once()
    result = mock_write.call_args[0][0][0]
    assert result["event_data_file"] == "prod/*.hdf5"
    assert result["array_name"] == "alpha"
    assert result["telescope_ids"] == ["LSTN-01"]


def test_generate_corsika_limits_grid_uses_all_arrays_when_array_names_not_given(
    mocker, mock_args_dict, tmp_test_directory
):
    """Load all array names from the trigger-histogram file when no filter is given."""
    args = mock_args_dict.copy()
    args["trigger_histogram_file"] = "trigger_histograms.hdf5"
    args["array_names"] = None

    metadata = Table(
        rows=[
            {
                "production_index": 0,
                "event_data_file": "prod/*.hdf5",
                "array_name": "alpha",
                "telescope_ids": "LSTN-01",
            },
            {
                "production_index": 0,
                "event_data_file": "prod/*.hdf5",
                "array_name": "beta",
                "telescope_ids": "MSTS-01",
            },
        ]
    )
    histograms_alpha = mocker.Mock()
    histograms_beta = mocker.Mock()
    mock_load = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits.load_event_data_histograms",
        return_value=[(metadata[0], histograms_alpha), (metadata[1], histograms_beta)],
    )
    mock_derive = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits._derive_limits_from_histograms",
        side_effect=[_pool_result(array_name="alpha"), _pool_result(array_name="beta")],
    )
    mock_write = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits.write_results"
    )
    mocker.patch(
        "simtools.production_configuration.derive_corsika_limits.io_handler.IOHandler"
    ).return_value.get_output_directory.return_value = tmp_test_directory

    derive_corsika_limits.generate_corsika_limits_grid(args)

    mock_load.assert_called_once_with("trigger_histograms.hdf5", array_names=None)
    assert mock_derive.call_count == 2
    results = mock_write.call_args[0][0]
    assert [result["array_name"] for result in results] == ["alpha", "beta"]
    assert results[0]["telescope_ids"] == ["LSTN-01"]
    assert results[1]["telescope_ids"] == ["MSTS-01"]


def test_resolve_telescope_configs_wraps_single_layout_result(mocker):
    """Wrap a non-list layout resolution result into a list before DB lookup."""
    mock_resolve = mocker.patch(
        "simtools.production_configuration.production_event_data_helpers.resolve_array_layout_name",
        return_value="single-layout",
    )
    mock_db_lookup = mocker.patch(
        (
            "simtools.production_configuration.production_event_data_helpers."
            "get_array_elements_from_db_for_layouts"
        ),
        return_value={"LST": ["LSTN-01"]},
    )

    result = event_data_helpers.resolve_telescope_configs(
        {
            "array_layout_name": "layout",
            "model_version": "1.0.0",
            "site": "South",
        }
    )

    mock_resolve.assert_called_once_with("layout", "1.0.0")
    mock_db_lookup.assert_called_once_with(["single-layout"], "South", "1.0.0")
    assert result == {"LST": ["LSTN-01"]}


@pytest.mark.parametrize(
    ("allowed_losses", "error_match"),
    [
        (["core_distance,0.2"], "Expected format"),
        (["core_distance,abc,10"], "fraction must be float"),
        (["invalid,0.2,10"], "Invalid axis"),
        (["core_distance,0.2,10"], "Missing --allowed_losses entries"),
    ],
)
def test_parse_allowed_losses_error_paths(allowed_losses, error_match):
    """Validate error handling for malformed or incomplete allowed-loss inputs."""
    with pytest.raises(ValueError, match=error_match):
        derive_corsika_limits._parse_allowed_losses(allowed_losses)


def test_parse_allowed_losses_raises_when_not_provided():
    """Reject missing allowed-loss configuration."""
    with pytest.raises(ValueError, match="No allowed-loss configuration provided"):
        derive_corsika_limits._parse_allowed_losses(None)


def test_compute_limits_lower():
    hist = np.array([1, 2, 3, 4, 5])
    bin_edges = np.array([0, 1, 2, 3, 4, 5])
    loss_fraction = 0.2

    with pytest.raises(ValueError, match="limit_type must be 'lower' or 'upper'"):
        derive_corsika_limits._integral_limits(hist, bin_edges, loss_fraction, limit_type="blabla")

    result = derive_corsika_limits._integral_limits(
        hist,
        bin_edges,
        loss_fraction,
        loss_min_events=0,
        limit_type="lower",
    )
    assert result == 3


def test_compute_limits_upper():
    hist = np.array([5, 4, 3, 2, 1])
    bin_edges = np.array([0, 1, 2, 3, 4, 5])
    loss_fraction = 0.2

    result = derive_corsika_limits._integral_limits(
        hist,
        bin_edges,
        loss_fraction,
        loss_min_events=0,
        limit_type="upper",
    )
    assert result == 3


def test_compute_limits_default_type():
    hist = np.array([1, 2, 3, 4, 5])
    bin_edges = np.array([0, 1, 2, 3, 4, 5])
    loss_fraction = 0.2

    result = derive_corsika_limits._integral_limits(
        hist,
        bin_edges,
        loss_fraction,
        loss_min_events=0,
    )
    assert result == 3


def test_compute_limits_enforces_minimum_lost_events_upper():
    """Test _compute_limits enforces an absolute minimum number of lost events."""
    hist = np.array([5, 4, 3, 2, 1])
    bin_edges = np.array([0, 1, 2, 3, 4, 5])

    result = derive_corsika_limits._integral_limits(
        hist,
        bin_edges,
        loss_fraction=0.2,
        loss_min_events=10,
        limit_type="upper",
    )
    assert result == 1


def test_compute_limits_enforces_minimum_lost_events_lower():
    """Test lower limits also honor the absolute minimum loss requirement."""
    hist = np.array([1, 2, 3, 4, 5])
    bin_edges = np.array([0, 1, 2, 3, 4, 5])

    result = derive_corsika_limits._integral_limits(
        hist,
        bin_edges,
        loss_fraction=0.2,
        loss_min_events=10,
        limit_type="lower",
    )
    assert result == 5


def test_compute_lower_energy_limit(mocker):
    """Test compute_lower_energy_limit function with mocked histograms."""
    mock_hist = np.array([1.0, 12.0, 20.0, 12.0, 1.0])
    mock_bins = np.logspace(-3, 3, 6)

    # Mock the histograms object
    mock_histograms = mocker.MagicMock()
    mock_histograms.histograms = {"energy": {"histogram": mock_hist}}
    mock_histograms.energy_bins = mock_bins
    mock_histograms.file_info = {}

    result = derive_corsika_limits.compute_lower_energy_limit(mock_histograms, 0.2)

    assert isinstance(result, u.Quantity)
    assert result.unit == u.TeV
    assert result.value > 0

    expected = (
        derive_corsika_limits._find_low_energy_threshold_from_histogram(
            mock_hist,
            mock_bins,
            threshold_fraction=0.2,
        )
        * u.TeV
    )
    assert result == expected


def test_compute_lower_energy_limit_never_below_broad_range_min(mocker):
    """Test compute_lower_energy_limit applies broad-range lower-energy floor."""
    mock_hist = np.array([1.0, 12.0, 20.0, 12.0, 1.0])
    mock_bins = np.array([0.01, 0.02, 0.04, 0.08, 0.16, 0.32])

    mock_histograms = mocker.MagicMock()
    mock_histograms.histograms = {"energy": {"histogram": mock_hist}}
    mock_histograms.energy_bins = mock_bins
    mock_histograms.file_info = {"energy_min": 0.014 * u.TeV}

    result = derive_corsika_limits.compute_lower_energy_limit(mock_histograms, 0.2)

    assert_quantity_allclose(result, 0.014 * u.TeV)


def test_apply_broad_range_lower_energy_floor_same_bin_uses_broad_range_min():
    """If derived and broad-range minimum share a bin, use broad-range minimum."""
    derived = 0.01 * u.TeV
    broad_range_min = 0.014 * u.TeV
    energy_bins = np.array([0.01, 0.02, 0.04])

    result = derive_corsika_limits._apply_broad_range_lower_energy_floor(
        derived,
        broad_range_min,
        energy_bins,
    )

    assert_quantity_allclose(result, 0.014 * u.TeV)


def test_apply_broad_range_lower_energy_floor_without_broad_range_min_returns_derived():
    """Return derived limit unchanged when no broad-range minimum is provided."""
    derived = 0.02 * u.TeV
    energy_bins = np.array([0.01, 0.02, 0.04])

    result = derive_corsika_limits._apply_broad_range_lower_energy_floor(
        derived,
        None,
        energy_bins,
    )

    assert_quantity_allclose(result, derived)


def test_apply_broad_range_lower_energy_floor_uses_enforced_minimum_for_different_bins():
    """Enforce broad-range floor when derived and broad-range minima are in different bins."""
    derived = 0.020 * u.TeV
    broad_range_min = 0.030 * u.TeV
    energy_bins = np.array([0.01, 0.02, 0.03, 0.04])

    result = derive_corsika_limits._apply_broad_range_lower_energy_floor(
        derived,
        broad_range_min,
        energy_bins,
    )

    assert_quantity_allclose(result, 0.030 * u.TeV)


def test_enforce_minimum_value_handles_quantity_and_scalar_mixed_types():
    """Cover all mixed quantity/scalar branches in minimum-value enforcement."""
    assert_quantity_allclose(
        derive_corsika_limits._enforce_minimum_value(1.0 * u.TeV, 1.2 * u.TeV),
        1.2 * u.TeV,
    )
    assert_quantity_allclose(
        derive_corsika_limits._enforce_minimum_value(1.0 * u.TeV, 1.2),
        1.2 * u.TeV,
    )
    assert derive_corsika_limits._enforce_minimum_value(1.0, 1.2 * u.TeV) == pytest.approx(1.2)


def test_enforce_minimum_value_returns_candidate_when_minimum_is_none():
    """Return candidate unchanged when no minimum is configured."""
    assert_quantity_allclose(
        derive_corsika_limits._enforce_minimum_value(1.0 * u.TeV, None),
        1.0 * u.TeV,
    )


def test_create_table_columns_uses_object_dtype_for_curve_columns():
    """Curve-like list values must be stored with object dtype columns."""
    cols = ["core_distance_vs_energy_curve"]
    columns = {"core_distance_vs_energy_curve": [[1.0, 2.0]]}
    units = {"core_distance_vs_energy_curve": None}

    table_cols = derive_corsika_limits._create_table_columns(cols, columns, units)

    assert table_cols[0].dtype == object


def test_create_results_table_rounding_keeps_lower_energy_at_or_above_broad_range_min():
    """Rounding must not push lower_energy_limit below br_energy_min."""
    results = [
        {
            "primary_particle": "proton",
            "array_name": "LST",
            "zenith": 20.0 * u.deg,
            "azimuth": 180.0 * u.deg,
            "nsb_level": 1.0,
            "lower_energy_limit": 0.0142 * u.TeV,
            "upper_radius_limit": 400.0 * u.m,
            "viewcone_radius": 5.0 * u.deg,
            "br_energy_min": 0.0142 * u.TeV,
            "br_energy_max": 300.0 * u.TeV,
            "br_core_scatter_max": 800.0 * u.m,
            "br_viewcone_max": 10.0 * u.deg,
        }
    ]

    table = derive_corsika_limits._create_results_table(results, DEFAULT_ALLOWED_LOSSES, 0.1)

    assert table["lower_energy_limit"][0] >= table["br_energy_min"][0]
    assert table["lower_energy_limit"][0] == pytest.approx(0.0142)


def test_find_low_energy_threshold_from_histogram_basic():
    """Test nominal threshold finding from peak toward lower energies."""
    counts = np.array([1.0, 12.0, 20.0, 12.0, 1.0])
    bin_edges = np.array([0.1, 0.2, 0.4, 0.8, 1.6, 3.2])

    # Peak index=2, N_peak=(12+20+12)/3=14.666..., threshold=1.466...
    # Walking left from idx=2: 20,12,1 -> first below threshold at idx=0
    result = derive_corsika_limits._find_low_energy_threshold_from_histogram(counts, bin_edges)
    assert result == pytest.approx(0.1)


def test_find_low_energy_threshold_from_histogram_peak_at_first_bin():
    """Test edge case where absolute maximum is at the first bin."""
    counts = np.array([10.0, 4.0, 1.0, 0.0])
    bin_edges = np.array([0.05, 0.1, 0.2, 0.4, 0.8])

    # No bins left of peak; fallback to first edge is expected.
    result = derive_corsika_limits._find_low_energy_threshold_from_histogram(counts, bin_edges)
    assert result == pytest.approx(0.05)


def test_find_low_energy_threshold_from_histogram_peak_at_last_bin():
    """Test edge case where absolute maximum is at the last bin."""
    counts = np.array([0.0, 0.2, 0.5, 10.0])
    bin_edges = np.array([0.1, 0.2, 0.4, 0.8, 1.6])

    # Peak index=3, N_peak=(0.5+10)/2=5.25, threshold=0.525
    # Walking left from idx=3: 10,0.5 -> first below threshold at idx=2
    result = derive_corsika_limits._find_low_energy_threshold_from_histogram(counts, bin_edges)
    assert result == pytest.approx(0.4)


@pytest.mark.parametrize(
    ("counts", "bin_edges", "threshold_fraction", "error_match"),
    [
        (np.array([[1.0, 2.0]]), np.array([0.1, 0.2, 0.4]), 0.1, "one-dimensional arrays"),
        (np.array([]), np.array([0.1]), 0.1, "must not be empty"),
        (np.array([1.0, 2.0]), np.array([0.1, 0.2]), 0.1, r"len\(counts\) \+ 1"),
        (np.array([1.0, 2.0]), np.array([0.1, 0.2, 0.4]), 0.0, "interval"),
    ],
)
def test_find_low_energy_threshold_from_histogram_validation_errors(
    counts,
    bin_edges,
    threshold_fraction,
    error_match,
):
    """Reject invalid histogram shapes and threshold settings."""
    with pytest.raises(ValueError, match=error_match):
        derive_corsika_limits._find_low_energy_threshold_from_histogram(
            counts,
            bin_edges,
            threshold_fraction=threshold_fraction,
        )


def test_find_low_energy_threshold_from_histogram_raises_for_all_zero_counts():
    """Reject histograms without positive entries."""
    counts = np.array([0.0, 0.0, 0.0, 0.0])
    bin_edges = np.array([0.1, 0.2, 0.4, 0.8, 1.6])

    with pytest.raises(ValueError, match="at least one positive entry"):
        derive_corsika_limits._find_low_energy_threshold_from_histogram(counts, bin_edges)


def test_is_close(caplog):
    """Test _is_close function behavior."""
    test_message = "Test message"

    with caplog.at_level("WARNING"):
        derive_corsika_limits._is_close(1.0 * u.m, None, test_message)
        assert test_message not in caplog.text

        derive_corsika_limits._is_close(1.0 * u.m, 25.0 * u.m, test_message)
        assert test_message not in caplog.text

        result = derive_corsika_limits._is_close(1.0 * u.m, 1.0 * u.m, test_message)
        assert test_message in caplog.text
        assert result.value == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("file_info", "expected_core_scatter_max", "expected_viewcone_max"),
    [
        (
            {"core_scatter_max": 120.0 * u.m, "viewcone_max": 3.0 * u.deg},
            120.0 * u.m,
            3.0 * u.deg,
        ),
        ({}, None, None),
    ],
)
def test_compute_limits(mocker, file_info, expected_core_scatter_max, expected_viewcone_max):
    """Test _compute_limits forwards slices and preserves units."""
    histograms = mocker.MagicMock()
    histograms.energy_bins = np.array([1.0, 10.0])
    histograms.core_distance_bins = np.array([0.0, 100.0])
    histograms.view_cone_bins = np.array([0.0, 5.0])
    histograms.histograms = {
        "core_distance_vs_energy": {"histogram": "core-hist"},
        "angular_distance_vs_energy": {"histogram": "viewcone-hist"},
    }
    histograms.file_info = file_info

    mock_diff_limits = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits._differential_upper_limits",
        side_effect=[
            (120.0, [110.0, 120.0], [1.0, 10.0]),
            (3.0, [2.5, 3.0], [1.0, 10.0]),
        ],
    )
    mock_is_close = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits._is_close",
        side_effect=[125.0 * u.m, 3.25 * u.deg],
    )

    derive_corsika_limits._compute_limits(histograms, DEFAULT_ALLOWED_LOSSES, 2)

    expected_diff_bins = np.logspace(0, 1, 3)
    np.testing.assert_allclose(mock_diff_limits.call_args_list[0].args[3], expected_diff_bins)
    np.testing.assert_allclose(mock_diff_limits.call_args_list[1].args[3], expected_diff_bins)
    assert mock_diff_limits.call_args_list[0].args[0] == "core-hist"
    assert mock_diff_limits.call_args_list[0].args[5:] == ("core_scatter", "m")
    assert mock_diff_limits.call_args_list[1].args[0] == "viewcone-hist"
    assert mock_diff_limits.call_args_list[1].args[5:] == ("viewcone", "deg")
    assert mock_diff_limits.call_args_list[0].args[4] == DEFAULT_ALLOWED_LOSSES["core_distance"]
    assert mock_diff_limits.call_args_list[1].args[4] == DEFAULT_ALLOWED_LOSSES["angular_distance"]

    assert mock_is_close.call_args_list[0].args[0].value == pytest.approx(120.0)
    assert mock_is_close.call_args_list[0].args[1] == expected_core_scatter_max
    assert mock_is_close.call_args_list[1].args[0].value == pytest.approx(3.0)
    assert mock_is_close.call_args_list[1].args[1] == expected_viewcone_max


def test_compute_limits_with_integral_fallback_curves(mocker):
    """Test _compute_limits returns energy-independent curves for integral limits."""
    histograms = mocker.MagicMock()
    histograms.energy_bins = np.array([1.0, 10.0])
    histograms.core_distance_bins = np.array([0.0, 100.0])
    histograms.view_cone_bins = np.array([0.0, 5.0])
    histograms.histograms = {
        "core_distance": {"histogram": np.array([1.0, 2.0])},
        "angular_distance": {"histogram": np.array([3.0, 4.0])},
    }
    histograms.file_info = {}

    mock_integral_limits = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits._integral_limits",
        side_effect=[120.0, 3.0],
    )
    mock_diff_limits = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits._differential_upper_limits"
    )
    mocker.patch(
        "simtools.production_configuration.derive_corsika_limits._is_close",
        side_effect=lambda value, *_: value,
    )

    result = derive_corsika_limits._compute_limits(
        histograms,
        DEFAULT_ALLOWED_LOSSES,
        bins_per_decade=0,
    )

    assert mock_integral_limits.call_count == 2
    mock_diff_limits.assert_not_called()
    assert result["core_distance_vs_energy_curve"] == {"x": [120.0, 120.0], "y": [1.0, 10.0]}
    assert result["angular_distance_vs_energy_curve"] == {"x": [3.0, 3.0], "y": [1.0, 10.0]}


def test_compute_limits_uses_exact_constant_angular_distance(mocker):
    """A fixed angular distance bypasses histogram limits and their bin-edge offset."""
    histograms = mocker.MagicMock()
    histograms.energy_bins = np.array([1.0, 10.0])
    histograms.core_distance_bins = np.array([0.0, 100.0])
    histograms.view_cone_bins = np.array([0.0, 0.5])
    histograms.data_ranges = {"angular_distance": (0.0, 0.0)}
    histograms.histograms = {
        "core_distance": {"histogram": np.array([10.0])},
        "angular_distance": {"histogram": np.array([10.0])},
    }
    histograms.file_info = {"viewcone_max": 0.0 * u.deg}

    integral_limits = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits._integral_limits",
        return_value=100.0,
    )

    result = derive_corsika_limits._compute_limits(histograms, DEFAULT_ALLOWED_LOSSES, 0)

    integral_limits.assert_called_once()
    assert_quantity_allclose(result["viewcone_radius"], 0.0 * u.deg)
    assert result["angular_distance_is_constant"] is True
    assert result["angular_distance_vs_energy_curve"] == {"x": [0.0, 0.0], "y": [1.0, 10.0]}


def test_get_constant_data_value():
    """Return the constant value only when the stored range is effectively flat."""
    histograms = type(
        "HistogramContainer",
        (),
        {
            "data_ranges": {
                "angular_distance": (1.0, 1.0 + 1.0e-13),
                "angular_distance_near_zero": (0.0, 1.0e-12),
                "angular_distance_small": (0.009, 0.018),
                "core_distance": (1.0, 2.0),
            }
        },
    )()

    assert derive_corsika_limits._get_constant_data_value(
        histograms, "angular_distance"
    ) == pytest.approx(1.0)
    assert derive_corsika_limits._get_constant_data_value(
        histograms, "angular_distance_near_zero"
    ) == pytest.approx(0.0)
    assert derive_corsika_limits._get_constant_data_value(
        histograms, "angular_distance_small"
    ) == pytest.approx(0.0135)
    assert derive_corsika_limits._get_constant_data_value(histograms, "core_distance") is None
    assert derive_corsika_limits._get_constant_data_value(histograms, "missing") is None


def test_constant_angular_distance_is_not_rounded_in_results_table(mock_results):
    """Preserve the raw constant value instead of rounding to viewcone increments."""
    mock_results[0]["viewcone_radius"] = 0.1 * u.deg
    mock_results[0]["angular_distance_is_constant"] = True

    table = derive_corsika_limits._create_results_table(mock_results, DEFAULT_ALLOWED_LOSSES, 0.1)

    assert table["viewcone_radius"][0] == pytest.approx(0.1)


def test_constant_angular_distance_distributions_are_not_plotted(mocker, tmp_test_directory):
    """Suppress all angular-distance-vs-* plots for fixed-direction simulations."""
    histograms = mocker.MagicMock()
    histograms.file_info = {}
    histograms.histograms = {
        "energy": {"histogram": np.array([1.0])},
        "angular_distance_vs_energy": {"histogram": np.array([[1.0]])},
        "angular_distance_vs_energy_mc": {"histogram": np.array([[1.0]])},
        "angular_distance_vs_energy_cumulative": {"histogram": np.array([[1.0]])},
    }
    mocker.patch(COMPUTE_LOWER_ENERGY_LIMIT_PATH, return_value=1.0 * u.TeV)
    mocker.patch(
        COMPUTE_LIMITS_PATH,
        return_value={
            "upper_radius_limit": 100.0 * u.m,
            "viewcone_radius": 0.0 * u.deg,
            "angular_distance_is_constant": True,
        },
    )
    plot = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits.plot_simtel_event_histograms.plot"
    )

    derive_corsika_limits._derive_limits_from_histograms(
        histograms,
        "MockArray",
        DEFAULT_ALLOWED_LOSSES,
        0.01,
        True,
        tmp_test_directory,
        0,
    )

    plotted_histograms = plot.call_args.args[0]
    assert set(plotted_histograms) == set()
    assert plot.call_args.kwargs["add_distance_projections"] is True
    assert plot.call_args.kwargs["use_broad_range_limits"] is True


def test_differential_upper_limits(mocker):
    """Test _differential_upper_limits slices energies and skips empty bins."""
    mock_integral_limits = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits._integral_limits",
        side_effect=[1.5, 2.5],
    )
    mock_log = mocker.patch("simtools.production_configuration.derive_corsika_limits._logger.info")

    max_limit, limits, energy_centers = derive_corsika_limits._differential_upper_limits(
        histogram2d=np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]),
        x_bins=np.array([0.0, 1.0, 2.0, 3.0]),
        y_bins=np.array([1.0, 2.0, 4.0]),
        diff_e_bins=np.array([1.0, 2.0, 2.5, 3.0]),
        allowed_loss=DEFAULT_ALLOWED_LOSSES["core_distance"],
        name="core_scatter",
        unit="m",
    )

    np.testing.assert_array_equal(
        mock_integral_limits.call_args_list[0].args[0], np.array([1.0, 2.0, 3.0])
    )
    np.testing.assert_array_equal(
        mock_integral_limits.call_args_list[1].args[0], np.array([10.0, 20.0, 30.0])
    )
    assert max_limit == pytest.approx(2.5)
    assert limits == [1.5, 2.5]
    assert energy_centers == pytest.approx([np.sqrt(2.0), np.sqrt(7.5)])
    assert mock_log.call_count == 2


def test_differential_upper_limits_falls_back_to_last_bin_edge(mocker):
    """Test _differential_upper_limits falls back when all slices are empty."""
    mock_integral_limits = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits._integral_limits"
    )
    mock_log = mocker.patch("simtools.production_configuration.derive_corsika_limits._logger.info")

    result = derive_corsika_limits._differential_upper_limits(
        histogram2d=np.zeros((3, 2)),
        x_bins=np.array([0.0, 1.0, 2.0, 3.0]),
        y_bins=np.array([1.0, 2.0, 4.0]),
        diff_e_bins=np.array([1.0, 2.0, 3.0]),
        allowed_loss=DEFAULT_ALLOWED_LOSSES["angular_distance"],
        name="viewcone",
        unit="deg",
    )

    assert result == (3.0, [], [])
    mock_integral_limits.assert_not_called()
    mock_log.assert_not_called()


def test_normalize_event_data_file_single_string():
    """Test _normalize_event_data_file with single string input."""
    result = event_data_helpers.normalize_event_data_file("pattern_*.hdf5")
    assert result == ["pattern_*.hdf5"]
    assert isinstance(result, list)


def test_normalize_event_data_file_list():
    """Test _normalize_event_data_file with list input."""
    patterns = ["pattern_1_*.hdf5", "pattern_2_*.hdf5"]
    result = event_data_helpers.normalize_event_data_file(patterns)
    assert result == patterns
    # Should preserve order
    assert result[0] == "pattern_1_*.hdf5"
    assert result[1] == "pattern_2_*.hdf5"


def test_normalize_event_data_file_invalid_type():
    """Test _normalize_event_data_file raises on invalid type."""
    with pytest.raises(TypeError):
        event_data_helpers.normalize_event_data_file(123)


def test_get_production_directory_name_readable_and_deterministic():
    """Test _get_production_directory_name generates readable deterministic names."""
    # Same inputs should produce same output when no collision exists
    name1 = event_data_helpers.get_production_directory_name("pattern_1_*.hdf5")
    name2 = event_data_helpers.get_production_directory_name("pattern_1_*.hdf5")
    assert name1 == name2

    # Different patterns should produce different readable names
    name3 = event_data_helpers.get_production_directory_name("pattern_2_*.hdf5")
    assert name1 != name3

    # Names should be filesystem-safe (no special chars)
    assert all(c.isalnum() or c == "_" for c in name1)
    assert name1 == "production_pattern_1"


def test_get_production_directory_name_uses_parent_dir_only():
    """Parent directory name is used alone to avoid duplication with the filename stem."""
    name = event_data_helpers.get_production_directory_name(
        "/data/electron_z20_north_dark10p/electron_20deg_0deg_run00000*hdf5"
    )
    assert name == "production_electron_z20_north_dark10p"
    # Must not repeat "electron" from the filename stem
    assert name.count("electron") == 1


def test_get_production_directory_name_appends_uuid_on_collision(mocker):
    """Test _get_production_directory_name appends UUID when names collide."""
    mock_uuid = mocker.patch(
        "simtools.production_configuration.production_event_data_helpers.get_uuid",
        return_value="019d776b-e24c-741d-bc05-e3f6f7ec77c7",
    )

    name = event_data_helpers.get_production_directory_name(
        "pattern_1_*.hdf5",
        existing_names={"production_pattern_1"},
    )

    assert name == "production_pattern_1_019d776b-e24c-741d-bc05-e3f6f7ec77c7"
    mock_uuid.assert_called_once()


def test_parse_allowed_losses_explicit_axes():
    """Test _parse_allowed_losses with explicit per-axis entries."""
    result = derive_corsika_limits._parse_allowed_losses(
        [
            "core_distance,2e-6,20",
            "angular_distance,3e-6,30",
        ]
    )

    assert result["core_distance"]["loss_fraction"] == pytest.approx(2e-6)
    assert result["core_distance"]["loss_min_events"] == 20
    assert result["angular_distance"]["loss_fraction"] == pytest.approx(3e-6)
    assert result["angular_distance"]["loss_min_events"] == 30


def test_parse_allowed_losses_all_and_override():
    """Test _parse_allowed_losses supports all plus later axis override."""
    result = derive_corsika_limits._parse_allowed_losses(
        [
            "all,1e-6,10",
            "core_distance,5e-7,5",
        ]
    )

    assert result["core_distance"]["loss_fraction"] == pytest.approx(5e-7)
    assert result["core_distance"]["loss_min_events"] == 5
    assert result["angular_distance"]["loss_fraction"] == pytest.approx(1e-6)
    assert result["angular_distance"]["loss_min_events"] == 10


def test_parse_allowed_losses_missing_axis_raises():
    """Test _parse_allowed_losses raises when required axes are missing."""
    with pytest.raises(ValueError, match="Missing --allowed_losses entries"):
        derive_corsika_limits._parse_allowed_losses(
            [
                "core_distance,1e-6,10",
            ]
        )


def test_parse_allowed_losses_invalid_axis_raises():
    """Test _parse_allowed_losses rejects invalid axis names."""
    with pytest.raises(ValueError, match="Invalid axis for --allowed_losses"):
        derive_corsika_limits._parse_allowed_losses(
            [
                "core_distance,1e-6,10",
                "viewcone,1e-6,10",
            ]
        )


def test_build_production_subdirectories_single_production(tmp_test_directory):
    """Test build_production_subdirectories creates a single directory."""
    result = event_data_helpers.build_production_subdirectories(
        ["pattern_1_*.hdf5"],
        tmp_test_directory,
    )
    assert set(result.keys()) == {"pattern_1_*.hdf5"}
    assert result["pattern_1_*.hdf5"].exists()


def test_build_production_subdirectories_creates_dirs(tmp_test_directory):
    """Test build_production_subdirectories creates per-production directories."""
    patterns = ["pattern_1_*.hdf5", "pattern_2_*.hdf5"]
    result = event_data_helpers.build_production_subdirectories(
        patterns,
        tmp_test_directory,
    )

    assert set(result.keys()) == set(patterns)
    for output_subdir in result.values():
        assert output_subdir.exists()
        assert output_subdir.isdir()


def test_create_results_table_with_production_columns(mock_results):
    """Test _create_results_table includes production-origin columns for multi-production."""
    # Add production metadata to mock results
    for i, res in enumerate(mock_results):
        res["production_index"] = i
        res["event_data_file"] = f"pattern_{i}_*.hdf5"

    table = derive_corsika_limits._create_results_table(mock_results, DEFAULT_ALLOWED_LOSSES, 0.1)

    # Should include production-origin columns
    assert "production_index" in table.colnames
    assert "event_data_file" not in table.colnames

    # Check values
    assert table["production_index"][0] == 0


def test_create_results_table_without_production_columns(mock_results):
    """Test _create_results_table with missing production metadata values."""
    # Results without production metadata (old format)
    table = derive_corsika_limits._create_results_table(mock_results, DEFAULT_ALLOWED_LOSSES, 0.1)

    # Production-origin column is included and filled with None if missing
    assert "production_index" in table.colnames
    assert "event_data_file" not in table.colnames
    assert table["production_index"][0] is None

    # Standard columns should be present
    assert "primary_particle" in table.colnames
    assert "array_name" in table.colnames


@pytest.fixture
def mock_args_dict():
    """Fixture to provide mock arguments dictionary with required keys."""
    return {
        "config_file": "dummy_config.yml",
        "steps": None,
        "ignore_runtime_environment": False,
        "trigger_histogram_file": "trigger_histograms.hdf5",
        "array_names": None,
        "output_file": "corsika_limits.ecsv",
        "allowed_losses": [
            "core_distance,0.2,10",
            "angular_distance,0.2,10",
        ],
        "energy_threshold_fraction": 0.1,
        "plot_histograms": False,
    }


@pytest.fixture
def mock_results():
    """Fixture to provide one standard result row for table/writer tests."""
    return [
        {
            "primary_particle": "gamma",
            "array_name": "LST",
            "zenith": 20.0 * u.deg,
            "azimuth": 180.0 * u.deg,
            "nsb_level": 1.0,
            "lower_energy_limit": 0.5 * u.TeV,
            "upper_radius_limit": 400.0 * u.m,
            "viewcone_radius": 5.0 * u.deg,
            "br_energy_min": 0.03 * u.TeV,
            "br_energy_max": 300.0 * u.TeV,
            "br_core_scatter_max": 800.0 * u.m,
            "br_viewcone_max": 10.0 * u.deg,
        }
    ]
