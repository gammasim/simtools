from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.table import Table
from astropy.tests.helper import assert_quantity_allclose

import simtools.production_configuration.derive_corsika_limits as derive_corsika_limits
import simtools.production_configuration.production_event_data_helpers as event_data_helpers
from simtools.histogram.histogram import Histogram
from simtools.io import table_handler

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


def _write_particle_metadata(file_path, particle, viewcone_min=0.0, viewcone_max=10.0):
    """Write the metadata needed by directory particle discovery."""
    metadata = Table(
        {
            "primary_particle": [particle],
            "viewcone_min": [viewcone_min] * u.deg,
            "viewcone_max": [viewcone_max] * u.deg,
        }
    )
    metadata.meta["EXTNAME"] = derive_corsika_limits.TRIGGER_HISTOGRAM_METADATA_TABLE
    table_handler.write_tables(
        [metadata],
        file_path,
        overwrite_existing=True,
        file_type="HDF5",
    )


def _pool_result(
    production_index=0,
    array_name="LST",
    telescope_ids=None,
    lower_energy_limit=0.5 * u.TeV,
    upper_radius_limit=400.0 * u.m,
    viewcone_radius=5.0 * u.deg,
):
    """Build a standard mocked pool result row for grid execution tests."""
    return {
        "production_index": production_index,
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


def test_write_results(mocker, mock_args_dict, mock_results, tmp_test_directory, run_context):
    tmp_test_directory = Path(tmp_test_directory)
    run_context.paths.output_path["default"] = tmp_test_directory
    metadata = {"cta": {"activity": {"name": "production_derive_corsika_limits"}}}
    mock_metadata_collector = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits.MetadataCollector"
    )
    mock_metadata_collector.return_value.get_top_level_metadata.return_value = metadata

    derive_corsika_limits.write_results(
        mock_results, mock_args_dict, DEFAULT_ALLOWED_LOSSES, 0.1, run_context
    )

    output_file = tmp_test_directory / "corsika_limits.ecsv"
    output_table = Table.read(output_file, format="ascii.ecsv")
    assert output_table.meta["cta"] == metadata["cta"]
    assert not output_file.with_suffix(".meta.yml").exists()
    mock_metadata_collector.assert_called_once_with(mock_args_dict, context=run_context)


def test_generate_corsika_limits_grid_requires_trigger_histogram_file(mock_args_dict):
    args = mock_args_dict.copy()
    args["trigger_histogram_file"] = None

    args["trigger_histogram_directory"] = None

    with pytest.raises(ValueError, match="Use trigger_histogram_file"):
        derive_corsika_limits.generate_corsika_limits_grid(args)


def test_generate_corsika_limits_grid_from_trigger_histogram_file(
    mocker, mock_args_dict, tmp_test_directory, run_context
):
    args = mock_args_dict.copy()
    args["trigger_histogram_file"] = "trigger_histograms.hdf5"
    args["array_layout_name"] = ["alpha"]

    metadata = Table(
        rows=[
            {
                "production_index": 0,
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
    run_context.paths.output_path["default"] = Path(tmp_test_directory)

    derive_corsika_limits.generate_corsika_limits_grid(args, context=run_context)

    mock_load.assert_called_once_with("trigger_histograms.hdf5", array_names=["alpha"])
    mock_derive.assert_called_once()
    assert mock_derive.call_args.args[5] == tmp_test_directory
    result = mock_write.call_args[0][0][0]
    assert result["array_name"] == "alpha"
    assert result["telescope_ids"] == ["LSTN-01"]


def test_generate_corsika_limits_grid_uses_all_arrays_when_array_names_not_given(
    mocker, mock_args_dict, tmp_test_directory, run_context
):
    args = mock_args_dict.copy()
    args["trigger_histogram_file"] = "trigger_histograms.hdf5"
    args["array_layout_name"] = None

    metadata = Table(
        rows=[
            {
                "production_index": 0,
                "array_name": "alpha",
                "telescope_ids": "LSTN-01",
            },
            {
                "production_index": 0,
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
    run_context.paths.output_path["default"] = Path(tmp_test_directory)

    derive_corsika_limits.generate_corsika_limits_grid(args, context=run_context)

    mock_load.assert_called_once_with("trigger_histograms.hdf5", array_names=None)
    assert mock_derive.call_count == 2
    results = mock_write.call_args[0][0]
    assert [result["array_name"] for result in results] == ["alpha", "beta"]
    assert results[0]["telescope_ids"] == ["LSTN-01"]
    assert results[1]["telescope_ids"] == ["MSTS-01"]


def test_generate_corsika_limits_grid_skips_empty_energy_histograms(
    mocker, mock_args_dict, tmp_test_directory, run_context
):
    args = mock_args_dict.copy()
    args["trigger_histogram_file"] = "trigger_histograms.hdf5"

    metadata = Table(
        rows=[
            {
                "production_index": 0,
                "array_name": "empty",
                "telescope_ids": "LSTN-01",
            },
            {
                "production_index": 0,
                "array_name": "valid",
                "telescope_ids": "MSTS-01",
            },
        ]
    )
    empty_histograms = mocker.Mock()
    empty_histograms.histograms = {
        "energy": Histogram.from_edges([np.arange(3)], values=np.zeros(2))
    }
    valid_histograms = mocker.Mock()
    valid_histograms.histograms = {
        "energy": Histogram.from_edges([np.arange(3)], values=np.array([0.0, 1.0]))
    }
    mocker.patch(
        "simtools.production_configuration.derive_corsika_limits.load_event_data_histograms",
        return_value=[
            (metadata[0], empty_histograms),
            (metadata[1], valid_histograms),
        ],
    )
    mock_derive = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits._derive_limits_from_histograms",
        return_value=_pool_result(array_name="valid"),
    )
    mock_write = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits.write_results"
    )
    run_context.paths.output_path["default"] = Path(tmp_test_directory)

    derive_corsika_limits.generate_corsika_limits_grid(args, context=run_context)

    mock_derive.assert_called_once()
    assert mock_derive.call_args.args[1] == "valid"
    mock_write.assert_called_once()


def test_generate_corsika_limits_grid_plots_only_selected_array_layouts(
    mocker, mock_args_dict, tmp_test_directory, run_context
):
    args = mock_args_dict.copy()
    args["trigger_histogram_file"] = "trigger_histograms.hdf5"
    args["plot_histograms"] = ["alpha"]

    metadata = Table(
        rows=[
            {
                "production_index": 0,
                "array_name": "alpha",
                "telescope_ids": "LSTN-01",
            },
            {
                "production_index": 0,
                "array_name": "beta",
                "telescope_ids": "MSTS-01",
            },
        ]
    )
    mocker.patch(
        "simtools.production_configuration.derive_corsika_limits.load_event_data_histograms",
        return_value=[(metadata[0], mocker.Mock()), (metadata[1], mocker.Mock())],
    )
    mock_derive = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits._derive_limits_from_histograms",
        side_effect=[_pool_result(array_name="alpha"), _pool_result(array_name="beta")],
    )
    mocker.patch("simtools.production_configuration.derive_corsika_limits.write_results")
    run_context.paths.output_path["default"] = Path(tmp_test_directory)

    derive_corsika_limits.generate_corsika_limits_grid(args, context=run_context)

    assert [call.args[4] for call in mock_derive.call_args_list] == [True, False]


def test_generate_corsika_limits_grid_expands_trigger_histogram_glob(
    mocker, mock_args_dict, tmp_test_directory, run_context
):
    args = mock_args_dict.copy()
    first_file = Path(tmp_test_directory) / "electron_20.hdf5"
    second_file = Path(tmp_test_directory) / "electron_40.hdf5"
    first_file.touch()
    second_file.touch()
    args["trigger_histogram_file"] = str(Path(tmp_test_directory) / "electron*.hdf5")

    metadata = Table(
        rows=[
            {
                "production_index": 0,
                "array_name": "alpha",
                "telescope_ids": "LSTN-01",
            }
        ]
    )
    mock_load = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits.load_event_data_histograms",
        return_value=[(metadata[0], mocker.Mock())],
    )
    mocker.patch(
        "simtools.production_configuration.derive_corsika_limits._derive_limits_from_histograms",
        return_value=_pool_result(array_name="alpha"),
    )
    mock_write = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits.write_results"
    )
    run_context.paths.output_path["default"] = Path(tmp_test_directory)

    derive_corsika_limits.generate_corsika_limits_grid(args, context=run_context)

    assert sorted(mock_load.call_args_list, key=lambda call: call.args[0]) == [
        mocker.call(str(first_file), array_names=None),
        mocker.call(str(second_file), array_names=None),
    ]
    assert len(mock_write.call_args.args[0]) == 2


def test_generate_corsika_limits_grid_writes_one_file_per_particle(
    mocker, mock_args_dict, tmp_test_directory, run_context
):
    input_directory = Path(tmp_test_directory) / "trigger_histograms"
    input_directory.mkdir()
    _write_particle_metadata(input_directory / "electron_z20.trigger_histograms.hdf5", "electron")
    _write_particle_metadata(
        input_directory / "gamma-diffuse_z20.trigger_histograms.hdf5", "gamma", 0.0, 10.0
    )
    _write_particle_metadata(
        input_directory / "gamma_z20.trigger_histograms.hdf5", "gamma", 0.0, 0.0
    )

    args = mock_args_dict.copy()
    args["trigger_histogram_file"] = None
    args["trigger_histogram_directory"] = str(input_directory)
    args["output_file"] = "activity-id-simtools-production-derive-corsika-limits.ecsv"
    args["output_file_from_default"] = True

    run_context.paths.output_path["default"] = Path(tmp_test_directory) / "output"
    mock_generate = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits._generate_corsika_limits_from_histogram_file",
        return_value=[_pool_result()],
    )
    mock_write = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits.write_results"
    )

    derive_corsika_limits.generate_corsika_limits_grid(args, context=run_context)

    output_root = Path(tmp_test_directory) / "output"
    assert [call.kwargs["histogram_files"] for call in mock_generate.call_args_list] == [
        [str(input_directory / "electron_z20.trigger_histograms.hdf5")],
        [str(input_directory / "gamma-diffuse_z20.trigger_histograms.hdf5")],
        [str(input_directory / "gamma_z20.trigger_histograms.hdf5")],
    ]
    assert [call.args[1]["output_file"] for call in mock_write.call_args_list] == [
        str(output_root / "electron" / "corsika_limits.ecsv"),
        str(output_root / "gamma" / "corsika_limits.ecsv"),
        str(output_root / "gamma-0.00deg" / "corsika_limits.ecsv"),
    ]
    assert all((output_root / name).is_dir() for name in ("electron", "gamma", "gamma-0.00deg"))


def test_resolve_telescope_configs_wraps_single_layout_result(mocker):
    mock_resolve = mocker.patch(
        "simtools.production_configuration.production_event_data_helpers.resolve_array_layout_name",
        return_value="single-layout",
    )
    mock_repository_lookup = mocker.patch(
        (
            "simtools.production_configuration.production_event_data_helpers."
            "get_array_elements_from_model_repository"
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
    mock_repository_lookup.assert_called_once_with(
        ["single-layout"], "South", "1.0.0", context=None
    )
    assert result == {"LST": ["LSTN-01"]}


@pytest.mark.parametrize(
    ("allowed_losses", "error_match"),
    [
        (["core_distance,0.2"], "Expected format"),
        (["core_distance,abc,10"], "fraction must be float"),
        (["invalid,0.2,10"], "Invalid axis"),
        (["core_distance,0.2,10"], "Missing --allowed_losses entries"),
        (["core_distance,-0.1,10", "angular_distance,0.2,10"], r"interval \[0, 1\]"),
        (["core_distance,1.1,10", "angular_distance,0.2,10"], r"interval \[0, 1\]"),
        (["core_distance,nan,10", "angular_distance,0.2,10"], "finite"),
        (["core_distance,0.2,-1", "angular_distance,0.2,10"], "non-negative integer"),
    ],
)
def test_parse_allowed_losses_error_paths(allowed_losses, error_match):
    with pytest.raises(ValueError, match=error_match):
        derive_corsika_limits.parse_allowed_losses(allowed_losses)


def test_parse_allowed_losses_raises_when_not_provided():
    with pytest.raises(ValueError, match="No allowed-loss configuration provided"):
        derive_corsika_limits.parse_allowed_losses(None)


def test_compute_lower_energy_limit(mocker):
    mock_hist = np.array([1.0, 12.0, 20.0, 12.0, 1.0])
    mock_bins = np.logspace(-3, 3, 6)

    # Mock the histograms object
    mock_histograms = mocker.MagicMock()
    mock_histograms.histograms = {"energy": Histogram.from_edges([mock_bins], values=mock_hist)}
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
    mock_hist = np.array([1.0, 12.0, 20.0, 12.0, 1.0])
    mock_bins = np.array([0.01, 0.02, 0.04, 0.08, 0.16, 0.32])

    mock_histograms = mocker.MagicMock()
    mock_histograms.histograms = {"energy": Histogram.from_edges([mock_bins], values=mock_hist)}
    mock_histograms.energy_bins = mock_bins
    mock_histograms.file_info = {"energy_min": 0.014 * u.TeV}

    result = derive_corsika_limits.compute_lower_energy_limit(mock_histograms, 0.2)

    assert_quantity_allclose(result, 0.014 * u.TeV)


def test_apply_broad_range_lower_energy_floor_uses_enforced_minimum_for_different_bins():
    derived = 0.020 * u.TeV
    broad_range_min = 0.030 * u.TeV
    energy_bins = np.array([0.01, 0.02, 0.03, 0.04])

    result = derive_corsika_limits._apply_broad_range_lower_energy_floor(
        derived,
        broad_range_min,
        energy_bins,
    )

    assert_quantity_allclose(result, 0.030 * u.TeV)


def test_find_low_energy_threshold_from_histogram_peak_at_first_bin():
    counts = np.array([10.0, 4.0, 1.0, 0.0])
    bin_edges = np.array([0.05, 0.1, 0.2, 0.4, 0.8])

    # No bins left of peak; fallback to first edge is expected.
    result = derive_corsika_limits._find_low_energy_threshold_from_histogram(counts, bin_edges)
    assert result == pytest.approx(0.05)


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
    with pytest.raises(ValueError, match=error_match):
        derive_corsika_limits._find_low_energy_threshold_from_histogram(
            counts,
            bin_edges,
            threshold_fraction=threshold_fraction,
        )


def test_find_low_energy_threshold_from_histogram_raises_for_all_zero_counts():
    counts = np.array([0.0, 0.0, 0.0, 0.0])
    bin_edges = np.array([0.1, 0.2, 0.4, 0.8, 1.6])

    with pytest.raises(ValueError, match="at least one positive entry"):
        derive_corsika_limits._find_low_energy_threshold_from_histogram(counts, bin_edges)


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
    histograms = mocker.MagicMock()
    histograms.energy_bins = np.array([1.0, 10.0])
    histograms.core_distance_bins = np.array([0.0, 100.0])
    histograms.view_cone_bins = np.array([0.0, 5.0])
    histograms.histograms = {
        "core_distance_vs_energy": Histogram.from_edges([[0, 100], [1, 10]], values=[[1]]),
        "angular_distance_vs_energy": Histogram.from_edges([[0, 5], [1, 10]], values=[[2]]),
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
    np.testing.assert_array_equal(mock_diff_limits.call_args_list[0].args[0], [[1]])
    assert mock_diff_limits.call_args_list[0].args[5:] == ("core_scatter", "m")
    np.testing.assert_array_equal(mock_diff_limits.call_args_list[1].args[0], [[2]])
    assert mock_diff_limits.call_args_list[1].args[5:] == ("viewcone", "deg")
    assert mock_diff_limits.call_args_list[0].args[4] == DEFAULT_ALLOWED_LOSSES["core_distance"]
    assert mock_diff_limits.call_args_list[1].args[4] == DEFAULT_ALLOWED_LOSSES["angular_distance"]

    assert mock_is_close.call_args_list[0].args[0].value == pytest.approx(120.0)
    assert mock_is_close.call_args_list[0].args[1] == expected_core_scatter_max
    assert mock_is_close.call_args_list[1].args[0].value == pytest.approx(3.0)
    assert mock_is_close.call_args_list[1].args[1] == expected_viewcone_max


def test_compute_limits_with_integral_fallback_curves(mocker):
    histograms = mocker.MagicMock()
    histograms.energy_bins = np.array([1.0, 10.0])
    histograms.core_distance_bins = np.array([0.0, 100.0])
    histograms.view_cone_bins = np.array([0.0, 5.0])
    histograms.histograms = {
        "core_distance": Histogram.from_edges([np.arange(3)], values=np.array([1.0, 2.0])),
        "angular_distance": Histogram.from_edges([np.arange(3)], values=np.array([3.0, 4.0])),
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
    histograms = mocker.MagicMock()
    histograms.energy_bins = np.array([1.0, 10.0])
    histograms.core_distance_bins = np.array([0.0, 100.0])
    histograms.view_cone_bins = np.array([0.0, 0.5])
    histograms.data_ranges = {"angular_distance": (0.0, 0.0)}
    histograms.histograms = {
        "core_distance": Histogram.from_edges([np.arange(2)], values=np.array([10.0])),
        "angular_distance": Histogram.from_edges([np.arange(2)], values=np.array([10.0])),
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


def test_constant_angular_distance_is_not_rounded_in_results_table(mock_results):
    mock_results[0]["viewcone_radius"] = 0.1 * u.deg
    mock_results[0]["angular_distance_is_constant"] = True

    table = derive_corsika_limits._create_results_table(mock_results, DEFAULT_ALLOWED_LOSSES, 0.1)

    assert table["viewcone_radius"][0] == pytest.approx(0.1)


def test_create_results_table_sorts_production_configuration_columns():
    results = [
        _pool_result(production_index=1, array_name="beta"),
        _pool_result(production_index=0, array_name="alpha"),
        _pool_result(production_index=0, array_name="beta"),
        _pool_result(production_index=0, array_name="alpha"),
        _pool_result(production_index=0, array_name="alpha"),
    ]
    results[1].update({"primary_particle": "proton", "zenith": 20.0 * u.deg})
    results[2].update({"primary_particle": "electron", "zenith": 20.0 * u.deg})
    results[3].update({"primary_particle": "electron", "zenith": 40.0 * u.deg})
    results[4].update({"primary_particle": "electron", "zenith": 20.0 * u.deg})

    table = derive_corsika_limits._create_results_table(results, DEFAULT_ALLOWED_LOSSES, 0.1)

    assert list(
        zip(
            table["production_index"],
            table["primary_particle"],
            table["array_name"],
            table["zenith"].quantity.to_value(u.deg),
            strict=True,
        )
    ) == [
        (0, "electron", "alpha", 20.0),
        (0, "electron", "alpha", 40.0),
        (0, "electron", "beta", 20.0),
        (0, "proton", "alpha", 20.0),
        (1, "gamma", "beta", 20.0),
    ]


def test_constant_angular_distance_distributions_are_not_plotted(mocker, tmp_test_directory):
    histograms = mocker.MagicMock()
    histograms.file_info = {}
    histograms.histograms = {
        "energy": Histogram.from_edges([np.arange(2)], values=np.array([1.0])),
        "angular_distance_vs_energy": Histogram.from_edges(
            [np.arange(2), np.arange(2)], values=np.array([[1.0]])
        ),
        "angular_distance_vs_energy_mc": Histogram.from_edges(
            [np.arange(2), np.arange(2)], values=np.array([[1.0]])
        ),
        "angular_distance_vs_energy_cumulative": Histogram.from_edges(
            [np.arange(2), np.arange(2)], values=np.array([[1.0]])
        ),
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


def test_reduced_histogram_plot_selection(mocker, tmp_test_directory):
    reduced_histogram_names = {
        "angular_distance_vs_energy",
        "core_distance_vs_energy",
    }
    histograms = mocker.MagicMock()
    histograms.file_info = {}
    histograms.histograms = {
        name: Histogram.from_edges([np.arange(2)], values=np.array([1.0]))
        for name in reduced_histogram_names
    }
    histograms.histograms.update(
        {
            "energy": Histogram.from_edges([np.arange(2)], values=np.array([1.0])),
            "core_distance": Histogram.from_edges([np.arange(2)], values=np.array([1.0])),
            "reuse_max_vs_core_distance_vs_energy": Histogram.from_edges(
                [np.arange(2)], values=np.array([1.0])
            ),
            "x_core_shower_vs_y_core_shower": Histogram.from_edges(
                [np.arange(2)], values=np.array([1.0])
            ),
        }
    )
    mocker.patch(COMPUTE_LOWER_ENERGY_LIMIT_PATH, return_value=1.0 * u.TeV)
    mocker.patch(COMPUTE_LIMITS_PATH, return_value={})
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
        True,
    )

    assert set(plot.call_args.args[0]) == reduced_histogram_names


def test_differential_upper_limits(mocker):
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


def test_parse_allowed_losses_all_and_override():
    result = derive_corsika_limits.parse_allowed_losses(
        [
            "all,1e-6,10",
            "core_distance,5e-7,5",
        ]
    )

    assert result["core_distance"]["loss_fraction"] == pytest.approx(5e-7)
    assert result["core_distance"]["loss_min_events"] == 5
    assert result["angular_distance"]["loss_fraction"] == pytest.approx(1e-6)
    assert result["angular_distance"]["loss_min_events"] == 10


@pytest.mark.parametrize("value", [-1, "-1", "invalid"])
def test_validate_differential_loss_bins_per_decade_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="non-negative integer"):
        derive_corsika_limits.validate_differential_loss_bins_per_decade(value)


def test_validate_differential_loss_bins_per_decade_accepts_zero_and_positive_values():
    assert derive_corsika_limits.validate_differential_loss_bins_per_decade(0) == 0
    assert derive_corsika_limits.validate_differential_loss_bins_per_decade("5") == 5


@pytest.fixture
def mock_args_dict():
    """Fixture to provide mock arguments dictionary with required keys."""
    return {
        "config_file": "dummy_config.yml",
        "steps": None,
        "ignore_runtime_environment": False,
        "trigger_histogram_file": "trigger_histograms.hdf5",
        "array_layout_name": None,
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
