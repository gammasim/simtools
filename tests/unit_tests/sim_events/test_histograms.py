import logging

import astropy.units as u
import numpy as np
import pytest

from simtools.histogram.histogram import Histogram
from simtools.sim_events.histograms import EventDataHistograms


@pytest.fixture
def mock_reader(mocker):
    mocker.patch(
        "simtools.sim_events.histograms.resolve_file_patterns",
        side_effect=lambda file_names: file_names if isinstance(file_names, list) else [file_names],
    )
    mock = mocker.patch("simtools.sim_events.histograms.EventDataReader")
    mock.return_value.triggered_shower_data.simulated_energy = np.array([1, 10, 100])
    mock.return_value.shower_data.simulated_energy = np.array([1, 10, 100])
    mock.return_value.triggered_data = mocker.Mock()
    mock.return_value.triggered_data.angular_distance = np.array([0.5, 1.0, 1.5])
    return mock


@pytest.fixture
def hdf5_file_name():
    return "test_file.h5"


@pytest.fixture
def reduced_file_info():
    """Common reduced file info mock for fill() tests."""
    return {
        "primary_particle": "proton",
        "zenith": 20.0 * u.deg,
        "azimuth": 0.0 * u.deg,
        "nsb_level": 1.0,
        "energy_min": 0.1 * u.TeV,
        "core_scatter_max": 100.0 * u.m,
        "viewcone_max": 2.0 * u.deg,
        "solid_angle": 1.0 * u.sr,
        "scatter_area": 1.0 * u.cm**2,
    }


def test_init(mock_reader, hdf5_file_name):
    test_telescope_list = [1, 2]
    histograms = EventDataHistograms(hdf5_file_name, "test_array", test_telescope_list)

    assert histograms.event_data_file == hdf5_file_name
    mock_reader.assert_called_once_with(hdf5_file_name, telescope_list=test_telescope_list)


def test_energy_bins(mock_reader, hdf5_file_name):
    histograms = EventDataHistograms(hdf5_file_name)
    mock_reader.return_value.triggered_shower_data.simulated_energy = np.array([1, 10, 100])
    bins = histograms.energy_bins
    assert isinstance(bins, np.ndarray)
    assert len(bins) == 61
    assert bins[0] == pytest.approx(1.0e-3)
    assert bins[-1] == pytest.approx(1.0e3)
    assert np.allclose(np.diff(np.log10(bins)), 0.1)


def test_core_distance_bins(mock_reader, hdf5_file_name):
    histograms = EventDataHistograms(hdf5_file_name)
    mock_reader.return_value.triggered_shower_data.core_distance_shower = np.array([10, 20, 30])
    bins = histograms.core_distance_bins
    assert isinstance(bins, np.ndarray)
    assert len(bins) == 100


def test_core_distance_bins_use_fixed_width_and_flexible_range():
    histograms = EventDataHistograms.create_accumulator(core_distance_bin_width=20.0 * u.m)
    histograms.file_info = {
        "core_scatter_min": 15.0 * u.m,
        "core_scatter_max": 65.0 * u.m,
    }

    np.testing.assert_allclose(histograms.core_distance_bins, [0.0, 20.0, 40.0, 60.0, 80.0])


def test_core_distance_bin_width_must_be_positive():
    with pytest.raises(ValueError, match="core_distance_bin_width must be positive"):
        EventDataHistograms.create_accumulator(core_distance_bin_width=0.0 * u.m)


def test_view_cone_bins(mock_reader, hdf5_file_name):
    histograms = EventDataHistograms(hdf5_file_name)
    mock_reader.return_value.triggered_data.angular_distance = np.array([0.5, 1.0, 1.5])
    bins = histograms.view_cone_bins
    assert isinstance(bins, np.ndarray)
    assert len(bins) == 100


def test_release_event_data_handles_non_1d_none_event_data(mock_reader, hdf5_file_name):
    histograms = EventDataHistograms(hdf5_file_name)
    histograms._histogram_specs = {
        "energy": {"1d": True, "event_data": mock_reader.return_value},
        "reuse_mean_vs_core_distance_vs_energy": {"1d": False, "event_data": None},
        "core_distance_vs_energy": {
            "1d": False,
            "event_data": (mock_reader.return_value, mock_reader.return_value),
        },
    }

    histograms._release_event_data()

    assert histograms._histogram_specs["energy"]["event_data"] is None
    assert (
        histograms._histogram_specs["reuse_mean_vs_core_distance_vs_energy"]["event_data"] is None
    )
    assert histograms._histogram_specs["core_distance_vs_energy"]["event_data"] == (None, None)


def test_fill(mock_reader, hdf5_file_name, mocker):
    histograms = EventDataHistograms(hdf5_file_name)

    mock_file_info = mocker.Mock()
    mock_event_data = mocker.Mock()
    mock_triggered_data = mocker.Mock()
    mock_shower_data = mocker.Mock()

    mock_reader.return_value.read_event_data.return_value = (
        mock_file_info,
        mock_shower_data,
        mock_event_data,
        mock_triggered_data,
    )

    # Event (triggered) data
    mock_event_data.simulated_energy = np.array([1, 2, 3])
    mock_event_data.core_distance_shower = np.array([10, 20, 30])
    mock_event_data.x_core_shower = np.array([-10, 0, 10])
    mock_event_data.y_core_shower = np.array([-5, 0, 5])
    mock_triggered_data.angular_distance = np.array([0.5, 1.0, 1.5])

    # Shower (MC truth) data
    mock_shower_data.simulated_energy = np.array([1, 2, 3])
    mock_shower_data.core_distance_shower = np.array([10, 20, 30])
    mock_shower_data.x_core_shower = np.array([-10, 0, 10])
    mock_shower_data.y_core_shower = np.array([-5, 0, 5])
    mock_shower_data.angular_distance = np.array([0.5, 1.0, 1.5])

    mock_reader.return_value.data_sets = ["test_dataset"]

    # Patch _fill_histogram_and_bin_edges to simulate filling histograms with required structure
    def fake_fill_histogram_and_bin_edges(name, data):
        histogram = histograms._create_histogram(data)
        histogram.values = np.ones(histogram.shape)
        histograms.histograms[name] = histogram

    mocker.patch.object(
        histograms, "_fill_histogram_and_bin_edges", side_effect=fake_fill_histogram_and_bin_edges
    )

    mock_energy_bins = np.array([0, 2, 4])
    mock_core_distance_bins = np.array([0, 20, 40])
    mock_view_cone_bins = np.array([0, 1, 2])

    mocker.patch.object(EventDataHistograms, "energy_bins", property(lambda self: mock_energy_bins))
    mocker.patch.object(
        EventDataHistograms,
        "core_distance_bins",
        property(lambda self: mock_core_distance_bins),
    )
    mocker.patch.object(
        EventDataHistograms, "view_cone_bins", property(lambda self: mock_view_cone_bins)
    )

    histograms.fill()

    mock_reader.return_value.read_event_data.assert_called_once_with(
        hdf5_file_name, table_name_map="test_dataset"
    )


def test_fill_populates_primary_particle(mock_reader, hdf5_file_name, mocker, reduced_file_info):
    histograms = EventDataHistograms(hdf5_file_name)

    mock_reader.return_value.data_sets = ["test_dataset"]
    mock_reader.return_value.read_event_data.return_value = (
        mocker.Mock(),
        mocker.Mock(),
        mocker.Mock(),
        mocker.Mock(),
    )
    mock_reader.return_value.get_reduced_simulation_file_info.return_value = reduced_file_info
    mocker.patch.object(histograms, "_define_histograms", return_value={})
    mocker.patch.object(histograms, "print_summary")
    mocker.patch.object(histograms, "calculate_efficiency_data")
    mocker.patch.object(histograms, "calculate_cumulative_data")

    histograms.fill()

    assert histograms.file_info["primary_particle"] == "proton"


def test_fill_reads_multiple_files_sequentially(mock_reader, mocker):
    histograms = EventDataHistograms(["test_file_1.h5", "test_file_2.h5"])

    mock_file_info = mocker.Mock()
    mock_event_data = mocker.Mock()
    mock_triggered_data = mocker.Mock()
    mock_shower_data = mocker.Mock()

    mock_reader.return_value.read_event_data.return_value = (
        mock_file_info,
        mock_shower_data,
        mock_event_data,
        mock_triggered_data,
    )
    mock_reader.return_value.get_reduced_simulation_file_info.return_value = {
        "energy_min": 1.0 * u.TeV,
        "core_scatter_max": 100.0 * u.m,
        "viewcone_max": 5.0 * u.deg,
        "solid_angle": 1.0 * u.sr,
        "scatter_area": 1.0 * u.cm**2,
    }
    mock_reader.return_value.data_sets = ["test_dataset"]
    mocker.patch.object(histograms, "_define_histograms", return_value={})
    mocker.patch.object(histograms, "print_summary")
    mocker.patch.object(histograms, "calculate_efficiency_data")
    mocker.patch.object(histograms, "calculate_cumulative_data")

    histograms.fill()

    opened_files = [call.args[0] for call in mock_reader.call_args_list]
    assert set(opened_files) >= {"test_file_1.h5", "test_file_2.h5"}
    assert all(call.kwargs == {"telescope_list": None} for call in mock_reader.call_args_list)

    read_calls = mock_reader.return_value.read_event_data.call_args_list
    assert len(read_calls) == 2
    assert all(call.kwargs.get("table_name_map") == "test_dataset" for call in read_calls)


@pytest.mark.parametrize("exception", [OSError("bad file"), KeyError("missing dataset")])
def test_iter_readers_skips_unreadable_files_when_enabled(mocker, exception):
    mocker.patch(
        "simtools.sim_events.histograms.resolve_file_patterns",
        return_value=["bad_file.h5"],
    )
    mocker.patch("simtools.sim_events.histograms.EventDataReader", side_effect=exception)

    histograms = EventDataHistograms("*.h5", skip_invalid_event_data_files=True)

    assert list(histograms._iter_readers()) == []


@pytest.mark.parametrize("exception", [OSError("bad file"), KeyError("missing dataset")])
def test_fill_skips_read_errors_when_enabled(mock_reader, mocker, hdf5_file_name, exception):
    histograms = EventDataHistograms(hdf5_file_name, skip_invalid_event_data_files=True)
    mock_reader.return_value.data_sets = ["test_dataset"]
    mock_reader.return_value.read_event_data.side_effect = exception
    mocker.patch.object(histograms, "print_summary")
    mocker.patch.object(histograms, "calculate_efficiency_data")
    mocker.patch.object(histograms, "calculate_cumulative_data")

    with pytest.raises(ValueError, match="No readable event data files or datasets found"):
        histograms.fill()

    mock_reader.return_value.read_event_data.assert_called_once_with(
        hdf5_file_name, table_name_map="test_dataset"
    )


def test_fill_accumulates_histograms_across_data_sets(
    mock_reader, hdf5_file_name, mocker, reduced_file_info
):
    histograms = EventDataHistograms(hdf5_file_name)

    mock_reader.return_value.data_sets = ["dataset_0", "dataset_1"]
    mock_reader.return_value.read_event_data.return_value = (
        mocker.Mock(),
        mocker.Mock(),
        mocker.Mock(),
        mocker.Mock(),
    )
    mock_reader.return_value.get_reduced_simulation_file_info.return_value = reduced_file_info

    dataset_0 = mocker.Mock()
    dataset_0.simulated_energy = np.array([0.2, 0.4])
    dataset_1 = mocker.Mock()
    dataset_1.simulated_energy = np.array([1.4])

    mocker.patch.object(
        histograms,
        "_define_histograms",
        side_effect=[
            {
                "energy": {
                    "1d": True,
                    "event_data": dataset_0,
                    "event_data_column": "simulated_energy",
                    "edges": [np.array([0.0, 1.0, 2.0])],
                    "histogram": None,
                }
            },
            {
                "energy": {
                    "1d": True,
                    "event_data": dataset_1,
                    "event_data_column": "simulated_energy",
                    "edges": [np.array([0.0, 1.0, 2.0])],
                    "histogram": None,
                }
            },
        ],
    )

    histograms.fill()

    np.testing.assert_array_equal(histograms.histograms["energy"].values, np.array([2, 1]))
    assert mock_reader.return_value.read_event_data.call_count == 2


def test_update_file_info_rejects_mixed_exposure_metadata(
    mock_reader, hdf5_file_name, reduced_file_info
):
    histograms = EventDataHistograms(hdf5_file_name)
    histograms._update_file_info(reduced_file_info, source_file="first.h5")

    different_info = dict(reduced_file_info, scatter_area=2.0 * u.cm**2)
    with pytest.raises(ValueError, match=r"first\.h5.*second\.h5.*scatter_area"):
        histograms._update_file_info(different_info, source_file="second.h5")


def test_merge_histograms_rejects_different_bin_edges(mock_reader, hdf5_file_name):
    histograms = EventDataHistograms(hdf5_file_name)
    histograms.histograms = {
        "energy": Histogram.from_edges([np.array([0.0, 1.0, 2.0])], values=np.array([1, 2]))
    }

    with pytest.raises(ValueError, match="different bins"):
        histograms._merge_histograms(
            {"energy": {"edges": [np.array([0.0, 2.0, 4.0])], "histogram": None}}
        )


def test_triggered_reuse_counts_repeat_per_triggered_event(mock_reader, hdf5_file_name, mocker):
    histograms = EventDataHistograms(hdf5_file_name)
    triggered_data = mocker.Mock(
        file_id=np.array([0, 0, 0]),
        shower_id=np.array([1, 1, 2]),
    )

    reuse_counts = histograms._triggered_reuse_counts(triggered_data)

    np.testing.assert_array_equal(reuse_counts, np.array([2.0, 2.0, 1.0]))


def test_accumulate_reuse_bin_values_vectorized_1d_and_2d():
    accumulator_1d = {
        "count": np.zeros(3, dtype=int),
        "sum": np.zeros(3, dtype=float),
        "sum_sq": np.zeros(3, dtype=float),
        "max": np.full(3, np.nan, dtype=float),
    }
    EventDataHistograms._accumulate_reuse_bin_values(
        accumulator_1d,
        np.array([0, 0, 1, 2, 2]),
        np.array([2.0, 0.0, -1.0, 1.0, 3.0]),
    )
    np.testing.assert_array_equal(accumulator_1d["count"], np.array([1, 0, 2]))
    np.testing.assert_allclose(accumulator_1d["sum"], np.array([2.0, 0.0, 4.0]))
    np.testing.assert_allclose(accumulator_1d["sum_sq"], np.array([4.0, 0.0, 10.0]))
    np.testing.assert_allclose(accumulator_1d["max"], np.array([2.0, np.nan, 3.0]), equal_nan=True)

    accumulator_2d = {
        "count": np.zeros((2, 3), dtype=int),
        "sum": np.zeros((2, 3), dtype=float),
        "sum_sq": np.zeros((2, 3), dtype=float),
        "max": np.full((2, 3), np.nan, dtype=float),
    }
    EventDataHistograms._accumulate_reuse_bin_values(
        accumulator_2d,
        np.array([[0, 0], [0, 0], [0, 1], [1, 2], [1, 2], [1, 0]]),
        np.array([1.0, 4.0, 0.0, -2.0, 5.0, 3.0]),
    )
    np.testing.assert_array_equal(accumulator_2d["count"], np.array([[2, 0, 0], [1, 0, 1]]))
    np.testing.assert_allclose(accumulator_2d["sum"], np.array([[5.0, 0.0, 0.0], [3.0, 0.0, 5.0]]))
    np.testing.assert_allclose(
        accumulator_2d["sum_sq"], np.array([[17.0, 0.0, 0.0], [9.0, 0.0, 25.0]])
    )
    np.testing.assert_allclose(
        accumulator_2d["max"],
        np.array([[4.0, np.nan, np.nan], [3.0, np.nan, 5.0]]),
        equal_nan=True,
    )


def test_accumulate_reuse_histograms_match_triggered_event_phase_space(
    mock_reader, hdf5_file_name, mocker
):
    histograms = EventDataHistograms.create_accumulator(
        energy_bins_per_decade=1,
        angular_distance_bin_count=3,
        core_distance_bin_count=3,
    )
    histograms.file_info = {
        "energy_min": 0.1 * u.TeV,
        "energy_max": 100.0 * u.TeV,
        "core_scatter_min": 0.0 * u.m,
        "core_scatter_max": 60.0 * u.m,
        "viewcone_min": 0.0 * u.deg,
        "viewcone_max": 0.6 * u.deg,
    }
    histograms._histogram_specs = histograms._define_histograms(None, None, None)

    event_data = mocker.Mock(
        file_id=np.array([0, 0, 0]),
        shower_id=np.array([1, 1, 2]),
        simulated_energy=np.array([0.2, 0.2, 2.0]),
        core_distance_shower=np.array([10.0, 20.0, 10.0]),
    )
    triggered_data = mocker.Mock(
        file_id=np.array([0, 0, 0]),
        shower_id=np.array([1, 1, 2]),
        angular_distance=np.array([0.1, 0.2, 0.1]),
    )

    histograms._accumulate_reuse_histograms(event_data, triggered_data)

    np.testing.assert_allclose(
        histograms.histograms["reuse_mean_vs_energy"].values,
        np.array([2.0, 1.0, np.nan]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        histograms.histograms["reuse_max_vs_energy"].values,
        np.array([2.0, 1.0, np.nan]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        histograms.histograms["reuse_std_vs_energy"].values,
        np.array([0.0, 0.0, np.nan]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        histograms.histograms["reuse_mean_vs_core_distance"].values,
        np.array([5.0 / 3.0, np.nan]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        histograms.histograms["reuse_max_vs_core_distance"].values,
        np.array([2.0, np.nan]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        histograms.histograms["reuse_std_vs_core_distance"].values,
        np.array([np.sqrt(2.0 / 9.0), np.nan]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        histograms.histograms["reuse_mean_vs_angular_distance"].values,
        np.array([5.0 / 3.0, np.nan]),
        equal_nan=True,
    )
    expected_2d_mean = np.array([[2.0, 1.0, np.nan], [np.nan, np.nan, np.nan]])
    expected_2d_std = np.array([[0.0, 0.0, np.nan], [np.nan, np.nan, np.nan]])
    np.testing.assert_allclose(
        histograms.histograms["reuse_mean_vs_core_distance_vs_energy"].values,
        expected_2d_mean,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        histograms.histograms["reuse_std_vs_core_distance_vs_energy"].values,
        expected_2d_std,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        histograms.histograms["reuse_max_vs_angular_distance_vs_energy"].values,
        np.array([[2.0, 1.0, np.nan], [np.nan, np.nan, np.nan]]),
        equal_nan=True,
    )


@pytest.fixture
def mock_histograms(mocker):
    """Create a mocked EventDataHistograms that doesn't require a file."""
    mocker.patch(
        "simtools.sim_events.histograms.resolve_file_patterns",
        side_effect=lambda file_names: file_names if isinstance(file_names, list) else [file_names],
    )
    mocker.patch("simtools.sim_events.histograms.EventDataReader")
    return EventDataHistograms("dummy_file.h5", "test_array")


def test_core_distance_bins_with_existing_edges(mock_reader, hdf5_file_name):
    histograms = EventDataHistograms(hdf5_file_name)
    mock_edges = np.linspace(0, 1000, 50)
    histograms.loaded_bin_edges["core_distance"] = mock_edges

    bins = histograms.core_distance_bins

    assert isinstance(bins, np.ndarray)
    assert len(bins) == 50
    assert np.array_equal(bins, mock_edges)


def test_view_cone_bins_from_bin_width(mock_reader, hdf5_file_name):
    histograms = EventDataHistograms(hdf5_file_name, angular_distance_bin_width=0.5 * u.deg)
    histograms.file_info = {
        "viewcone_min": 0.25 * u.deg,
        "viewcone_max": 1.5 * u.deg,
    }

    np.testing.assert_allclose(histograms.view_cone_bins, [0.25, 0.75, 1.25, 1.5])


def test_view_cone_bins_with_histogram_data(mock_reader, hdf5_file_name):
    histograms = EventDataHistograms(hdf5_file_name)
    mock_bins = np.linspace(0.0, 5.0, 50)
    histograms.loaded_bin_edges["viewcone"] = mock_bins
    bins = histograms.view_cone_bins
    assert isinstance(bins, np.ndarray)
    assert len(bins) == 50
    assert np.array_equal(bins, mock_bins)


def test_calculate_cumulative_data(mock_reader, hdf5_file_name):
    histograms = EventDataHistograms(hdf5_file_name)
    histograms.reader.data_sets = [{"TRIGGERS": "TRIGGERS"}]

    # Seed the public histogram mapping with values and bin edges.
    histograms.histograms = {
        "energy": Histogram.from_edges([np.arange(5)], values=np.array([10, 20, 30, 40])),
        "core_distance": Histogram.from_edges([np.arange(5)], values=np.array([5, 15, 25, 35])),
        "angular_distance": Histogram.from_edges([np.arange(5)], values=np.array([2, 4, 6, 8])),
        "core_distance_vs_energy": Histogram.from_edges(
            [np.arange(3), np.arange(3)], values=np.array([[1, 2], [3, 4]])
        ),
        "angular_distance_vs_energy": Histogram.from_edges(
            [np.arange(3), np.arange(3)], values=np.array([[2, 3], [4, 5]])
        ),
    }
    cumulative_data = histograms.calculate_cumulative_data()
    # Expected 1D cumulative distributions
    expected_cumulative_energy = np.array([100, 90, 70, 40])  # reverse cumulative
    expected_cumulative_core_distance = np.array([5, 20, 45, 80])
    expected_cumulative_angular_distance = np.array([2, 6, 12, 20])
    # Expected normalized cumulative for 2D histograms along axis=0 (column-wise)
    expected_norm_core_distance_vs_energy = np.array([[1 / 4, 2 / 6], [1.0, 1.0]])
    expected_norm_ang_vs_energy = np.array([[2 / 6, 3 / 8], [1.0, 1.0]])
    assert set(cumulative_data.keys()) == {
        "core_distance_cumulative",
        "angular_distance_cumulative",
        "angular_distance_vs_energy_cumulative",
        "core_distance_vs_energy_cumulative",
        "energy_cumulative",
    }
    np.testing.assert_array_equal(
        cumulative_data["energy_cumulative"].values, expected_cumulative_energy
    )
    np.testing.assert_array_equal(
        cumulative_data["core_distance_cumulative"].values,
        expected_cumulative_core_distance,
    )
    np.testing.assert_array_equal(
        cumulative_data["angular_distance_cumulative"].values,
        expected_cumulative_angular_distance,
    )
    np.testing.assert_allclose(
        cumulative_data["core_distance_vs_energy_cumulative"].values,
        expected_norm_core_distance_vs_energy,
    )
    np.testing.assert_allclose(
        cumulative_data["angular_distance_vs_energy_cumulative"].values,
        expected_norm_ang_vs_energy,
    )


def test_calculate_efficiency_data(mock_reader, hdf5_file_name):
    histograms = EventDataHistograms(hdf5_file_name)
    histograms.reader.data_sets = [{"TRIGGERS": "TRIGGERS"}]

    # Mock histograms for triggered and simulated events.
    histograms.histograms = {
        "energy": Histogram.from_edges([np.arange(4)], values=np.array([10, 20, 30])),
        "energy_mc": Histogram.from_edges([np.arange(4)], values=np.array([20, 40, 60])),
        "core_distance": Histogram.from_edges([np.arange(4)], values=np.array([5, 10, 15])),
        "core_distance_mc": Histogram.from_edges([np.arange(4)], values=np.array([10, 20, 30])),
        "angular_distance": Histogram.from_edges([np.arange(4)], values=np.array([0, 5, 10])),
        "angular_distance_mc": Histogram.from_edges([np.arange(4)], values=np.array([0, 10, 20])),
    }
    # Call the method
    efficiency_data = histograms.calculate_efficiency_data()
    # Check efficiency histograms
    assert "energy_eff" in efficiency_data
    assert "core_distance_eff" in efficiency_data
    assert "angular_distance_eff" in efficiency_data
    np.testing.assert_array_almost_equal(
        efficiency_data["energy_eff"].values, np.array([0.5, 0.5, 0.5])
    )
    np.testing.assert_array_almost_equal(
        efficiency_data["core_distance_eff"].values, np.array([0.5, 0.5, 0.5])
    )
    np.testing.assert_array_almost_equal(
        efficiency_data["angular_distance_eff"].values, np.array([0.0, 0.5, 0.5])
    )


def test_calculate_efficiency_data_shape_mismatch(mock_reader, hdf5_file_name, caplog):
    histograms = EventDataHistograms(hdf5_file_name)
    histograms.reader.data_sets = [{"TRIGGERS": "TRIGGERS"}]
    histograms.histograms = {
        "energy": Histogram.from_edges([np.arange(3)], values=np.array([10, 20])),
        "energy_mc": Histogram.from_edges([np.arange(4)], values=np.array([20, 40, 60])),
    }
    with caplog.at_level(logging.WARNING):
        efficiency_data = histograms.calculate_efficiency_data()
    # Ensure no efficiency histogram is created
    assert "energy_eff" not in efficiency_data
    # Check for warning message
    assert any(
        "Shape mismatch for energy and energy_mc, skipping efficiency calculation."
        in record.message
        for record in caplog.records
    )


def test_calculate_efficiency_data_missing_histograms(mock_reader, hdf5_file_name):
    histograms = EventDataHistograms(hdf5_file_name)
    histograms.reader.data_sets = [{"TRIGGERS": "TRIGGERS"}]

    # Mock histograms with missing triggered histogram (as dict)
    histograms.histograms = {
        "energy_mc": Histogram.from_edges([np.arange(4)], values=np.array([20, 40, 60])),
    }
    efficiency_data = histograms.calculate_efficiency_data()
    # Ensure no efficiency histogram is created
    assert "energy_eff" not in efficiency_data


def test_energy_bins_with_histogram_edges(mock_reader, hdf5_file_name):
    histograms = EventDataHistograms(hdf5_file_name)
    mock_edges = np.linspace(0, 10, 5)
    histograms.loaded_bin_edges["energy"] = mock_edges
    bins = histograms.energy_bins
    assert np.array_equal(bins, mock_edges)


def test_view_cone_bins_min_equals_max(mock_reader, hdf5_file_name):
    histograms = EventDataHistograms(hdf5_file_name)
    histograms.file_info = {
        "viewcone_min": 5.0 * u.deg,
        "viewcone_max": 5.0 * u.deg,
    }
    bins = histograms.view_cone_bins
    assert isinstance(bins, np.ndarray)
    assert len(bins) == 100
    assert bins[0] == pytest.approx(5.0)
    assert bins[-1] == pytest.approx(5.5)


def test_accumulate_tracks_angular_distance_range(mock_reader, hdf5_file_name, mocker):
    histograms = EventDataHistograms.create_accumulator()
    triggered_data = mocker.Mock(angular_distance=np.array([0.1, 0.1]))
    mocker.patch.object(histograms, "_update_file_info")
    mocker.patch.object(histograms, "_define_histograms", return_value={})

    histograms.accumulate(mocker.Mock(), mocker.Mock(), mocker.Mock(), triggered_data)
    histograms.accumulate(mocker.Mock(), mocker.Mock(), mocker.Mock(), triggered_data)

    assert histograms.data_ranges["angular_distance"] == pytest.approx((0.1, 0.1))


def test_histogram_definitions_are_filled_independently(mocker):
    histograms = EventDataHistograms.create_accumulator()
    definitions = histograms._define_histograms(None, None, None)
    assert all("event_data_column" in definition for definition in definitions.values())
    energy = definitions["energy"]
    energy["event_data"] = mocker.Mock(simulated_energy=np.array([1.0]))
    histograms._histogram_specs = definitions
    histograms._fill_histogram_and_bin_edges("energy", energy)
    assert np.sum(histograms.histograms["energy"].values) == 1
    assert "energy_mc" not in histograms.histograms
    np.testing.assert_array_equal(
        histograms.histograms["energy"].edges[0], definitions["energy_mc"]["edges"][0]
    )
