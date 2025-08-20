import logging

import astropy.units as u
import numpy as np
import pytest

from simtools.simtel.simtel_io_event_histograms import SimtelIOEventHistograms


@pytest.fixture
def mock_reader(mocker):
    mock = mocker.patch("simtools.simtel.simtel_io_event_histograms.SimtelIOEventDataReader")
    mock.return_value.triggered_shower_data.simulated_energy = np.array([1, 10, 100])
    mock.return_value.shower_data.simulated_energy = np.array([1, 10, 100])
    mock.return_value.triggered_data = mocker.Mock()
    mock.return_value.triggered_data.angular_distance = np.array([0.5, 1.0, 1.5])
    return mock


@pytest.fixture
def hdf5_file_name():
    return "test_file.h5"


def test_init(mock_reader, hdf5_file_name):
    """Test initialization with telescope list."""
    test_telescope_list = [1, 2]
    histograms = SimtelIOEventHistograms(hdf5_file_name, "test_array", test_telescope_list)

    assert histograms.event_data_file == hdf5_file_name
    mock_reader.assert_called_once_with(hdf5_file_name, telescope_list=test_telescope_list)


def test_init_default_telescope_list(mock_reader, hdf5_file_name):
    """Test initialization without telescope list."""
    histograms = SimtelIOEventHistograms(hdf5_file_name)

    assert histograms.event_data_file == hdf5_file_name
    mock_reader.assert_called_once_with(hdf5_file_name, telescope_list=None)


def test_energy_bins(mock_reader, hdf5_file_name):
    histograms = SimtelIOEventHistograms(hdf5_file_name)
    mock_reader.return_value.triggered_shower_data.simulated_energy = np.array([1, 10, 100])
    bins = histograms.energy_bins
    assert isinstance(bins, np.ndarray)
    assert len(bins) == 100


def test_core_distance_bins(mock_reader, hdf5_file_name):
    histograms = SimtelIOEventHistograms(hdf5_file_name)
    mock_reader.return_value.triggered_shower_data.core_distance_shower = np.array([10, 20, 30])
    bins = histograms.core_distance_bins
    assert isinstance(bins, np.ndarray)
    assert len(bins) == 100


def test_view_cone_bins(mock_reader, hdf5_file_name):
    histograms = SimtelIOEventHistograms(hdf5_file_name)
    mock_reader.return_value.triggered_data.angular_distance = np.array([0.5, 1.0, 1.5])
    bins = histograms.view_cone_bins
    assert isinstance(bins, np.ndarray)
    assert len(bins) == 100


def test_fill_histogram_and_bin_edges_1d_new(mock_reader, hdf5_file_name):
    histograms = SimtelIOEventHistograms(hdf5_file_name)
    data = {
        "1d": True,
        "event_data": mock_reader.return_value,
        "event_data_column": "simulated_energy",
        "bin_edges": np.array([0, 2, 4, 6]),
        "histogram": None,
    }
    # Patch the attribute to return the test data
    setattr(mock_reader.return_value, "simulated_energy", np.array([1, 2, 3, 4, 5]))
    histograms._fill_histogram_and_bin_edges(data)
    assert data["histogram"] is not None
    expected_hist, _ = np.histogram(np.array([1, 2, 3, 4, 5]), bins=np.array([0, 2, 4, 6]))
    assert np.array_equal(data["histogram"], expected_hist)


def test_fill_histogram_and_bin_edges_1d_existing(mock_reader, hdf5_file_name):
    histograms = SimtelIOEventHistograms(hdf5_file_name)
    data = {
        "1d": True,
        "event_data": mock_reader.return_value,
        "event_data_column": "simulated_energy",
        "bin_edges": np.array([0, 2, 4, 6]),
        "histogram": None,
    }
    # Patch the attribute to return the test data
    setattr(mock_reader.return_value, "simulated_energy", np.array([1, 2, 3]))
    histograms._fill_histogram_and_bin_edges(data)
    # Add more data
    setattr(mock_reader.return_value, "simulated_energy", np.array([4, 5, 6]))
    histograms._fill_histogram_and_bin_edges(data)
    expected_total_hist, _ = np.histogram(
        np.concatenate([np.array([1, 2, 3]), np.array([4, 5, 6])]), bins=np.array([0, 2, 4, 6])
    )
    assert np.array_equal(data["histogram"], expected_total_hist)


def test_fill_histogram_and_bin_edges_2d_new(mock_reader, hdf5_file_name):
    histograms = SimtelIOEventHistograms(hdf5_file_name)
    data = {
        "1d": False,
        "event_data": (mock_reader.return_value, mock_reader.return_value),
        "event_data_column": ("simulated_energy", "core_distance_shower"),
        "bin_edges": [np.array([0, 2, 4]), np.array([3, 5, 7])],
        "histogram": None,
    }
    setattr(mock_reader.return_value, "simulated_energy", np.array([1, 2, 3]))
    setattr(mock_reader.return_value, "core_distance_shower", np.array([4, 5, 6]))
    histograms._fill_histogram_and_bin_edges(data)
    assert data["histogram"] is not None
    expected_hist, _, _ = np.histogram2d(
        np.array([1, 2, 3]), np.array([4, 5, 6]), bins=[np.array([0, 2, 4]), np.array([3, 5, 7])]
    )
    assert np.array_equal(data["histogram"], expected_hist)


def test_fill_histogram_and_bin_edges_2d_existing(mock_reader, hdf5_file_name):
    histograms = SimtelIOEventHistograms(hdf5_file_name)
    data = {
        "1d": False,
        "event_data": (mock_reader.return_value, mock_reader.return_value),
        "event_data_column": ("simulated_energy", "core_distance_shower"),
        "bin_edges": [np.array([0, 2, 4]), np.array([3, 5, 7])],
        "histogram": None,
    }
    setattr(mock_reader.return_value, "simulated_energy", np.array([1, 2]))
    setattr(mock_reader.return_value, "core_distance_shower", np.array([4, 5]))
    histograms._fill_histogram_and_bin_edges(data)
    setattr(mock_reader.return_value, "simulated_energy", np.array([2, 3]))
    setattr(mock_reader.return_value, "core_distance_shower", np.array([5, 6]))
    histograms._fill_histogram_and_bin_edges(data)
    combined_data = (
        np.concatenate([np.array([1, 2]), np.array([2, 3])]),
        np.concatenate([np.array([4, 5]), np.array([5, 6])]),
    )
    expected_total_hist, _, _ = np.histogram2d(
        combined_data[0], combined_data[1], bins=[np.array([0, 2, 4]), np.array([3, 5, 7])]
    )
    assert np.array_equal(data["histogram"], expected_total_hist)


def test_fill(mock_reader, hdf5_file_name, mocker):
    histograms = SimtelIOEventHistograms(hdf5_file_name)

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
    def fake_fill_histogram_and_bin_edges(data):
        data["histogram"] = np.array([1, 2, 3])
        data["axis_titles"] = ["x", "y"]

    mocker.patch.object(
        histograms, "_fill_histogram_and_bin_edges", side_effect=fake_fill_histogram_and_bin_edges
    )

    mock_energy_bins = np.array([0, 2, 4])
    mock_core_distance_bins = np.array([0, 20, 40])
    mock_view_cone_bins = np.array([0, 1, 2])

    mocker.patch.object(
        SimtelIOEventHistograms, "energy_bins", property(lambda self: mock_energy_bins)
    )
    mocker.patch.object(
        SimtelIOEventHistograms,
        "core_distance_bins",
        property(lambda self: mock_core_distance_bins),
    )
    mocker.patch.object(
        SimtelIOEventHistograms, "view_cone_bins", property(lambda self: mock_view_cone_bins)
    )

    histograms.fill()

    mock_reader.return_value.read_event_data.assert_called_once_with(
        hdf5_file_name, table_name_map="test_dataset"
    )


def test_calculate_cumulative_histogram(mock_reader, hdf5_file_name):
    """Test calculation of cumulative histogram."""
    histograms = SimtelIOEventHistograms(hdf5_file_name)

    # Test None case
    result_none = histograms._calculate_cumulative_histogram(None)
    assert result_none is None

    # Test 1D histogram
    test_hist_1d = np.array([1, 2, 3, 4])

    # Test direct call to _calculate_cumulative_1d
    result_1d_direct = histograms._calculate_cumulative_1d(test_hist_1d, False)
    expected_1d = np.array([1, 3, 6, 10])
    np.testing.assert_array_equal(result_1d_direct, expected_1d)

    # Normal cumulative (left to right)
    result_1d = histograms._calculate_cumulative_histogram(test_hist_1d)
    expected_1d = np.array([1, 3, 6, 10])
    np.testing.assert_array_equal(result_1d, expected_1d)

    # Reverse cumulative (right to left)
    result_1d_reverse = histograms._calculate_cumulative_histogram(test_hist_1d, reverse=True)
    expected_1d_reverse = np.array([10, 9, 7, 4])
    np.testing.assert_array_equal(result_1d_reverse, expected_1d_reverse)

    # Test 1D histogram with normalization
    result_1d_normalized = histograms._calculate_cumulative_histogram(test_hist_1d, normalize=True)
    expected_1d_normalized = np.array([0.1, 0.3, 0.6, 1.0])
    np.testing.assert_allclose(result_1d_normalized, expected_1d_normalized)

    # Test 2D histogram - use float dtype to avoid casting issues
    test_hist_2d = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)

    # Test direct call to _calculate_cumulative_2d
    result_2d_direct = histograms._calculate_cumulative_2d(test_hist_2d, False)
    expected_2d_direct = np.array([[1, 3, 6], [4, 9, 15]])
    np.testing.assert_array_equal(result_2d_direct, expected_2d_direct)

    # Test _apply_cumsum_along_axis with different parameters
    # Test axis=1, reverse=False
    result_axis1_no_reverse = histograms._apply_cumsum_along_axis(
        test_hist_2d.copy(), axis=1, reverse=False
    )
    expected_axis1_no_reverse = np.array([[1, 3, 6], [4, 9, 15]])
    np.testing.assert_array_equal(result_axis1_no_reverse, expected_axis1_no_reverse)

    # Test axis=1, reverse=True
    result_axis1_reverse = histograms._apply_cumsum_along_axis(
        test_hist_2d.copy(), axis=1, reverse=True
    )
    expected_axis1_reverse = np.array([[6, 5, 3], [15, 11, 6]])
    np.testing.assert_array_equal(result_axis1_reverse, expected_axis1_reverse)

    # Test axis=0, reverse=False
    result_axis0_no_reverse = histograms._apply_cumsum_along_axis(
        test_hist_2d.copy(), axis=0, reverse=False
    )
    expected_axis0_no_reverse = np.array([[1, 2, 3], [5, 7, 9]])
    np.testing.assert_array_equal(result_axis0_no_reverse, expected_axis0_no_reverse)

    # Test axis=0, reverse=True
    result_axis0_reverse = histograms._apply_cumsum_along_axis(
        test_hist_2d.copy(), axis=0, reverse=True
    )
    expected_axis0_reverse = np.array([[5, 7, 9], [4, 5, 6]])
    np.testing.assert_array_equal(result_axis0_reverse, expected_axis0_reverse)

    # Default axis (axis=1)
    result_2d = histograms._calculate_cumulative_histogram(test_hist_2d)
    expected_2d = np.array([[1, 3, 6], [4, 9, 15]])
    np.testing.assert_array_equal(result_2d, expected_2d)

    # Test 2D histogram with normalization - row normalization
    result_2d_normalized = histograms._calculate_cumulative_histogram(
        test_hist_2d, normalize=True, axis=1
    )
    expected_2d_normalized = np.array([[1 / 6, 3 / 6, 1.0], [4 / 15, 9 / 15, 1.0]])
    np.testing.assert_allclose(result_2d_normalized, expected_2d_normalized)

    # Along axis 0
    result_2d_axis0 = histograms._calculate_cumulative_histogram(test_hist_2d, axis=0)
    expected_2d_axis0 = np.array([[1, 2, 3], [5, 7, 9]])
    np.testing.assert_array_equal(result_2d_axis0, expected_2d_axis0)

    # With reverse=True
    result_2d_reverse = histograms._calculate_cumulative_histogram(test_hist_2d, reverse=True)
    np.testing.assert_array_equal(result_2d_reverse, np.array([[6, 5, 3], [15, 11, 6]]))


def test_normalized_cumulative_histogram(mock_reader, hdf5_file_name):
    """Test normalized cumulative histogram calculation for alpha plots."""
    histograms = SimtelIOEventHistograms(hdf5_file_name)

    # Test None case
    result_none = histograms._calculate_cumulative_histogram(None, normalize=True)
    assert result_none is None

    test_hist_2d = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [0, 0, 0],
        ],
        dtype=float,
    )

    # Test with explicit axis=1
    result_axis1 = histograms._calculate_cumulative_histogram(test_hist_2d, normalize=True, axis=1)

    expected_axis1 = np.array([[1 / 6, 3 / 6, 1.0], [4 / 15, 9 / 15, 1.0], [0, 0, 0]])

    np.testing.assert_allclose(result_axis1, expected_axis1, rtol=1e-4)

    # Test with axis=0
    result_axis0 = histograms._calculate_cumulative_histogram(test_hist_2d, axis=0, normalize=True)

    expected_axis0 = np.array([[1 / 5, 2 / 7, 3 / 9], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

    np.testing.assert_allclose(result_axis0, expected_axis0, rtol=1e-4)

    # Test 1D histogram normalization
    test_hist_1d = np.array([10, 20, 30, 40], dtype=float)
    result_1d = histograms._calculate_cumulative_histogram(test_hist_1d, normalize=True)
    expected_1d = np.array([0.1, 0.3, 0.6, 1.0])
    np.testing.assert_allclose(result_1d, expected_1d)

    # Test normalization with reverse=True
    result_reverse = histograms._calculate_cumulative_histogram(
        test_hist_1d, reverse=True, normalize=True
    )
    expected_reverse = np.array([1.0, 0.9, 0.7, 0.4])
    np.testing.assert_allclose(result_reverse, expected_reverse)


@pytest.fixture
def mock_histograms(mocker):
    """Create a mocked SimtelIOEventHistograms that doesn't require a file."""
    mocker.patch("simtools.simtel.simtel_io_event_histograms.SimtelIOEventDataReader")
    return SimtelIOEventHistograms("dummy_file.h5", "test_array")


def test_rebin_2d_histogram(mock_histograms):
    """Test rebinning a 2D histogram along the energy dimension (y-axis) only."""
    hist = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    x_bins = np.array([0, 1, 2, 3, 4])
    y_bins = np.array([0, 10, 20, 30, 40])

    histograms = mock_histograms

    rebinned_hist, rebinned_x_bins, rebinned_y_bins = histograms.rebin_2d_histogram(
        hist, x_bins, y_bins, rebin_factor=2
    )

    expected_hist = np.array([[3, 7], [11, 15], [19, 23], [27, 31]])
    expected_x_bins = x_bins
    expected_y_bins = np.array([0, 20, 40])

    assert np.array_equal(rebinned_hist, expected_hist)
    assert np.array_equal(rebinned_x_bins, expected_x_bins)
    assert np.array_equal(rebinned_y_bins, expected_y_bins)

    rebinned_hist, rebinned_x_bins, rebinned_y_bins = histograms.rebin_2d_histogram(
        hist, x_bins, y_bins, rebin_factor=1
    )

    assert np.array_equal(rebinned_hist, hist)
    assert np.array_equal(rebinned_x_bins, x_bins)
    assert np.array_equal(rebinned_y_bins, y_bins)

    rebinned_hist, rebinned_x_bins, rebinned_y_bins = histograms.rebin_2d_histogram(
        hist, x_bins, y_bins, rebin_factor=4
    )

    expected_hist = np.array([[10], [26], [42], [58]])

    expected_y_bins = np.array([0, 40])

    assert np.array_equal(rebinned_hist, expected_hist)
    assert np.array_equal(rebinned_x_bins, expected_x_bins)
    assert np.array_equal(rebinned_y_bins, expected_y_bins)


def test_energy_bins_default(mock_reader, hdf5_file_name):
    """Test energy_bins property with default values."""
    histograms = SimtelIOEventHistograms(hdf5_file_name)
    mock_reader.return_value.get_reduced_simulation_file_info.return_value = {}

    bins = histograms.energy_bins

    assert isinstance(bins, np.ndarray)
    assert len(bins) == 100
    assert bins[0] == pytest.approx(1.0e-3)
    assert bins[-1] == pytest.approx(1.0e3)


def test_core_distance_bins_with_file_info(mock_reader, hdf5_file_name):
    """Test core_distance_bins with file_info values."""
    histograms = SimtelIOEventHistograms(hdf5_file_name)
    histograms.file_info = {
        "core_scatter_min": 10.0 * u.m,
        "core_scatter_max": 500.0 * u.m,
    }

    bins = histograms.core_distance_bins

    assert isinstance(bins, np.ndarray)
    assert len(bins) == 100
    assert bins[0] == 10.0
    assert bins[-1] == 500.0


def test_core_distance_bins_with_existing_edges(mock_reader, hdf5_file_name):
    """Test core_distance_bins when bin edges already exist in histograms."""
    histograms = SimtelIOEventHistograms(hdf5_file_name)
    mock_edges = np.linspace(0, 1000, 50)
    histograms.histograms["core_distance_bin_edges"] = mock_edges

    bins = histograms.core_distance_bins

    assert isinstance(bins, np.ndarray)
    assert len(bins) == 50
    assert np.array_equal(bins, mock_edges)


def test_view_cone_bins_default(mock_reader, hdf5_file_name):
    """Test default view_cone_bins when no histogram data is present."""
    histograms = SimtelIOEventHistograms(hdf5_file_name)
    histograms.file_info = {
        "viewcone_min": 0.0 * u.deg,
        "viewcone_max": 10.0 * u.deg,
    }
    bins = histograms.view_cone_bins
    assert isinstance(bins, np.ndarray)
    assert len(bins) == 100
    assert bins[0] == 0.0
    assert bins[-1] == 10.0


def test_view_cone_bins_with_histogram_data(mock_reader, hdf5_file_name):
    """Test view_cone_bins when histogram data is already present."""
    histograms = SimtelIOEventHistograms(hdf5_file_name)
    mock_bins = np.linspace(0.0, 5.0, 50)
    histograms.histograms["viewcone_bin_edges"] = mock_bins
    bins = histograms.view_cone_bins
    assert isinstance(bins, np.ndarray)
    assert len(bins) == 50
    assert np.array_equal(bins, mock_bins)


def test_view_cone_bins_no_file_info(mock_reader, hdf5_file_name):
    """Test view_cone_bins when file_info is empty."""
    histograms = SimtelIOEventHistograms(hdf5_file_name)
    bins = histograms.view_cone_bins
    assert isinstance(bins, np.ndarray)
    assert len(bins) == 100
    assert bins[0] == 0.0
    assert bins[-1] == 20.0


def test_calculate_cumulative_data(mock_reader, hdf5_file_name):
    """Test calculate_cumulative_data end-to-end without patching internals."""
    histograms = SimtelIOEventHistograms(hdf5_file_name)

    # Seed histogram dictionary with minimal required histograms (as dicts)
    histograms.histograms = {
        "energy": {"histogram": np.array([10, 20, 30, 40]), "axis_titles": ["E", ""]},
        "core_distance": {"histogram": np.array([5, 15, 25, 35]), "axis_titles": ["r", ""]},
        "angular_distance": {"histogram": np.array([2, 4, 6, 8]), "axis_titles": ["theta", ""]},
        "core_vs_energy": {"histogram": np.array([[1, 2], [3, 4]]), "axis_titles": ["r", "E"]},
        "angular_distance_vs_energy": {
            "histogram": np.array([[2, 3], [4, 5]]),
            "axis_titles": ["theta", "E"],
        },
    }
    cumulative_data = histograms.calculate_cumulative_data()
    # Expected 1D cumulative distributions
    expected_cumulative_energy = np.array([100, 90, 70, 40])  # reverse cumulative
    expected_cumulative_core_distance = np.array([5, 20, 45, 80])
    expected_cumulative_angular_distance = np.array([2, 6, 12, 20])
    # Expected normalized cumulative for 2D histograms along axis=0 (column-wise)
    expected_norm_core_vs_energy = np.array([[1 / 4, 2 / 6], [1.0, 1.0]])
    expected_norm_ang_vs_energy = np.array([[2 / 6, 3 / 8], [1.0, 1.0]])
    assert set(cumulative_data.keys()) == {
        "core_distance_cumulative",
        "angular_distance_cumulative",
        "angular_distance_vs_energy_cumulative",
        "core_vs_energy_cumulative",
        "energy_cumulative",
    }
    np.testing.assert_array_equal(
        cumulative_data["energy_cumulative"]["histogram"], expected_cumulative_energy
    )
    np.testing.assert_array_equal(
        cumulative_data["core_distance_cumulative"]["histogram"], expected_cumulative_core_distance
    )
    np.testing.assert_array_equal(
        cumulative_data["angular_distance_cumulative"]["histogram"],
        expected_cumulative_angular_distance,
    )
    np.testing.assert_allclose(
        cumulative_data["core_vs_energy_cumulative"]["histogram"], expected_norm_core_vs_energy
    )
    np.testing.assert_allclose(
        cumulative_data["angular_distance_vs_energy_cumulative"]["histogram"],
        expected_norm_ang_vs_energy,
    )


def test_calculate_efficiency_data(mock_reader, hdf5_file_name):
    """Test calculate_efficiency_data method."""
    histograms = SimtelIOEventHistograms(hdf5_file_name)

    # Mock histograms for triggered and simulated events (as dicts)
    histograms.histograms = {
        "energy": {"histogram": np.array([10, 20, 30]), "axis_titles": ["E", ""]},
        "energy_mc": {"histogram": np.array([20, 40, 60]), "axis_titles": ["E", ""]},
        "core_distance": {"histogram": np.array([5, 10, 15]), "axis_titles": ["r", ""]},
        "core_distance_mc": {"histogram": np.array([10, 20, 30]), "axis_titles": ["r", ""]},
        "angular_distance": {"histogram": np.array([0, 5, 10]), "axis_titles": ["theta", ""]},
        "angular_distance_mc": {"histogram": np.array([0, 10, 20]), "axis_titles": ["theta", ""]},
    }
    # Call the method
    efficiency_data = histograms.calculate_efficiency_data()
    # Check efficiency histograms
    assert "energy_eff" in efficiency_data
    assert "core_distance_eff" in efficiency_data
    assert "angular_distance_eff" in efficiency_data
    np.testing.assert_array_almost_equal(
        efficiency_data["energy_eff"]["histogram"], np.array([0.5, 0.5, 0.5])
    )
    np.testing.assert_array_almost_equal(
        efficiency_data["core_distance_eff"]["histogram"], np.array([0.5, 0.5, 0.5])
    )
    np.testing.assert_array_almost_equal(
        efficiency_data["angular_distance_eff"]["histogram"], np.array([0.0, 0.5, 0.5])
    )


def test_calculate_efficiency_data_shape_mismatch(mock_reader, hdf5_file_name, caplog):
    """Test calculate_efficiency_data with shape mismatch."""
    histograms = SimtelIOEventHistograms(hdf5_file_name)

    # Mock histograms with mismatched shapes (as dicts)
    histograms.histograms = {
        "energy": {"histogram": np.array([10, 20]), "axis_titles": ["E", ""]},
        "energy_mc": {"histogram": np.array([20, 40, 60]), "axis_titles": ["E", ""]},
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
    """Test calculate_efficiency_data with missing histograms."""
    histograms = SimtelIOEventHistograms(hdf5_file_name)

    # Mock histograms with missing triggered histogram (as dict)
    histograms.histograms = {
        "energy_mc": {"histogram": np.array([20, 40, 60]), "axis_titles": ["E", ""]},
    }
    efficiency_data = histograms.calculate_efficiency_data()
    # Ensure no efficiency histogram is created
    assert "energy_eff" not in efficiency_data


def test_energy_bins_with_histogram_edges(mock_reader, hdf5_file_name):
    histograms = SimtelIOEventHistograms(hdf5_file_name)
    mock_edges = np.linspace(0, 10, 5)
    histograms.histograms["energy_bin_edges"] = mock_edges
    bins = histograms.energy_bins
    assert np.array_equal(bins, mock_edges)


def test_print_summary(mock_histograms, mocker, caplog):
    """Test the print_summary method."""
    histograms = mock_histograms

    # Mock histogram data
    mock_histograms.histograms = {
        "energy_mc": {"histogram": np.array([10, 20, 30])},
        "energy": {"histogram": np.array([5, 15, 25])},
    }

    # Capture log output
    with caplog.at_level(logging.INFO):
        histograms.print_summary()

    # Verify log messages
    assert "Total simulated events: 60" in caplog.text
    assert "Total triggered events: 45" in caplog.text
