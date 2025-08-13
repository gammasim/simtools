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
    name = "test_hist"
    data = np.array([1, 2, 3, 4, 5])
    bins = np.array([0, 2, 4, 6])

    histograms._fill_histogram_and_bin_edges(name, data, bins)

    assert name in histograms.histograms
    assert f"{name}_bin_edges" in histograms.histograms
    assert np.array_equal(histograms.histograms[f"{name}_bin_edges"], bins)

    expected_hist, _ = np.histogram(data, bins=bins)
    assert np.array_equal(histograms.histograms[name], expected_hist)


def test_fill_histogram_and_bin_edges_1d_existing(mock_reader, hdf5_file_name):
    histograms = SimtelIOEventHistograms(hdf5_file_name)
    name = "test_hist"
    initial_data = np.array([1, 2, 3])
    additional_data = np.array([4, 5, 6])
    bins = np.array([0, 2, 4, 6])

    histograms._fill_histogram_and_bin_edges(name, initial_data, bins)
    histograms.histograms[name].copy()

    histograms._fill_histogram_and_bin_edges(name, additional_data, bins)

    expected_total_hist, _ = np.histogram(
        np.concatenate([initial_data, additional_data]), bins=bins
    )
    assert np.array_equal(histograms.histograms[name], expected_total_hist)


def test_fill_histogram_and_bin_edges_2d_new(mock_reader, hdf5_file_name):
    histograms = SimtelIOEventHistograms(hdf5_file_name)
    name = "test_hist_2d"
    data = (np.array([1, 2, 3]), np.array([4, 5, 6]))
    bins = [np.array([0, 2, 4]), np.array([3, 5, 7])]

    histograms._fill_histogram_and_bin_edges(name, data, bins, hist1d=False)
    histograms.histograms[name].copy()

    assert name in histograms.histograms
    assert f"{name}_bin_x_edges" in histograms.histograms
    assert f"{name}_bin_y_edges" in histograms.histograms
    assert np.array_equal(histograms.histograms[f"{name}_bin_x_edges"], bins[0])
    assert np.array_equal(histograms.histograms[f"{name}_bin_y_edges"], bins[1])

    expected_hist, _, _ = np.histogram2d(data[0], data[1], bins=bins)
    assert np.array_equal(histograms.histograms[name], expected_hist)


def test_fill_histogram_and_bin_edges_2d_existing(mock_reader, hdf5_file_name):
    histograms = SimtelIOEventHistograms(hdf5_file_name)
    name = "test_hist_2d"
    initial_data = (np.array([1, 2]), np.array([4, 5]))
    additional_data = (np.array([2, 3]), np.array([5, 6]))
    bins = [np.array([0, 2, 4]), np.array([3, 5, 7])]

    histograms._fill_histogram_and_bin_edges(name, initial_data, bins, hist1d=False)

    histograms._fill_histogram_and_bin_edges(name, additional_data, bins, hist1d=False)

    combined_data = (
        np.concatenate([initial_data[0], additional_data[0]]),
        np.concatenate([initial_data[1], additional_data[1]]),
    )
    expected_total_hist, _, _ = np.histogram2d(combined_data[0], combined_data[1], bins=bins)
    assert np.array_equal(histograms.histograms[name], expected_total_hist)


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

    mock_fill_hist = mocker.patch.object(histograms, "_fill_histogram_and_bin_edges")

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

    assert mock_fill_hist.call_count == 12

    # xy bins are internally computed; no direct assertion on values needed here

    expected_call_names = [
        "energy",
        "energy_mc",
        "core_distance",
        "core_distance_mc",
        "angular_distance",
        "angular_distance_mc",
        "x_core_shower_vs_y_core_shower",
        "x_core_shower_vs_y_core_shower_mc",
        "core_vs_energy",
        "core_vs_energy_mc",
        "angular_distance_vs_energy",
        "angular_distance_vs_energy_mc",
    ]
    actual_call_names = [c.args[0] for c in mock_fill_hist.call_args_list]
    assert actual_call_names == expected_call_names

    # First six histograms are 1D, remaining six are 2D
    hist1d_flags = [c.kwargs.get("hist1d", True) for c in mock_fill_hist.call_args_list]
    assert hist1d_flags == [True] * 6 + [False] * 6


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


def test_energy_bins_with_file_info(mock_reader, hdf5_file_name):
    """Test energy_bins property with file info values."""
    histograms = SimtelIOEventHistograms(hdf5_file_name)
    histograms.file_info = {
        "energy_min": 1.0 * u.TeV,
        "energy_max": 10.0 * u.TeV,
    }

    bins = histograms.energy_bins

    assert isinstance(bins, np.ndarray)
    assert len(bins) == 100
    assert bins[0] == pytest.approx(1.0)
    assert bins[-1] == pytest.approx(10.0)


def test_energy_bins_with_existing_edges(mock_reader, hdf5_file_name):
    """Test energy_bins property when bin edges already exist."""
    histograms = SimtelIOEventHistograms(hdf5_file_name)
    histograms.histograms["energy_bin_edges"] = np.array([0.1, 1.0, 10.0])

    bins = histograms.energy_bins

    assert isinstance(bins, np.ndarray)
    assert np.array_equal(bins, np.array([0.1, 1.0, 10.0]))


def test_core_distance_bins_default(mock_reader, hdf5_file_name):
    """Test core_distance_bins with default file_info values."""
    histograms = SimtelIOEventHistograms(hdf5_file_name)
    mock_reader.return_value.get_reduced_simulation_file_info.return_value = {}

    bins = histograms.core_distance_bins

    assert isinstance(bins, np.ndarray)
    assert len(bins) == 100
    assert bins[0] == 0.0
    assert bins[-1] == 1.0e5


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

    # Seed histogram dictionary with minimal required histograms
    energy = np.array([10, 20, 30, 40])
    core_distance = np.array([5, 15, 25, 35])
    angular_distance = np.array([2, 4, 6, 8])
    core_vs_energy = np.array([[1, 2], [3, 4]])  # shape (core_distance, energy)
    angular_distance_vs_energy = np.array([[2, 3], [4, 5]])

    histograms.histograms = {
        "energy": energy,
        "core_distance": core_distance,
        "angular_distance": angular_distance,
        "core_vs_energy": core_vs_energy,
        "angular_distance_vs_energy": angular_distance_vs_energy,
    }

    cumulative_data = histograms.calculate_cumulative_data()

    # Expected 1D cumulative distributions
    expected_cumulative_energy = np.array([100, 90, 70, 40])  # reverse cumulative
    expected_cumulative_core_distance = np.array([5, 20, 45, 80])
    expected_cumulative_angular_distance = np.array([2, 6, 12, 20])

    # Expected normalized cumulative for 2D histograms along axis=0 (column-wise)
    # For core_vs_energy:
    # cumulative along axis 0 -> [[1,2],[4,6]], totals per column [4,6]
    expected_norm_core_vs_energy = np.array([[1 / 4, 2 / 6], [1.0, 1.0]])
    # For angular_distance_vs_energy:
    # cumulative along axis 0 -> [[2,3],[6,8]], totals per column [6,8]
    expected_norm_ang_vs_energy = np.array([[2 / 6, 3 / 8], [1.0, 1.0]])

    assert set(cumulative_data.keys()) == {
        "cumulative_energy",
        "cumulative_core_distance",
        "cumulative_angular_distance",
        "normalized_cumulative_core_vs_energy",
        "normalized_cumulative_angular_distance_vs_energy",
    }

    np.testing.assert_array_equal(cumulative_data["cumulative_energy"], expected_cumulative_energy)
    np.testing.assert_array_equal(
        cumulative_data["cumulative_core_distance"], expected_cumulative_core_distance
    )
    np.testing.assert_array_equal(
        cumulative_data["cumulative_angular_distance"], expected_cumulative_angular_distance
    )
    np.testing.assert_allclose(
        cumulative_data["normalized_cumulative_core_vs_energy"], expected_norm_core_vs_energy
    )
    np.testing.assert_allclose(
        cumulative_data["normalized_cumulative_angular_distance_vs_energy"],
        expected_norm_ang_vs_energy,
    )


def test_get_existing_key(mock_histograms):
    """Test retrieving an existing key from the histogram dictionary."""
    mock_histograms.histograms["test_key"] = "test_value"
    result = mock_histograms.get("test_key")
    assert result == "test_value"


def test_get_non_existing_key_with_default(mock_histograms):
    """Test retrieving a non-existing key with a default value."""
    result = mock_histograms.get("non_existing_key", default="default_value")
    assert result == "default_value"


def test_get_non_existing_key_without_default(mock_histograms):
    """Test retrieving a non-existing key without a default value."""
    result = mock_histograms.get("non_existing_key")
    assert result is None
