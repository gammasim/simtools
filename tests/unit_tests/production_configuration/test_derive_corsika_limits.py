import astropy.units as u
import numpy as np
import pytest

from simtools.production_configuration.derive_corsika_limits import LimitCalculator


@pytest.fixture
def mock_reader(mocker):
    mock = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits.SimtelIOEventDataReader"
    )
    mock.return_value.triggered_shower_data.simulated_energy = np.array([1, 10, 100])
    mock.return_value.triggered_data = mocker.Mock()
    mock.return_value.triggered_data.angular_distance = np.array([0.5, 1.0, 1.5])
    return mock


@pytest.fixture
def hdf5_file_name():
    return "test_file.h5"


def test_init(mock_reader, hdf5_file_name):
    """Test initialization with telescope list."""
    test_telescope_list = [1, 2]
    calculator = LimitCalculator(hdf5_file_name, "test_array", test_telescope_list)

    assert calculator.event_data_file == hdf5_file_name
    mock_reader.assert_called_once_with(hdf5_file_name, telescope_list=test_telescope_list)


def test_init_default_telescope_list(mock_reader, hdf5_file_name):
    """Test initialization without telescope list."""
    calculator = LimitCalculator(hdf5_file_name)

    assert calculator.event_data_file == hdf5_file_name
    mock_reader.assert_called_once_with(hdf5_file_name, telescope_list=None)


def test_compute_limits_lower(mock_reader, hdf5_file_name):
    calculator = LimitCalculator(hdf5_file_name)

    hist = np.array([1, 2, 3, 4, 5])
    bin_edges = np.array([0, 1, 2, 3, 4, 5])
    loss_fraction = 0.2

    with pytest.raises(ValueError, match="limit_type must be 'lower' or 'upper'"):
        calculator._compute_limits(hist, bin_edges, loss_fraction, limit_type="blabla")

    result = calculator._compute_limits(hist, bin_edges, loss_fraction, limit_type="lower")
    assert result == 2


def test_compute_limits_upper(mock_reader, hdf5_file_name):
    calculator = LimitCalculator(hdf5_file_name)

    hist = np.array([5, 4, 3, 2, 1])
    bin_edges = np.array([0, 1, 2, 3, 4, 5])
    loss_fraction = 0.2

    result = calculator._compute_limits(hist, bin_edges, loss_fraction, limit_type="upper")

    assert result == 3


def test_compute_limits_default_type(mock_reader, hdf5_file_name):
    calculator = LimitCalculator(hdf5_file_name)

    hist = np.array([1, 2, 3, 4, 5])
    bin_edges = np.array([0, 1, 2, 3, 4, 5])
    loss_fraction = 0.2

    result = calculator._compute_limits(hist, bin_edges, loss_fraction)  # Default is lower

    assert result == 2


def test_energy_bins(mock_reader, hdf5_file_name):
    calculator = LimitCalculator(hdf5_file_name)
    mock_reader.return_value.triggered_shower_data.simulated_energy = np.array([1, 10, 100])
    bins = calculator.energy_bins
    assert isinstance(bins, np.ndarray)
    assert len(bins) == 100


def test_core_distance_bins(mock_reader, hdf5_file_name):
    calculator = LimitCalculator(hdf5_file_name)
    mock_reader.return_value.triggered_shower_data.core_distance_shower = np.array([10, 20, 30])
    bins = calculator.core_distance_bins
    assert isinstance(bins, np.ndarray)
    assert len(bins) == 100


def test_view_cone_bins(mock_reader, hdf5_file_name):
    calculator = LimitCalculator(hdf5_file_name)
    mock_reader.return_value.triggered_data.angular_distance = np.array([0.5, 1.0, 1.5])
    bins = calculator.view_cone_bins
    assert isinstance(bins, np.ndarray)
    assert len(bins) == 100


def test_compute_viewcone(mock_reader, hdf5_file_name, mocker):
    calculator = LimitCalculator(hdf5_file_name)

    mock_hist = np.array([10, 8, 6, 4, 2])
    mock_bins = np.linspace(0, 20.0, 100)
    calculator.histograms = {"angular_distance": mock_hist, "angular_distance_bin_edges": mock_bins}
    result = calculator.compute_viewcone(0.2)

    assert isinstance(result, u.Quantity)
    assert result.unit == u.deg
    assert result.value > 0

    expected = calculator._compute_limits(mock_hist, mock_bins, 0.2, limit_type="upper") * u.deg
    assert result.value == pytest.approx(expected.value)


def test_compute_lower_energy_limit(mock_reader, hdf5_file_name, mocker):
    calculator = LimitCalculator(hdf5_file_name)

    mock_hist = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mock_bins = calculator.energy_bins
    calculator.histograms = {"energy": mock_hist, "energy_bin_edges": mock_bins}
    result = calculator.compute_lower_energy_limit(0.2)

    assert isinstance(result, u.Quantity)
    assert result.unit == u.TeV
    assert result.value > 0

    expected = calculator._compute_limits(mock_hist, mock_bins, 0.2, limit_type="lower") * u.TeV
    assert result == expected


def test_compute_upper_radius_limit(mock_reader, hdf5_file_name, mocker):
    calculator = LimitCalculator(hdf5_file_name)

    mock_hist = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    mock_bins = calculator.core_distance_bins
    calculator.histograms = {"core_distance": mock_hist, "core_distance_bin_edges": mock_bins}
    result = calculator.compute_upper_radius_limit(0.2)

    assert isinstance(result, u.Quantity)
    assert result.unit == u.m
    assert result.value > 0

    expected = calculator._compute_limits(mock_hist, mock_bins, 0.2, limit_type="upper") * u.m
    assert result == expected


def test_plot_data(mock_reader, hdf5_file_name, mocker, tmp_path):
    """Test plotting of data with limits."""
    calculator = LimitCalculator(hdf5_file_name)

    mock_reader.return_value.triggered_shower_data.core_distance_shower = np.array(
        [10.0, 20.0, 30.0]
    )
    mock_reader.return_value.triggered_shower_data.simulated_energy = np.array([1.0, 2.0, 3.0])
    mock_reader.return_value.triggered_shower_data.x_core_shower = np.array([1.0, 2.0, 3.0])
    mock_reader.return_value.triggered_shower_data.y_core_shower = np.array([1.0, 2.0, 3.0])
    mock_reader.return_value.triggered_data.angular_distance = np.array([0.5, 1.0, 1.5])

    mocker.patch("matplotlib.pyplot.figure")
    mocker.patch("matplotlib.pyplot.savefig")
    mocker.patch("matplotlib.pyplot.close")
    mock_create_plot = mocker.patch.object(calculator, "_create_plot")

    calculator.limits = {
        "lower_energy_limit": 1.0 * u.TeV,
        "upper_radius_limit": 100.0 * u.m,
        "viewcone_radius": 2.0 * u.deg,
    }

    # Mock the 2D histograms needed for plotting and cumulative energy distribution
    calculator.histograms["core_vs_energy"] = np.array([[1, 2, 3], [4, 5, 6]])
    calculator.histograms["core_vs_energy_bin_x_edges"] = np.array([0, 1, 2])
    calculator.histograms["core_vs_energy_bin_y_edges"] = np.array([0, 1, 2, 3])

    calculator.histograms["angular_distance_vs_energy"] = np.array([[1, 2, 3], [4, 5, 6]])
    calculator.histograms["angular_distance_vs_energy_bin_x_edges"] = np.array([0, 1, 2])
    calculator.histograms["angular_distance_vs_energy_bin_y_edges"] = np.array([0, 1, 2, 3])

    calculator.histograms["energy"] = np.array([1, 2, 3, 4])
    calculator.histograms["energy_bin_edges"] = np.array([0, 1, 2, 3, 4])

    calculator.histograms["core_distance"] = np.array([5, 6, 7, 8])
    calculator.histograms["core_distance_bin_edges"] = np.array([0, 10, 20, 30, 40])

    calculator.histograms["angular_distance"] = np.array([0.5, 0.7, 0.9, 1.1])
    calculator.histograms["angular_distance_bin_edges"] = np.array([0, 0.5, 1.0, 1.5, 2.0])

    calculator.plot_data(output_path=tmp_path)

    assert mock_create_plot.call_count == 11

    mock_create_plot.reset_mock()
    calculator.array_name = "test_array"
    calculator.plot_data(output_path=tmp_path)
    assert mock_create_plot.call_count == 11
    for call in mock_create_plot.call_args_list:
        _, kwargs = call
        assert "test_array" in str(kwargs.get("output_file"))


@pytest.fixture
def mock_figure(mocker):
    return mocker.patch("matplotlib.pyplot.figure")


@pytest.fixture
def mock_hist(mocker):
    return mocker.patch("matplotlib.pyplot.hist")


@pytest.fixture
def mock_hist2d(mocker):
    return mocker.patch("matplotlib.pyplot.pcolormesh")


@pytest.fixture
def mock_tight_layout(mocker):
    return mocker.patch("matplotlib.pyplot.tight_layout")


@pytest.fixture
def mock_show(mocker):
    return mocker.patch("matplotlib.pyplot.show")


@pytest.fixture
def mock_colorbar(mocker):
    return mocker.patch("matplotlib.pyplot.colorbar")


def test_create_plot_histogram(
    mocker,
    mock_reader,
    hdf5_file_name,
    mock_figure,
    mock_tight_layout,
    mock_show,
):
    calculator = LimitCalculator(hdf5_file_name)
    mock_axvline = mocker.patch("matplotlib.pyplot.axvline")
    mock_circle = mocker.patch("matplotlib.pyplot.Circle")

    x_data = np.array([1, 2, 3])
    bins = np.array([0, 1, 2, 3])
    plot_params = {"color": "blue"}

    fig = calculator._create_plot(
        data=x_data,
        bins=bins,
        plot_type="histogram",
        plot_params=plot_params,
        labels={"x": "X Label", "y": "Y Label", "title": "Test Plot"},
        lines={"x": 1.5, "y": 2.5, "r": 2.0},
    )

    mock_figure.assert_any_call(figsize=(8, 6))

    mock_tight_layout.assert_called_once()
    mock_show.assert_called_once()

    mock_axvline.assert_called_once()
    mock_circle.assert_called_once()
    assert fig == mock_figure.return_value


def test_create_plot_histogram2d(
    mock_reader, hdf5_file_name, mock_colorbar, mocker, mock_hist2d, tmp_test_directory
):
    calculator = LimitCalculator(hdf5_file_name)
    mock_savefig = mocker.patch("matplotlib.pyplot.savefig")

    x_data = np.array([[1, 2], [3, 4]])

    bins = [np.array([0, 1, 2]), np.array([0, 1, 2])]
    plot_params = {"cmap": "viridis"}

    calculator._create_plot(
        data=x_data,
        bins=bins,
        plot_type="histogram2d",
        plot_params=plot_params,
        colorbar_label="Counts",
        output_file=tmp_test_directory / "test_hist2d_plot.png",
    )

    mock_hist2d.assert_called_once()
    mock_savefig.assert_called_once()


def test_compute_limits_all_directions(mock_reader, mocker, hdf5_file_name):
    calculator = LimitCalculator(hdf5_file_name)

    mocker.patch.object(calculator, "_fill_histograms")
    calculator.limits = {}

    mock_energy_limit = 1.0 * u.TeV
    mock_radius_limit = 100.0 * u.m
    mock_viewcone_limit = 2.0 * u.deg

    mock_energy_limit_fn = mocker.patch.object(
        calculator, "compute_lower_energy_limit", return_value=mock_energy_limit
    )
    mock_radius_limit_fn = mocker.patch.object(
        calculator, "compute_upper_radius_limit", return_value=mock_radius_limit
    )
    mock_viewcone_fn = mocker.patch.object(
        calculator, "compute_viewcone", return_value=mock_viewcone_limit
    )

    loss_fraction = 0.2
    result = calculator.compute_limits(loss_fraction)

    mock_energy_limit_fn.assert_called_once_with(loss_fraction)
    mock_radius_limit_fn.assert_called_once_with(loss_fraction)
    mock_viewcone_fn.assert_called_once_with(loss_fraction)

    assert result["lower_energy_limit"] == mock_energy_limit
    assert result["upper_radius_limit"] == mock_radius_limit
    assert result["viewcone_radius"] == mock_viewcone_limit

    assert calculator.limits["lower_energy_limit"] == mock_energy_limit
    assert calculator.limits["upper_radius_limit"] == mock_radius_limit
    assert calculator.limits["viewcone_radius"] == mock_viewcone_limit


def test_fill_histogram_and_bin_edges_1d_new(mock_reader, hdf5_file_name):
    calculator = LimitCalculator(hdf5_file_name)
    name = "test_hist"
    data = np.array([1, 2, 3, 4, 5])
    bins = np.array([0, 2, 4, 6])

    calculator._fill_histogram_and_bin_edges(name, data, bins)

    assert name in calculator.histograms
    assert f"{name}_bin_edges" in calculator.histograms
    assert np.array_equal(calculator.histograms[f"{name}_bin_edges"], bins)

    expected_hist, _ = np.histogram(data, bins=bins)
    assert np.array_equal(calculator.histograms[name], expected_hist)


def test_fill_histogram_and_bin_edges_1d_existing(mock_reader, hdf5_file_name):
    calculator = LimitCalculator(hdf5_file_name)
    name = "test_hist"
    initial_data = np.array([1, 2, 3])
    additional_data = np.array([4, 5, 6])
    bins = np.array([0, 2, 4, 6])

    calculator._fill_histogram_and_bin_edges(name, initial_data, bins)
    calculator.histograms[name].copy()

    calculator._fill_histogram_and_bin_edges(name, additional_data, bins)

    expected_total_hist, _ = np.histogram(
        np.concatenate([initial_data, additional_data]), bins=bins
    )
    assert np.array_equal(calculator.histograms[name], expected_total_hist)


def test_fill_histogram_and_bin_edges_2d_new(mock_reader, hdf5_file_name):
    calculator = LimitCalculator(hdf5_file_name)
    name = "test_hist_2d"
    data = (np.array([1, 2, 3]), np.array([4, 5, 6]))
    bins = [np.array([0, 2, 4]), np.array([3, 5, 7])]

    calculator._fill_histogram_and_bin_edges(name, data, bins, hist1d=False)
    calculator.histograms[name].copy()

    assert name in calculator.histograms
    assert f"{name}_bin_x_edges" in calculator.histograms
    assert f"{name}_bin_y_edges" in calculator.histograms
    assert np.array_equal(calculator.histograms[f"{name}_bin_x_edges"], bins[0])
    assert np.array_equal(calculator.histograms[f"{name}_bin_y_edges"], bins[1])

    expected_hist, _, _ = np.histogram2d(data[0], data[1], bins=bins)
    assert np.array_equal(calculator.histograms[name], expected_hist)


def test_fill_histogram_and_bin_edges_2d_existing(mock_reader, hdf5_file_name):
    calculator = LimitCalculator(hdf5_file_name)
    name = "test_hist_2d"
    initial_data = (np.array([1, 2]), np.array([4, 5]))
    additional_data = (np.array([2, 3]), np.array([5, 6]))
    bins = [np.array([0, 2, 4]), np.array([3, 5, 7])]

    calculator._fill_histogram_and_bin_edges(name, initial_data, bins, hist1d=False)

    calculator._fill_histogram_and_bin_edges(name, additional_data, bins, hist1d=False)

    combined_data = (
        np.concatenate([initial_data[0], additional_data[0]]),
        np.concatenate([initial_data[1], additional_data[1]]),
    )
    expected_total_hist, _, _ = np.histogram2d(combined_data[0], combined_data[1], bins=bins)
    assert np.array_equal(calculator.histograms[name], expected_total_hist)


def test_fill_histograms(mock_reader, hdf5_file_name, mocker):
    calculator = LimitCalculator(hdf5_file_name)

    mock_file_info = mocker.Mock()
    mock_event_data = mocker.Mock()
    mock_triggered_data = mocker.Mock()

    mock_reader.return_value.read_event_data.return_value = (
        mock_file_info,
        None,
        mock_event_data,
        mock_triggered_data,
    )

    mock_event_data.simulated_energy = np.array([1, 2, 3])
    mock_event_data.core_distance_shower = np.array([10, 20, 30])
    mock_event_data.x_core_shower = np.array([-10, 0, 10])
    mock_event_data.y_core_shower = np.array([-5, 0, 5])
    mock_triggered_data.angular_distance = np.array([0.5, 1.0, 1.5])

    mock_reader.return_value.data_sets = ["test_dataset"]

    mock_fill_hist = mocker.patch.object(calculator, "_fill_histogram_and_bin_edges")

    mock_energy_bins = np.array([0, 2, 4])
    mock_core_distance_bins = np.array([0, 20, 40])
    mock_view_cone_bins = np.array([0, 1, 2])

    mocker.patch.object(LimitCalculator, "energy_bins", property(lambda self: mock_energy_bins))
    mocker.patch.object(
        LimitCalculator, "core_distance_bins", property(lambda self: mock_core_distance_bins)
    )
    mocker.patch.object(
        LimitCalculator, "view_cone_bins", property(lambda self: mock_view_cone_bins)
    )

    mock_prepare_limits = mocker.patch.object(
        calculator, "_prepare_limit_data", return_value={"test": "data"}
    )

    calculator._fill_histograms()

    mock_reader.return_value.read_event_data.assert_called_once_with(
        hdf5_file_name, table_name_map="test_dataset"
    )

    mock_prepare_limits.assert_called_once_with(mock_file_info)

    assert mock_fill_hist.call_count == 6

    expected_calls = [
        mocker.call("energy", mock_event_data.simulated_energy, mock_energy_bins),
        mocker.call("core_distance", mock_event_data.core_distance_shower, mock_core_distance_bins),
        mocker.call("angular_distance", mock_triggered_data.angular_distance, mock_view_cone_bins),
        mocker.call(
            "shower_cores",
            (mock_event_data.x_core_shower, mock_event_data.y_core_shower),
            [mocker.ANY, mocker.ANY],
            hist1d=False,
        ),
        mocker.call(
            "core_vs_energy",
            (mock_event_data.core_distance_shower, mock_event_data.simulated_energy),
            [mock_core_distance_bins, mock_energy_bins],
            hist1d=False,
        ),
    ]
    mock_fill_hist.assert_has_calls(expected_calls)


def test_prepare_limit_data(mock_reader, hdf5_file_name, mocker):
    calculator = LimitCalculator(hdf5_file_name)

    mock_file_info = {
        "primary_particle": "gamma",
        "zenith": 20.0,
        "azimuth": 0.0,
        "nsb_level": 1.0,
        "energy_min": 0.1,
        "energy_max": 100.0,
        "core_scatter_min": 0.0,
        "core_scatter_max": 1000.0,
    }

    mock_get_info = mocker.patch.object(
        calculator.reader, "get_reduced_simulation_file_info", return_value=mock_file_info
    )

    mock_file_info_table = mocker.Mock()

    result = calculator._prepare_limit_data(mock_file_info_table)

    mock_get_info.assert_called_once_with(mock_file_info_table)
    assert calculator.file_info == mock_file_info

    assert isinstance(result, dict)
    assert result["primary_particle"] == mock_file_info["primary_particle"]
    assert result["zenith"] == mock_file_info["zenith"]
    assert result["azimuth"] == mock_file_info["azimuth"]
    assert result["nsb_level"] == mock_file_info["nsb_level"]
    assert result["array_name"] == calculator.array_name
    assert result["telescope_ids"] == calculator.telescope_list
    assert result["lower_energy_limit"] is None
    assert result["upper_radius_limit"] is None
    assert result["viewcone_radius"] is None


def test_prepare_limit_data_with_array_and_telescopes(mock_reader, hdf5_file_name, mocker):
    array_name = "test_array"
    telescope_list = [1, 2, 3]
    calculator = LimitCalculator(
        hdf5_file_name, array_name=array_name, telescope_list=telescope_list
    )

    mock_file_info = {
        "primary_particle": "proton",
        "zenith": 30.0,
        "azimuth": 90.0,
        "nsb_level": 0.5,
    }

    mocker.patch.object(
        calculator.reader, "get_reduced_simulation_file_info", return_value=mock_file_info
    )

    result = calculator._prepare_limit_data(mocker.Mock())

    # Check array name and telescope list are correctly set
    assert result["array_name"] == array_name
    assert result["telescope_ids"] == telescope_list


def test_calculate_cumulative_histogram(mock_reader, hdf5_file_name):
    """Test calculation of cumulative histogram."""
    calculator = LimitCalculator(hdf5_file_name)

    # Test 1D histogram
    test_hist_1d = np.array([1, 2, 3, 4])

    # Normal cumulative (left to right)
    result_1d = calculator._calculate_cumulative_histogram(test_hist_1d)
    expected_1d = np.array([1, 3, 6, 10])
    np.testing.assert_array_equal(result_1d, expected_1d)

    # Reverse cumulative (right to left)
    result_1d_reverse = calculator._calculate_cumulative_histogram(test_hist_1d, reverse=True)
    expected_1d_reverse = np.array([10, 9, 7, 4])
    np.testing.assert_array_equal(result_1d_reverse, expected_1d_reverse)

    # Test 2D histogram
    test_hist_2d = np.array([[1, 2, 3], [4, 5, 6]])

    # Default axis (axis=1)
    result_2d = calculator._calculate_cumulative_histogram(test_hist_2d)
    expected_2d = np.array([[1, 3, 6], [4, 9, 15]])
    np.testing.assert_array_equal(result_2d, expected_2d)

    # Along axis 0
    result_2d_axis0 = calculator._calculate_cumulative_histogram(test_hist_2d, axis=0)
    expected_2d_axis0 = np.array([[1, 2, 3], [5, 7, 9]])
    np.testing.assert_array_equal(result_2d_axis0, expected_2d_axis0)

    # With reverse=True
    result_2d_reverse = calculator._calculate_cumulative_histogram(test_hist_2d, reverse=True)
    expected_2d_reverse = np.array([[6, 5, 3], [15, 11, 6]])
    np.testing.assert_array_equal(result_2d_reverse, expected_2d_reverse)
