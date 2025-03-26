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
    test_telescope_list = [1, 2]

    calculator = LimitCalculator(hdf5_file_name, test_telescope_list)

    assert calculator.event_data_file == hdf5_file_name
    assert calculator.telescope_list == test_telescope_list

    mock_reader.assert_called_once_with(hdf5_file_name, telescope_list=[1, 2])


def test_init_default_telescope_list(mock_reader, hdf5_file_name):
    calculator = LimitCalculator(hdf5_file_name)

    assert calculator.event_data_file == hdf5_file_name
    assert calculator.telescope_list is None

    mock_reader.assert_called_once_with(hdf5_file_name, telescope_list=None)


def test_compute_limits_lower(mock_reader, hdf5_file_name):
    calculator = LimitCalculator(hdf5_file_name)

    hist = np.array([1, 2, 3, 4, 5])
    bin_edges = np.array([0, 1, 2, 3, 4, 5])
    loss_fraction = 0.2

    result = calculator._compute_limits(hist, bin_edges, loss_fraction, limit_type="lower")
    assert result == 4  # With 20% loss from lower edge


def test_compute_limits_upper(mock_reader, hdf5_file_name):
    calculator = LimitCalculator(hdf5_file_name)

    hist = np.array([5, 4, 3, 2, 1])
    bin_edges = np.array([0, 1, 2, 3, 4, 5])
    loss_fraction = 0.2

    result = calculator._compute_limits(hist, bin_edges, loss_fraction, limit_type="upper")

    assert result == 2  # With 20% loss from lower edge


def test_compute_limits_default_type(mock_reader, hdf5_file_name):
    calculator = LimitCalculator(hdf5_file_name)

    hist = np.array([1, 2, 3, 4, 5])
    bin_edges = np.array([0, 1, 2, 3, 4, 5])
    loss_fraction = 0.2

    result = calculator._compute_limits(hist, bin_edges, loss_fraction)  # Default is lower

    assert result == 4  # With 20% loss from lower edge


def test_energy_bins(mock_reader, hdf5_file_name):
    calculator = LimitCalculator(hdf5_file_name)
    mock_reader.return_value.triggered_shower_data.simulated_energy = np.array([1, 10, 100])
    bins = calculator.energy_bins
    assert isinstance(bins, np.ndarray)
    assert len(bins) == 1000


def test_core_distance_bins(mock_reader, hdf5_file_name):
    calculator = LimitCalculator(hdf5_file_name)
    mock_reader.return_value.triggered_shower_data.core_distance_shower = np.array([10, 20, 30])
    bins = calculator.core_distance_bins
    assert isinstance(bins, np.ndarray)
    assert len(bins) == 1000


def test_view_cone_bins(mock_reader, hdf5_file_name):
    calculator = LimitCalculator(hdf5_file_name)
    mock_reader.return_value.triggered_data.angular_distance = np.array([0.5, 1.0, 1.5])
    bins = calculator.view_cone_bins
    assert isinstance(bins, np.ndarray)
    assert len(bins) == 1000


def test_compute_viewcone(mock_reader, hdf5_file_name, mocker):
    calculator = LimitCalculator(hdf5_file_name)

    # Mock the angular distance data
    mock_angular_distance = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    mock_reader.return_value.triggered_data.angular_distance = mock_angular_distance

    # Mock view_cone_bins property
    mock_bins = np.linspace(0, 3, 1000)
    mocker.patch.object(LimitCalculator, "view_cone_bins", property(lambda self: mock_bins))

    # Test with 20% loss fraction
    result = calculator.compute_viewcone(0.2)

    # Verify the result
    assert isinstance(result, u.Quantity)
    assert result.unit == u.deg
    assert result.value > 0

    # Verify the histogram and compute_limits were used correctly
    hist, _ = np.histogram(mock_angular_distance, bins=mock_bins)
    expected = calculator._compute_limits(hist, mock_bins, 0.2, limit_type="upper") * u.deg
    assert result == expected


def test_compute_lower_energy_limit(mock_reader, hdf5_file_name, mocker):
    calculator = LimitCalculator(hdf5_file_name)

    # Mock the simulated energy data
    mock_energy = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mock_reader.return_value.triggered_shower_data.simulated_energy = mock_energy

    # Mock energy_bins property
    mock_bins = np.logspace(0, 1, 1000)
    mocker.patch.object(LimitCalculator, "energy_bins", property(lambda self: mock_bins))

    # Test with 20% loss fraction
    result = calculator.compute_lower_energy_limit(0.2)

    # Verify the result
    assert isinstance(result, u.Quantity)
    assert result.unit == u.TeV
    assert result.value > 0

    # Verify the histogram and compute_limits were used correctly
    hist, _ = np.histogram(mock_energy, bins=mock_bins)
    expected = calculator._compute_limits(hist, mock_bins, 0.2, limit_type="lower") * u.TeV
    assert result == expected


def test_compute_upper_radial_distance(mock_reader, hdf5_file_name, mocker):
    calculator = LimitCalculator(hdf5_file_name)

    # Mock the core distance data
    mock_core_distance = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    mock_reader.return_value.triggered_shower_data.core_distance_shower = mock_core_distance

    # Mock core_distance_bins property
    mock_bins = np.linspace(0, 100, 1000)
    mocker.patch.object(LimitCalculator, "core_distance_bins", property(lambda self: mock_bins))

    # Test with 20% loss fraction
    result = calculator.compute_upper_radial_distance(0.2)

    # Verify the result
    assert isinstance(result, u.Quantity)
    assert result.unit == u.m
    assert result.value > 0

    # Verify the histogram and compute_limits were used correctly
    hist, _ = np.histogram(mock_core_distance, bins=mock_bins)
    expected = calculator._compute_limits(hist, mock_bins, 0.2, limit_type="upper") * u.m
    assert result == expected


def test_plot_data(mock_reader, hdf5_file_name, mocker, tmp_path):
    calculator = LimitCalculator(hdf5_file_name)

    # Mock the necessary data attributes
    mock_reader.return_value.triggered_shower_data.core_distance_shower = np.array(
        [10.0, 20.0, 30.0]
    )
    mock_reader.return_value.triggered_shower_data.simulated_energy = np.array([1.0, 2.0, 3.0])
    mock_reader.return_value.triggered_shower_data.x_core_shower = np.array([1.0, 2.0, 3.0])
    mock_reader.return_value.triggered_shower_data.y_core_shower = np.array([1.0, 2.0, 3.0])
    mock_reader.return_value.triggered_data.angular_distance = np.array([0.5, 1.0, 1.5])

    # Mock matplotlib to avoid actual plotting
    mocker.patch("matplotlib.pyplot.figure")
    mocker.patch("matplotlib.pyplot.savefig")
    mocker.patch("matplotlib.pyplot.close")
    mock_create_plot = mocker.patch.object(calculator, "_create_plot")

    # Create test input parameters
    lower_energy_limit = 1.0 * u.TeV
    upper_radial_distance = 100.0 * u.m
    viewcone = 2.0 * u.deg

    # Test with output path
    calculator.plot_data(lower_energy_limit, upper_radial_distance, viewcone, output_path=tmp_path)

    # Verify _create_plot was called correct number of times
    assert mock_create_plot.call_count == 5

    # Test without output path
    mock_create_plot.reset_mock()
    calculator.plot_data(lower_energy_limit, upper_radial_distance, viewcone, output_path=None)

    # Verify _create_plot was called correct number of times
    assert mock_create_plot.call_count == 5


@pytest.fixture
def mock_figure(mocker):
    return mocker.patch("matplotlib.pyplot.figure")


@pytest.fixture
def mock_hist(mocker):
    return mocker.patch("matplotlib.pyplot.hist")


@pytest.fixture
def mock_hist2d(mocker):
    return mocker.patch("matplotlib.pyplot.hist2d")


@pytest.fixture
def mock_xlabel(mocker):
    return mocker.patch("matplotlib.pyplot.xlabel")


@pytest.fixture
def mock_ylabel(mocker):
    return mocker.patch("matplotlib.pyplot.ylabel")


@pytest.fixture
def mock_title(mocker):
    return mocker.patch("matplotlib.pyplot.title")


@pytest.fixture
def mock_tight_layout(mocker):
    return mocker.patch("matplotlib.pyplot.tight_layout")


@pytest.fixture
def mock_show(mocker):
    return mocker.patch("matplotlib.pyplot.show")


@pytest.fixture
def mock_colorbar(mocker):
    return mocker.patch("matplotlib.pyplot.colorbar")


@pytest.fixture
def mock_scatter(mocker):
    return mocker.patch("matplotlib.pyplot.scatter")


def test_create_plot_histogram(
    mock_reader,
    hdf5_file_name,
    mock_figure,
    mock_hist,
    mock_xlabel,
    mock_ylabel,
    mock_title,
    mock_tight_layout,
    mock_show,
):
    calculator = LimitCalculator(hdf5_file_name)

    # Mock matplotlib functions

    x_data = [1, 2, 3]
    bins = [0, 1, 2, 3]
    plot_params = {"color": "blue"}

    # Test histogram plot
    fig = calculator._create_plot(
        x_data=x_data,
        bins=bins,
        plot_type="histogram",
        plot_params=plot_params,
        labels={"x": "X Label", "y": "Y Label", "title": "Test Plot"},
    )

    mock_figure.assert_called_once()
    mock_hist.assert_called_once_with(x_data, bins=bins, color="blue")
    mock_xlabel.assert_called_once_with("X Label")
    mock_ylabel.assert_called_once_with("Y Label")
    mock_title.assert_called_once_with("Test Plot")
    mock_tight_layout.assert_called_once()
    mock_show.assert_called_once()
    assert fig == mock_figure.return_value


def test_create_plot_histogram2d(mock_reader, hdf5_file_name, mock_colorbar, mock_hist2d):
    calculator = LimitCalculator(hdf5_file_name)

    x_data = [1, 2, 3]
    y_data = [4, 5, 6]
    bins = [0, 1, 2, 3]
    plot_params = {"cmap": "viridis"}

    calculator._create_plot(
        x_data=x_data,
        y_data=y_data,
        bins=bins,
        plot_type="histogram2d",
        plot_params=plot_params,
        colorbar_label="Counts",
    )

    mock_hist2d.assert_called_once_with(x_data, y_data, bins=bins, cmap="viridis")
    mock_colorbar.assert_called_once_with(label="Counts")


def test_create_plot_scatter(mock_reader, hdf5_file_name, mock_scatter):
    calculator = LimitCalculator(hdf5_file_name)

    x_data = [1, 2, 3]
    y_data = [4, 5, 6]
    plot_params = {"c": "red"}

    calculator._create_plot(
        x_data=x_data, y_data=y_data, plot_type="scatter", plot_params=plot_params
    )

    mock_scatter.assert_called_once_with(x_data, y_data, c="red")


def test_create_plot_with_lines(mock_reader, hdf5_file_name, mocker):
    calculator = LimitCalculator(hdf5_file_name)

    mock_axvline = mocker.patch("matplotlib.pyplot.axvline")
    mock_axhline = mocker.patch("matplotlib.pyplot.axhline")

    calculator._create_plot(x_data=[1, 2, 3], lines={"x": 1.5, "y": 2.5})

    mock_axvline.assert_called_once_with(1.5, color="r", linestyle="--")
    mock_axhline.assert_called_once_with(2.5, color="r", linestyle="--")


def test_create_plot_with_scales(mock_reader, hdf5_file_name, mocker):
    calculator = LimitCalculator(hdf5_file_name)

    mock_xscale = mocker.patch("matplotlib.pyplot.xscale")
    mock_yscale = mocker.patch("matplotlib.pyplot.yscale")

    calculator._create_plot(x_data=[1, 2, 3], scales={"x": "log", "y": "log"})

    mock_xscale.assert_called_once_with("log")
    mock_yscale.assert_called_once_with("log")


def test_create_plot_save_file(mock_reader, hdf5_file_name, mocker, tmp_path):
    calculator = LimitCalculator(hdf5_file_name)

    mock_savefig = mocker.patch("matplotlib.pyplot.savefig")
    mock_close = mocker.patch("matplotlib.pyplot.close")
    mock_logger = mocker.patch.object(calculator, "_logger")

    output_file = tmp_path / "test_plot.png"

    calculator._create_plot(x_data=[1, 2, 3], output_file=output_file)

    mock_logger.info.assert_called_once_with(f"Saving plot to {output_file}")
    mock_savefig.assert_called_once_with(output_file, dpi=300, bbox_inches="tight")
    mock_close.assert_called_once()
