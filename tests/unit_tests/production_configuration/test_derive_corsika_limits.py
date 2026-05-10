import astropy.units as u
import numpy as np
import pytest
from astropy.table import Table

import simtools.production_configuration.derive_corsika_limits as derive_corsika_limits
from simtools.utils.names import get_array_element_name_from_common_identifier

# Constants
SIM_EVENTS_HISTOGRAMS_PATH = (
    "simtools.production_configuration.derive_corsika_limits.EventDataHistograms"
)
COMPUTE_LOWER_ENERGY_LIMIT_PATH = (
    "simtools.production_configuration.derive_corsika_limits.compute_lower_energy_limit"
)
COMPUTE_UPPER_RADIUS_LIMIT_PATH = (
    "simtools.production_configuration.derive_corsika_limits.compute_upper_radius_limit"
)
COMPUTE_VIEWCONE_PATH = "simtools.production_configuration.derive_corsika_limits.compute_viewcone"
MOCK_FILE_PATH = "mock_file.fits"


@pytest.fixture
def mock_args_dict():
    """Create mock arguments dictionary."""
    return {
        "event_data_file": "data_files.hdf5",
        "telescope_ids": "telescope_ids.yml",
        "loss_fraction": 0.2,
        "plot_histograms": False,
        "output_file": "test_output.ecsv",
    }


@pytest.fixture
def mock_results():
    """Create mock results list."""
    return [
        {
            "primary_particle": "gamma",
            "telescope_ids": [1, 2],
            "zenith": 20.0 * u.deg,
            "azimuth": 180.0 * u.deg,
            "nsb_level": 1.0,
            "lower_energy_limit": 0.5 * u.TeV,
            "upper_radius_limit": 400.0 * u.m,
            "viewcone_radius": 5.0 * u.deg,
            "array_name": "LST",
            "layout": "LST",
        }
    ]


@pytest.fixture
def hdf5_file_name(tmp_test_directory):
    """Create temporary HDF5 file name."""
    return str(tmp_test_directory / "test_file.h5")


def test_generate_corsika_limits_grid(mocker, mock_args_dict):
    """Test generate_corsika_limits_grid function."""
    # Mock dependencies
    mock_collect = mocker.patch("simtools.io.ascii_handler.collect_data_from_file")
    mock_collect.side_effect = [
        {"telescope_configs": {"LST": [1, 2], "MST": [3, 4]}},
    ]

    mock_process = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits._process_file"
    )
    mock_write = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits.write_results"
    )

    # Run function
    derive_corsika_limits.generate_corsika_limits_grid(mock_args_dict)

    # Verify calls
    assert mock_collect.call_count == 1
    assert mock_process.call_count == 2  # 2 configs
    assert mock_write.call_count == 1


def test_generate_corsika_limits_grid_normalizes_telescope_ids(mocker, mock_args_dict):
    """Ensure numeric IDs are normalized to array-element names before processing."""
    mock_collect = mocker.patch("simtools.io.ascii_handler.collect_data_from_file")
    mock_collect.return_value = {"telescope_configs": {"LST": [1, 2]}}

    mock_process = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits._process_file"
    )
    mock_process.return_value = {}
    mocker.patch("simtools.production_configuration.derive_corsika_limits.write_results")

    derive_corsika_limits.generate_corsika_limits_grid(mock_args_dict)

    expected_telescopes = [
        get_array_element_name_from_common_identifier(1),
        get_array_element_name_from_common_identifier(2),
    ]
    call_args = mock_process.call_args[0]
    assert call_args[2] == expected_telescopes


def test_process_file(mocker):
    """Test _process_file function."""
    # Mock the EventDataHistograms class
    mock_histograms = mocker.MagicMock()
    mock_histogram_class = mocker.patch(SIM_EVENTS_HISTOGRAMS_PATH)
    mock_histogram_class.return_value = mock_histograms
    mocker.patch(
        "simtools.production_configuration.derive_corsika_limits.resolve_file_patterns",
        return_value=["test.fits"],
    )

    # Mock the individual limit computation functions
    mock_energy_limit = 1.0 * u.TeV
    mock_radius_limit = 100.0 * u.m
    mock_viewcone_limit = 2.0 * u.deg

    mocker.patch(
        COMPUTE_LOWER_ENERGY_LIMIT_PATH,
        return_value=mock_energy_limit,
    )
    mocker.patch(
        COMPUTE_UPPER_RADIUS_LIMIT_PATH,
        return_value=mock_radius_limit,
    )
    mocker.patch(
        COMPUTE_VIEWCONE_PATH,
        return_value=mock_viewcone_limit,
    )

    mocker.patch("simtools.io.io_handler.IOHandler")

    result = derive_corsika_limits._process_file("test.fits", "array_name", [1, 2], 0.2, False)

    expected_result = {
        "lower_energy_limit": mock_energy_limit,
        "upper_radius_limit": mock_radius_limit,
        "viewcone_radius": mock_viewcone_limit,
    }
    assert result == expected_result
    mock_histogram_class.assert_called_once_with(
        ["test.fits"], array_name="array_name", telescope_list=[1, 2]
    )
    mock_histograms.fill.assert_called_once()


def test_process_file_resolves_event_data_patterns(mocker):
    """Test _process_file resolves glob patterns before filling histograms."""
    mock_histograms = mocker.MagicMock()
    mock_histogram_class = mocker.patch(SIM_EVENTS_HISTOGRAMS_PATH, return_value=mock_histograms)
    mock_resolve = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits.resolve_file_patterns",
        return_value=["file_a.h5", "file_b.h5"],
    )
    mocker.patch(COMPUTE_LOWER_ENERGY_LIMIT_PATH, return_value=1.0 * u.TeV)
    mocker.patch(COMPUTE_UPPER_RADIUS_LIMIT_PATH, return_value=100.0 * u.m)
    mocker.patch(COMPUTE_VIEWCONE_PATH, return_value=2.0 * u.deg)

    derive_corsika_limits._process_file("input/*.h5", "array_name", [1, 2], 0.2, False)

    mock_resolve.assert_called_once_with("input/*.h5")
    mock_histogram_class.assert_called_once_with(
        ["file_a.h5", "file_b.h5"], array_name="array_name", telescope_list=[1, 2]
    )


def test_write_results(mocker, mock_args_dict, mock_results, tmp_test_directory):
    """Test write_results function."""
    mock_io = mocker.patch("simtools.io.io_handler.IOHandler")
    mock_io.return_value.get_output_directory.return_value = tmp_test_directory

    mock_dump = mocker.patch("simtools.data_model.metadata_collector.MetadataCollector.dump")

    derive_corsika_limits.write_results(mock_results, mock_args_dict)

    # Verify metadata was written
    mock_dump.assert_called_once()
    args = mock_dump.call_args[0]
    assert args[0] == mock_args_dict


def test_create_results_table(mock_results):
    """Test _create_results_table function."""
    table = derive_corsika_limits._create_results_table(mock_results, loss_fraction=0.2)
    table.info()

    assert isinstance(table, Table)
    assert len(table) == 1
    assert "zenith" in table.colnames
    assert table["zenith"].unit == u.deg
    assert table.meta["loss_fraction"] == pytest.approx(0.2)
    assert isinstance(table.meta["created"], str)
    assert "description" in table.meta


def test_round_value():
    """Test _round_value function for different key types."""

    # Test lower_energy_limit rounding
    assert derive_corsika_limits._round_value("lower_energy_limit", 1.2345) == pytest.approx(1.234)
    assert derive_corsika_limits._round_value("lower_energy_limit", 0.9876) == pytest.approx(0.987)
    assert derive_corsika_limits._round_value("lower_energy_limit", 2.0) == pytest.approx(2.0)

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


def test_generate_corsika_limits_grid_with_db_layouts(mocker, mock_args_dict):
    """Test generate_corsika_limits_grid using get_array_elements_from_db_for_layouts."""
    # Prepare args_dict to use array_layout_name
    args = mock_args_dict.copy()
    args["array_layout_name"] = ["LST", "MST"]
    args["site"] = "North"
    args["model_version"] = "v1.2.3"

    mock_read_layouts = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits."
        "get_array_elements_from_db_for_layouts"
    )
    mock_read_layouts.return_value = {"LST": [1, 2], "MST": [3, 4]}

    mock_process = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits._process_file"
    )
    mock_write = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits.write_results"
    )

    derive_corsika_limits.generate_corsika_limits_grid(args)

    mock_read_layouts.assert_called_once_with(
        args["array_layout_name"], args["site"], args["model_version"]
    )
    assert mock_process.call_count == 2  # 2 layouts
    assert mock_write.call_count == 1


def test_generate_corsika_limits_grid_with_array_element_list(mocker, mock_args_dict):
    """Test generate_corsika_limits_grid using inline array_element_list."""
    args = mock_args_dict.copy()
    args["array_element_list"] = ["LSTN-01", "LSTN-02", "MSTN-03"]
    args["telescope_ids"] = None

    mock_collect = mocker.patch("simtools.io.ascii_handler.collect_data_from_file")
    mock_process = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits._process_file"
    )
    mock_write = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits.write_results"
    )

    derive_corsika_limits.generate_corsika_limits_grid(args)

    mock_collect.assert_not_called()
    mock_process.assert_called_once()
    call_args = mock_process.call_args[0]
    assert call_args[1] == "array_element_list"
    assert call_args[2] == ["LSTN-01", "LSTN-02", "MSTN-03"]
    assert mock_write.call_count == 1


def test_generate_corsika_limits_grid_without_telescope_configuration(mock_args_dict):
    """Test generate_corsika_limits_grid raises if no telescope input is provided."""
    args = mock_args_dict.copy()
    args["array_layout_name"] = None
    args["array_element_list"] = None
    args["telescope_ids"] = None

    with pytest.raises(ValueError, match="No telescope configuration provided"):
        derive_corsika_limits.generate_corsika_limits_grid(args)


def test_compute_limits_lower():
    hist = np.array([1, 2, 3, 4, 5])
    bin_edges = np.array([0, 1, 2, 3, 4, 5])
    loss_fraction = 0.2

    with pytest.raises(ValueError, match="limit_type must be 'lower' or 'upper'"):
        derive_corsika_limits._compute_limits(hist, bin_edges, loss_fraction, limit_type="blabla")

    result = derive_corsika_limits._compute_limits(
        hist, bin_edges, loss_fraction, limit_type="lower"
    )
    assert result == 2


def test_compute_limits_upper():
    hist = np.array([5, 4, 3, 2, 1])
    bin_edges = np.array([0, 1, 2, 3, 4, 5])
    loss_fraction = 0.2

    result = derive_corsika_limits._compute_limits(
        hist, bin_edges, loss_fraction, limit_type="upper"
    )
    assert result == 3


def test_compute_limits_default_type():
    hist = np.array([1, 2, 3, 4, 5])
    bin_edges = np.array([0, 1, 2, 3, 4, 5])
    loss_fraction = 0.2

    result = derive_corsika_limits._compute_limits(hist, bin_edges, loss_fraction)
    assert result == 2


def test_compute_viewcone(hdf5_file_name, mocker):
    """Test compute_viewcone function with mocked histograms."""
    mock_hist = np.array([10, 8, 6, 4, 2])
    mock_bins = np.linspace(0, 20.0, 6)

    # Mock the histograms object
    mock_histograms = mocker.MagicMock()
    mock_histograms.histograms = {"angular_distance": {"histogram": mock_hist}}
    mock_histograms.view_cone_bins = mock_bins

    result = derive_corsika_limits.compute_viewcone(mock_histograms, 0.2)

    assert isinstance(result, u.Quantity)
    assert result.unit == u.deg
    assert result.value > 0

    expected = (
        derive_corsika_limits._compute_limits(mock_hist, mock_bins, 0.2, limit_type="upper") * u.deg
    )
    assert result.value == pytest.approx(expected.value)


def test_compute_lower_energy_limit(hdf5_file_name, mocker):
    """Test compute_lower_energy_limit function with mocked histograms."""
    mock_hist = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
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
        derive_corsika_limits._compute_limits(mock_hist, mock_bins, 0.2, limit_type="lower") * u.TeV
    )
    assert result == expected


def test_compute_upper_radius_limit(hdf5_file_name, mocker):
    """Test compute_upper_radius_limit function with mocked histograms."""
    mock_hist = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    mock_bins = np.linspace(0, 500, 6)

    # Mock the histograms object
    mock_histograms = mocker.MagicMock()
    mock_histograms.histograms = {"core_distance": {"histogram": mock_hist}}
    mock_histograms.core_distance_bins = mock_bins
    mock_histograms.file_info = {}

    result = derive_corsika_limits.compute_upper_radius_limit(mock_histograms, 0.2)

    assert isinstance(result, u.Quantity)
    assert result.unit == u.m
    assert result.value > 0

    expected = (
        derive_corsika_limits._compute_limits(mock_hist, mock_bins, 0.2, limit_type="upper") * u.m
    )
    assert result == expected


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


def test_process_file_with_mocked_histograms(mocker):
    """Test _process_file with mocked EventDataHistograms."""
    mock_histograms = mocker.MagicMock()
    mock_histograms.fill.return_value = None
    mocker.patch(
        "simtools.production_configuration.derive_corsika_limits.resolve_file_patterns",
        return_value=[MOCK_FILE_PATH],
    )

    mock_histogram_class = mocker.patch(
        SIM_EVENTS_HISTOGRAMS_PATH,
        return_value=mock_histograms,
    )

    mock_compute_lower_energy_limit = mocker.patch(
        COMPUTE_LOWER_ENERGY_LIMIT_PATH,
        return_value=1.0 * u.TeV,
    )
    mock_compute_upper_radius_limit = mocker.patch(
        COMPUTE_UPPER_RADIUS_LIMIT_PATH,
        return_value=100.0 * u.m,
    )
    mock_compute_viewcone = mocker.patch(
        COMPUTE_VIEWCONE_PATH,
        return_value=2.0 * u.deg,
    )

    result = derive_corsika_limits._process_file(
        file_path=MOCK_FILE_PATH,
        array_name="MockArray",
        telescope_ids=[1, 2],
        loss_fraction=0.2,
        plot_histograms=False,
    )

    assert result == {
        "lower_energy_limit": 1.0 * u.TeV,
        "upper_radius_limit": 100.0 * u.m,
        "viewcone_radius": 2.0 * u.deg,
    }

    mock_histogram_class.assert_called_once_with(
        [MOCK_FILE_PATH], array_name="MockArray", telescope_list=[1, 2]
    )
    mock_histograms.fill.assert_called_once()
    mock_compute_lower_energy_limit.assert_called_once_with(mock_histograms, 0.2)
    mock_compute_upper_radius_limit.assert_called_once_with(mock_histograms, 0.2)
    mock_compute_viewcone.assert_called_once_with(mock_histograms, 0.2)


def test_process_file_with_plot_histograms(mocker, tmp_test_directory):
    """Test _process_file with plot_histograms=True using plotting module function."""
    mock_histograms = mocker.MagicMock()
    mock_histograms.fill.return_value = None
    mocker.patch(
        "simtools.production_configuration.derive_corsika_limits.resolve_file_patterns",
        return_value=[MOCK_FILE_PATH],
    )

    mocker.patch(
        SIM_EVENTS_HISTOGRAMS_PATH,
        return_value=mock_histograms,
    )

    mock_io_handler = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits.io_handler.IOHandler"
    )
    mock_io_handler.return_value.get_output_directory.return_value = tmp_test_directory

    mocker.patch(
        COMPUTE_LOWER_ENERGY_LIMIT_PATH,
        return_value=1.0 * u.TeV,
    )
    mocker.patch(
        COMPUTE_UPPER_RADIUS_LIMIT_PATH,
        return_value=100.0 * u.m,
    )
    mocker.patch(
        COMPUTE_VIEWCONE_PATH,
        return_value=2.0 * u.deg,
    )

    mock_plot = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits.plot_simtel_event_histograms.plot"
    )

    derive_corsika_limits._process_file(
        file_path=MOCK_FILE_PATH,
        array_name="MockArray",
        telescope_ids=[1, 2],
        loss_fraction=0.2,
        plot_histograms=True,
    )

    mock_plot.assert_called_once()
    args, kwargs = mock_plot.call_args
    # First positional argument should be the histograms instance
    assert args[0] is mock_histograms.histograms
    assert kwargs["output_path"] == tmp_test_directory
    assert kwargs["limits"] == {
        "lower_energy_limit": 1.0 * u.TeV,
        "upper_radius_limit": 100.0 * u.m,
        "viewcone_radius": 2.0 * u.deg,
    }
    assert kwargs["array_name"] == "MockArray"
