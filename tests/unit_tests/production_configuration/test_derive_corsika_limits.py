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


@pytest.fixture
def mock_args_dict():
    """Create mock arguments dictionary."""
    return {
        "event_data_file": "data_files.hdf5",
        "telescope_ids": "telescope_ids.yml",
        "loss_fraction": 0.2,
        "plot_histograms": False,
        "output_file": "test_output.ecsv",
        "differential_loss_bins_per_decade": 0,
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


def test_generate_corsika_limits_grid(mocker, mock_args_dict, tmp_test_directory):
    """Test generate_corsika_limits_grid function with single production."""
    # Mock dependencies
    mock_collect = mocker.patch("simtools.io.ascii_handler.collect_data_from_file")
    mock_collect.side_effect = [
        {"telescope_configs": {"LST": [1, 2], "MST": [3, 4]}},
    ]

    mock_pool = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits.process_pool_map_ordered"
    )
    mock_pool.return_value = [
        {"primary_particle": "gamma", "array_name": "LST", "telescope_ids": [1, 2]},
        {"primary_particle": "gamma", "array_name": "MST", "telescope_ids": [3, 4]},
    ]

    mock_io = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits.io_handler.IOHandler"
    )
    mock_io.return_value.get_output_directory.return_value = tmp_test_directory

    mock_write = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits.write_results"
    )

    # Run function (single production mode)
    derive_corsika_limits.generate_corsika_limits_grid(mock_args_dict)

    # Verify calls
    assert mock_collect.call_count == 1
    mock_pool.assert_called_once()
    assert mock_write.call_count == 1


def test_generate_corsika_limits_grid_normalizes_telescope_ids(mocker, mock_args_dict):
    """Ensure numeric IDs are normalized to array-element names before processing."""
    mock_collect = mocker.patch("simtools.io.ascii_handler.collect_data_from_file")
    mock_collect.return_value = {"telescope_configs": {"LST": [1, 2]}}

    mock_pool = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits.process_pool_map_ordered"
    )
    mock_pool.return_value = [{}]
    mocker.patch("simtools.production_configuration.derive_corsika_limits.write_results")

    derive_corsika_limits.generate_corsika_limits_grid(mock_args_dict)

    expected_telescopes = [
        get_array_element_name_from_common_identifier(1),
        get_array_element_name_from_common_identifier(2),
    ]
    job_specs = mock_pool.call_args[0][1]
    assert job_specs[0]["telescope_ids"] == expected_telescopes
    assert job_specs[0]["differential_loss_bins_per_decade"] == 0


def test_process_file_passes_event_data_patterns_through(mocker):
    """Test _process_file passes glob patterns through to histogram resolution."""
    mock_histograms = mocker.MagicMock()
    mock_histogram_class = mocker.patch(SIM_EVENTS_HISTOGRAMS_PATH, return_value=mock_histograms)
    mocker.patch(COMPUTE_LOWER_ENERGY_LIMIT_PATH, return_value=1.0 * u.TeV)
    mocker.patch(COMPUTE_UPPER_RADIUS_LIMIT_PATH, return_value=100.0 * u.m)
    mocker.patch(COMPUTE_VIEWCONE_PATH, return_value=2.0 * u.deg)

    derive_corsika_limits._process_file("input/*.h5", "array_name", [1, 2], 0.2, False)

    mock_histogram_class.assert_called_once_with(
        "input/*.h5", array_name="array_name", telescope_list=[1, 2]
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


def test_generate_corsika_limits_grid_with_db_layouts(mocker, mock_args_dict, tmp_test_directory):
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

    mock_pool = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits.process_pool_map_ordered"
    )
    mock_pool.return_value = [
        {"primary_particle": "gamma", "array_name": "LST", "telescope_ids": [1, 2]},
        {"primary_particle": "gamma", "array_name": "MST", "telescope_ids": [3, 4]},
    ]

    mock_io = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits.io_handler.IOHandler"
    )
    mock_io.return_value.get_output_directory.return_value = tmp_test_directory

    mock_write = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits.write_results"
    )

    derive_corsika_limits.generate_corsika_limits_grid(args)

    mock_read_layouts.assert_called_once_with(
        args["array_layout_name"], args["site"], args["model_version"]
    )
    job_specs = mock_pool.call_args[0][1]
    assert len(job_specs) == 2  # 2 layouts
    assert mock_write.call_count == 1


def test_generate_corsika_limits_grid_with_array_element_list(
    mocker, mock_args_dict, tmp_test_directory
):
    """Test generate_corsika_limits_grid using inline array_element_list."""
    args = mock_args_dict.copy()
    args["array_element_list"] = ["LSTN-01", "LSTN-02", "MSTN-03"]
    args["telescope_ids"] = None

    mock_pool = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits.process_pool_map_ordered"
    )
    mock_pool.return_value = [
        {"primary_particle": "gamma", "array_name": "array_element_list"},
    ]

    mock_io = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits.io_handler.IOHandler"
    )
    mock_io.return_value.get_output_directory.return_value = tmp_test_directory

    mock_write = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits.write_results"
    )

    derive_corsika_limits.generate_corsika_limits_grid(args)

    job_specs = mock_pool.call_args[0][1]
    assert len(job_specs) == 1
    job_spec = job_specs[0]
    assert job_spec["array_name"] == "array_element_list"
    assert job_spec["telescope_ids"] == ["LSTN-01", "LSTN-02", "MSTN-03"]
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


def test_compute_viewcone(mocker):
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


def test_compute_lower_energy_limit(mocker):
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


def test_compute_upper_radius_limit(mocker):
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
    mock_histograms.file_info = {}

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
        "primary_particle": None,
        "zenith": None,
        "azimuth": None,
        "nsb_level": None,
        "lower_energy_limit": 1.0 * u.TeV,
        "upper_radius_limit": 100.0 * u.m,
        "viewcone_radius": 2.0 * u.deg,
    }

    mock_histogram_class.assert_called_once_with(
        MOCK_FILE_PATH, array_name="MockArray", telescope_list=[1, 2]
    )
    mock_histograms.fill.assert_called_once()
    mock_compute_lower_energy_limit.assert_called_once_with(mock_histograms, 0.2)
    mock_compute_upper_radius_limit.assert_called_once_with(mock_histograms, 0.2)
    mock_compute_viewcone.assert_called_once_with(mock_histograms, 0.2)


def test_process_file_with_differential_loss_per_energy_bin(mocker):
    """Test _process_file in differential-loss mode."""
    mock_histograms = mocker.MagicMock()
    mock_histograms.fill.return_value = None
    mock_histograms.file_info = {}

    mocker.patch(
        SIM_EVENTS_HISTOGRAMS_PATH,
        return_value=mock_histograms,
    )

    mock_compute_lower_energy_limit = mocker.patch(
        COMPUTE_LOWER_ENERGY_LIMIT_PATH,
        return_value=1.0 * u.TeV,
    )
    mock_compute_upper_radius_limit = mocker.patch(COMPUTE_UPPER_RADIUS_LIMIT_PATH)
    mock_compute_viewcone = mocker.patch(COMPUTE_VIEWCONE_PATH)
    mock_differential = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits._compute_differential_limits",
        return_value={
            "upper_radius_limit": 120.0 * u.m,
            "viewcone_radius": 3.0 * u.deg,
            "core_vs_energy_curve": {"x": [100.0, 120.0], "y": [0.1, 1.0]},
            "angular_distance_vs_energy_curve": {"x": [2.5, 3.0], "y": [0.1, 1.0]},
        },
    )

    result = derive_corsika_limits._process_file(
        file_path=MOCK_FILE_PATH,
        array_name="MockArray",
        telescope_ids=[1, 2],
        loss_fraction=0.2,
        plot_histograms=False,
        differential_loss_bins_per_decade=6,
    )

    assert result["lower_energy_limit"] == 1.0 * u.TeV
    assert result["upper_radius_limit"] == 120.0 * u.m
    assert result["viewcone_radius"] == 3.0 * u.deg
    assert result["core_vs_energy_curve"] == {"x": [100.0, 120.0], "y": [0.1, 1.0]}
    assert result["angular_distance_vs_energy_curve"] == {"x": [2.5, 3.0], "y": [0.1, 1.0]}

    mock_compute_lower_energy_limit.assert_called_once_with(mock_histograms, 0.2)
    mock_compute_upper_radius_limit.assert_not_called()
    mock_compute_viewcone.assert_not_called()
    mock_differential.assert_called_once_with(mock_histograms, 0.2, 6)


def test_process_file_with_plot_histograms(mocker, tmp_test_directory):
    """Test _process_file with plot_histograms=True using plotting module function."""
    mock_histograms = mocker.MagicMock()
    mock_histograms.fill.return_value = None
    mock_histograms.file_info = {}

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
        "primary_particle": None,
        "zenith": None,
        "azimuth": None,
        "nsb_level": None,
        "lower_energy_limit": 1.0 * u.TeV,
        "upper_radius_limit": 100.0 * u.m,
        "viewcone_radius": 2.0 * u.deg,
    }
    assert kwargs["array_name"] == "MockArray"


# Tests for multi-production and parallel execution support


def test_normalize_event_data_file_single_string():
    """Test _normalize_event_data_file with single string input."""
    result = derive_corsika_limits._normalize_event_data_file("pattern_*.hdf5")
    assert result == ["pattern_*.hdf5"]
    assert isinstance(result, list)


def test_normalize_event_data_file_list():
    """Test _normalize_event_data_file with list input."""
    patterns = ["pattern_1_*.hdf5", "pattern_2_*.hdf5"]
    result = derive_corsika_limits._normalize_event_data_file(patterns)
    assert result == patterns
    # Should preserve order
    assert result[0] == "pattern_1_*.hdf5"
    assert result[1] == "pattern_2_*.hdf5"


def test_normalize_event_data_file_invalid_type():
    """Test _normalize_event_data_file raises on invalid type."""
    with pytest.raises(TypeError):
        derive_corsika_limits._normalize_event_data_file(123)


def test_get_production_directory_name_readable_and_deterministic():
    """Test _get_production_directory_name generates readable deterministic names."""
    # Same inputs should produce same output when no collision exists
    name1 = derive_corsika_limits._get_production_directory_name("pattern_1_*.hdf5")
    name2 = derive_corsika_limits._get_production_directory_name("pattern_1_*.hdf5")
    assert name1 == name2

    # Different patterns should produce different readable names
    name3 = derive_corsika_limits._get_production_directory_name("pattern_2_*.hdf5")
    assert name1 != name3

    # Names should be filesystem-safe (no special chars)
    assert all(c.isalnum() or c == "_" for c in name1)
    assert name1 == "production_pattern_1"


def test_get_production_directory_name_appends_uuid_on_collision(mocker):
    """Test _get_production_directory_name appends UUID when names collide."""
    mock_uuid = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits.get_uuid",
        return_value="019d776b-e24c-741d-bc05-e3f6f7ec77c7",
    )

    name = derive_corsika_limits._get_production_directory_name(
        "pattern_1_*.hdf5",
        existing_names={"production_pattern_1"},
    )

    assert name == "production_pattern_1_019d776b-e24c-741d-bc05-e3f6f7ec77c7"
    mock_uuid.assert_called_once()


def test_execute_production_job_single_job(mocker):
    """Test _execute_production_job executes one job correctly."""
    mock_histograms = mocker.MagicMock()
    mock_histograms.fill.return_value = None
    mock_histograms.file_info = {}

    mocker.patch(
        SIM_EVENTS_HISTOGRAMS_PATH,
        return_value=mock_histograms,
    )

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

    job_spec = {
        "production_index": 0,
        "production_pattern": "pattern_*.hdf5",
        "array_name": "LST",
        "telescope_ids": ["LSTN-01"],
        "loss_fraction": 0.2,
        "plot_histograms": False,
        "output_subdir": None,
    }

    result = derive_corsika_limits._execute_production_job(job_spec)

    # Result should include production metadata
    assert result["production_index"] == 0
    assert result["event_data_file"] == "pattern_*.hdf5"
    assert result["array_name"] == "LST"
    assert "lower_energy_limit" in result
    assert "upper_radius_limit" in result
    assert "viewcone_radius" in result


def test_generate_corsika_limits_grid_multi_production(mocker, tmp_test_directory):
    """Test generate_corsika_limits_grid with multiple event_data_file patterns."""
    # Mock dependencies
    mock_collect = mocker.patch("simtools.io.ascii_handler.collect_data_from_file")
    mock_collect.return_value = {"telescope_configs": {"LST": ["LSTN-01"]}}

    mock_pool = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits.process_pool_map_ordered"
    )
    # Mock process_pool_map_ordered to return results directly
    mock_pool.return_value = [
        _pool_result(production_index=0, event_data_file="pattern_1_*.hdf5"),
        _pool_result(
            production_index=1,
            event_data_file="pattern_2_*.hdf5",
            lower_energy_limit=0.6 * u.TeV,
            upper_radius_limit=450.0 * u.m,
            viewcone_radius=5.5 * u.deg,
        ),
    ]

    mock_write = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits.write_results"
    )

    mock_io = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits.io_handler.IOHandler"
    )
    # Use actual tmp_test_directory to allow directory creation
    mock_io.return_value.get_output_directory.return_value = tmp_test_directory

    # Multi-production args
    args_dict = {
        "event_data_file": ["pattern_1_*.hdf5", "pattern_2_*.hdf5"],
        "telescope_ids": "telescope_ids.yml",
        "loss_fraction": 0.2,
        "plot_histograms": False,
        "output_file": "test_output.ecsv",
        "n_workers": 2,
    }

    derive_corsika_limits.generate_corsika_limits_grid(args_dict)

    # Verify process_pool_map_ordered was called with correct n_workers
    mock_pool.assert_called_once()
    call_kwargs = mock_pool.call_args[1]
    assert call_kwargs["max_workers"] == 2

    # Verify write_results was called with merged results
    mock_write.assert_called_once()
    written_results = mock_write.call_args[0][0]
    assert len(written_results) == 2  # Both productions merged


def test_generate_corsika_limits_grid_single_production_uses_pool(mocker, tmp_test_directory):
    """Test generate_corsika_limits_grid with single production uses process pool."""
    mock_collect = mocker.patch("simtools.io.ascii_handler.collect_data_from_file")
    mock_collect.return_value = {"telescope_configs": {"LST": ["LSTN-01"]}}

    mock_pool = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits.process_pool_map_ordered"
    )
    mock_pool.return_value = [_pool_result()]

    mock_execute_job = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits._execute_production_job"
    )
    mock_execute_job.return_value = {}

    mocker.patch("simtools.production_configuration.derive_corsika_limits.write_results")

    mock_io = mocker.patch(
        "simtools.production_configuration.derive_corsika_limits.io_handler.IOHandler"
    )
    mock_io.return_value.get_output_directory.return_value = tmp_test_directory

    args_dict = {
        "event_data_file": "pattern_*.hdf5",  # Single string, not list
        "telescope_ids": "telescope_ids.yml",
        "loss_fraction": 0.2,
        "plot_histograms": False,
        "output_file": "test_output.ecsv",
        "n_workers": 0,
    }

    derive_corsika_limits.generate_corsika_limits_grid(args_dict)

    mock_pool.assert_called_once()
    call_kwargs = mock_pool.call_args[1]
    assert call_kwargs["max_workers"] == 0
    mock_execute_job.assert_not_called()


def test_create_results_table_with_production_columns(mock_results):
    """Test _create_results_table includes production-origin columns for multi-production."""
    # Add production metadata to mock results
    for i, res in enumerate(mock_results):
        res["production_index"] = i
        res["event_data_file"] = f"pattern_{i}_*.hdf5"

    table = derive_corsika_limits._create_results_table(mock_results, loss_fraction=0.2)

    # Should include production-origin columns
    assert "production_index" in table.colnames
    assert "event_data_file" in table.colnames

    # Check values
    assert table["production_index"][0] == 0
    assert table["event_data_file"][0] == "pattern_0_*.hdf5"


def test_create_results_table_without_production_columns(mock_results):
    """Test _create_results_table with missing production metadata values."""
    # Results without production metadata (old format)
    table = derive_corsika_limits._create_results_table(mock_results, loss_fraction=0.2)

    # Production-origin columns are included and filled with None if missing
    assert "production_index" in table.colnames
    assert "event_data_file" in table.colnames
    assert table["production_index"][0] is None
    assert table["event_data_file"][0] is None

    # Standard columns should be present
    assert "primary_particle" in table.colnames
    assert "array_name" in table.colnames


def test_process_file_with_output_subdir(mocker, tmp_test_directory):
    """Test _process_file routes plots to specified output subdirectory."""
    mock_histograms = mocker.MagicMock()
    mock_histograms.fill.return_value = None
    mock_histograms.file_info = {}

    mocker.patch(
        SIM_EVENTS_HISTOGRAMS_PATH,
        return_value=mock_histograms,
    )

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

    output_subdir = tmp_test_directory / "production_pattern_1"

    derive_corsika_limits._process_file(
        file_path=MOCK_FILE_PATH,
        array_name="MockArray",
        telescope_ids=["LSTN-01"],
        loss_fraction=0.2,
        plot_histograms=True,
        output_subdir=output_subdir,
    )

    # Verify plot was called with the specified subdirectory
    mock_plot.assert_called_once()
    call_kwargs = mock_plot.call_args[1]
    assert call_kwargs["output_path"] == output_subdir
