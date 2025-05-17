import json
import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import tables
from eventio.simtel import (
    ArrayEvent,
    MCEvent,
    MCRunHeader,
    MCShower,
    TrackingPosition,
    TriggerInformation,
)

from simtools.simtel.simtel_io_event_writer import SimtelIOEventDataWriter
from simtools.utils.geometry import calculate_circular_mean

OUTPUT_FILE_NAME = "output.h5"


@pytest.fixture
def mock_eventio_file(tmp_path):
    # Create a mock EventIO file path
    file_path = tmp_path / "mock_eventio_file.simtel.zst"
    file_path.touch()  # Create an empty file
    return str(file_path)


@pytest.fixture
def lookup_table_generator(mock_eventio_file, tmp_path):
    output_file = tmp_path / OUTPUT_FILE_NAME
    return SimtelIOEventDataWriter([mock_eventio_file], output_file, max_files=1)


@pytest.fixture
def mock_corsika_run_header(mocker):
    # Mock the get_corsika_run_header function
    mock_get_header = mocker.patch("simtools.simtel.simtel_io_event_writer.get_corsika_run_header")
    mock_get_header.return_value = {
        "direction": [0.0, 70.0 / 57.3],
        "particle_id": 1,
    }
    return mock_get_header


def create_mock_eventio_objects(alt_range, az_range):
    """
    Helper function to create mock EventIO objects with specified alt_range and az_range.
    """
    mc_run_header = MagicMock(spec=MCRunHeader)
    mc_run_header.parse.return_value = {
        "n_use": 1,
        "viewcone": [0.0, 10.0],
    }

    mc_shower = MagicMock(spec=MCShower)
    mc_shower.parse.return_value = {"energy": 1.0, "azimuth": 0.1, "altitude": 0.1, "shower": 0}

    mc_event = MagicMock(spec=MCEvent)
    mc_event.parse.return_value = {
        "shower_num": 10,
        "xcore": 0.1,
        "ycore": 0.1,
        "aweight": 1.0,
        "shower_azimuth": 0.5,
        "shower_altitude": 0.3,
    }

    trigger_info = MagicMock(spec=TriggerInformation)
    trigger_info.parse.return_value = {"triggered_telescopes": [1, 2, 3]}

    tracking_position = MagicMock(spec=TrackingPosition)
    tracking_position.parse.return_value = {
        "altitude_raw": alt_range,
        "azimuth_raw": az_range,
    }

    array_event = MagicMock(spec=ArrayEvent)
    array_event.__iter__.return_value = iter([trigger_info, tracking_position])

    return [mc_run_header, mc_shower, mc_event, array_event]


def validate_datasets(reduced_data, triggered_data, file_info, trigger_telescope_list_list):
    """
    Helper function to validate that datasets are not empty.
    """
    assert len(reduced_data.col("simulated_energy")) > 0
    assert len(triggered_data.col("triggered_id")) > 0
    assert len(triggered_data.col("array_altitudes")) > 0
    assert len(triggered_data.col("array_azimuths")) > 0
    assert len(trigger_telescope_list_list) > 0
    assert len(reduced_data.col("x_core")) > 0
    assert len(reduced_data.col("y_core")) > 0
    assert len(file_info.col("file_name")) > 0
    assert len(reduced_data.col("shower_azimuth")) > 0
    assert len(reduced_data.col("shower_altitude")) > 0


@patch("simtools.simtel.simtel_io_event_writer.EventIOFile", autospec=True)
def test_process_files(mock_eventio_class, lookup_table_generator, mock_corsika_run_header):
    mock_eventio_class.return_value.__enter__.return_value.__iter__.return_value = (
        create_mock_eventio_objects([0.1, 0.1], [0.1, 0.1])
    )
    lookup_table_generator.process_files()

    with tables.open_file(lookup_table_generator.output_file, mode="r") as hdf:
        data_group = hdf.root.data
        reduced_data = data_group.reduced_data
        triggered_data = data_group.triggered_data
        file_info = data_group.file_info
        trigger_telescope_list_list = data_group.trigger_telescope_list_list

        # Validate datasets using the helper function
        validate_datasets(reduced_data, triggered_data, file_info, trigger_telescope_list_list)


@patch("simtools.simtel.simtel_io_event_writer.EventIOFile", autospec=True)
def test_no_input_files(mock_eventio_class):
    with pytest.raises(TypeError, match="No input files provided."):
        SimtelIOEventDataWriter(None, None)


@patch("simtools.simtel.simtel_io_event_writer.EventIOFile", autospec=True)
def test_multiple_files(mock_eventio_class, tmp_path, mock_corsika_run_header):
    mock_eventio_class.return_value.__enter__.return_value.__iter__.return_value = (
        create_mock_eventio_objects([0.1, 0.1], [0.1, 0.1])
    )
    input_files = [tmp_path / f"mock_eventio_file_{i}.simtel.zst" for i in range(3)]
    for file in input_files:
        file.touch()

    output_file = tmp_path / OUTPUT_FILE_NAME
    lookup_table_generator = SimtelIOEventDataWriter(input_files, output_file, max_files=3)
    lookup_table_generator.process_files()

    with tables.open_file(lookup_table_generator.output_file, mode="r") as hdf:
        data_group = hdf.root.data
        reduced_data = data_group.reduced_data
        triggered_data = data_group.triggered_data
        file_info = data_group.file_info
        trigger_telescope_list_list = data_group.trigger_telescope_list_list

        # Validate datasets using the helper function
        validate_datasets(reduced_data, triggered_data, file_info, trigger_telescope_list_list)


@patch("simtools.simtel.simtel_io_event_writer.EventIOFile", autospec=True)
def test_process_files_with_different_alt_az_ranges(
    mock_eventio_class, lookup_table_generator, mock_corsika_run_header, caplog
):
    """
    Test processing files where alt_range and az_range are different and ensure
    logger info is generated.
    """
    mock_eventio_class.return_value.__enter__.return_value.__iter__.return_value = (
        create_mock_eventio_objects([0.1, 0.2], [0.3, 0.4])
    )

    lookup_table_generator.process_files()

    with tables.open_file(lookup_table_generator.output_file, mode="r") as hdf:
        data_group = hdf.root.data
        reduced_data = data_group.reduced_data
        triggered_data = data_group.triggered_data
        file_info = data_group.file_info
        trigger_telescope_list_list = data_group.trigger_telescope_list_list

        # Validate datasets using the helper function
        validate_datasets(reduced_data, triggered_data, file_info, trigger_telescope_list_list)


@patch("simtools.simtel.simtel_io_event_writer.tables")
def test_write_data(mock_tables, lookup_table_generator, mock_eventio_file):
    """Test _write_data method for both write and append modes."""
    # Mock the tables.open_file context manager
    mock_file = MagicMock()
    mock_tables.open_file.return_value.__enter__.return_value = mock_file

    # Mock data group
    mock_data_group = MagicMock()
    mock_file.create_group.return_value = mock_data_group
    mock_file.get_node.return_value = mock_data_group

    # Mock tables and arrays
    mock_reduced_table = MagicMock()
    mock_triggered_table = MagicMock()
    mock_file_info_table = MagicMock()
    mock_vlarray = MagicMock()

    # Setup return values for table creation/getting
    lookup_table_generator._tables = MagicMock(
        return_value=(mock_reduced_table, mock_triggered_table, mock_file_info_table)
    )
    mock_file.create_vlarray.return_value = mock_vlarray
    mock_file.get_node.return_value = mock_vlarray

    # Add some test data
    lookup_table_generator.event_data.simulated_energy = [1.0]
    lookup_table_generator.triggered_data.triggered_id = [1]
    lookup_table_generator.file_name = ["test.simtel.gz"]

    # Test write mode
    lookup_table_generator._write_data(mode="w")

    # Verify write mode calls
    mock_tables.open_file.assert_called_with(lookup_table_generator.output_file, mode="w")
    mock_file.create_group.assert_called_with("/", "data", "Data group")
    mock_file.create_vlarray.assert_called()

    # Test append mode
    mock_file.reset_mock()
    mock_tables.reset_mock()
    mock_file.__contains__.return_value = True

    lookup_table_generator._write_data(mode="a")

    # Verify append mode calls
    mock_tables.open_file.assert_called_with(lookup_table_generator.output_file, mode="a")
    mock_file.get_node.assert_called()


def test_writer_triggered_data(lookup_table_generator):
    """Test _writer_triggered_data method."""
    # Mock the triggered table and vlarray
    mock_triggered_table = MagicMock()
    mock_vlarray = MagicMock()
    mock_row = MagicMock()
    mock_triggered_table.row = mock_row
    mock_vlarray.nrows = 0

    # Setup test data
    lookup_table_generator.triggered_data.triggered_id = [1, 2]
    lookup_table_generator.triggered_data.array_altitudes = [0.1, 0.2]
    lookup_table_generator.triggered_data.array_azimuths = [0.3, 0.4]
    lookup_table_generator.triggered_data.trigger_telescope_list_list = [[1, 2], [3, 4]]

    # Call the method
    lookup_table_generator._writer_triggered_data(mock_triggered_table, mock_vlarray)

    # Verify the interactions
    assert mock_row.append.call_count == 2
    assert mock_vlarray.append.call_count == 2
    mock_triggered_table.flush.assert_called_once()

    # Test empty triggered data
    mock_triggered_table.reset_mock()
    mock_vlarray.reset_mock()
    lookup_table_generator.triggered_data.triggered_id = []

    lookup_table_generator._writer_triggered_data(mock_triggered_table, mock_vlarray)

    # Verify no interactions for empty data
    mock_triggered_table.flush.assert_not_called()
    mock_vlarray.append.assert_not_called()


def test_write_event_data(lookup_table_generator):
    """Test _write_event_data method."""
    # Mock reduced table
    mock_reduced_table = MagicMock()
    mock_row = MagicMock()
    mock_reduced_table.row = mock_row

    # Test with empty data
    lookup_table_generator.event_data.simulated_energy = []
    lookup_table_generator._write_event_data(mock_reduced_table)
    mock_reduced_table.flush.assert_not_called()
    mock_row.append.assert_not_called()

    # Test with actual data
    lookup_table_generator.event_data.simulated_energy = [1.0, 2.0]
    lookup_table_generator.event_data.shower_id = [1]  # Shorter than energy list
    lookup_table_generator.event_data.x_core = [0.1, 0.2]
    lookup_table_generator.event_data.y_core = [0.3, 0.4]
    lookup_table_generator.event_data.area_weight = [1.0, 1.0]
    lookup_table_generator.event_data.shower_azimuth = [0.5, 0.6]
    lookup_table_generator.event_data.shower_altitude = [0.7, 0.8]

    lookup_table_generator._write_event_data(mock_reduced_table)

    # Verify row values were set and appended for each energy value
    assert mock_row.append.call_count == 2
    mock_reduced_table.flush.assert_called_once()

    # Verify first row values
    assert mock_row.__setitem__.call_args_list[0][0] == ("shower_id", 1)
    assert mock_row.__setitem__.call_args_list[1][0] == ("simulated_energy", 1.0)
    assert mock_row.__setitem__.call_args_list[2][0] == ("x_core", 0.1)
    assert mock_row.__setitem__.call_args_list[3][0] == ("y_core", 0.3)
    assert mock_row.__setitem__.call_args_list[4][0] == ("area_weight", 1.0)
    assert mock_row.__setitem__.call_args_list[5][0] == ("shower_azimuth", 0.5)
    assert mock_row.__setitem__.call_args_list[6][0] == ("shower_altitude", 0.7)

    # Verify second row shower_id defaults to 0 since shower_id list is shorter
    assert mock_row.__setitem__.call_args_list[7][0] == ("shower_id", 0)


def test_tables(lookup_table_generator):
    """Test _tables method for both write and append modes."""
    # Mock HDF5 file and data group
    mock_hdf5_file = MagicMock()
    mock_data_group = MagicMock()
    mock_table = MagicMock()

    # Mock create_table and get_node methods
    mock_hdf5_file.create_table.return_value = mock_table
    mock_hdf5_file.get_node.return_value = mock_table

    # Test write mode
    mock_hdf5_file.__contains__.return_value = False
    reduced_table, triggered_table, file_names_table = lookup_table_generator._tables(
        mock_hdf5_file, mock_data_group, mode="w"
    )

    # Verify tables were created in write mode
    assert mock_hdf5_file.create_table.call_count == 3
    assert reduced_table == mock_table
    assert triggered_table == mock_table
    assert file_names_table == mock_table

    # Reset mocks
    mock_hdf5_file.reset_mock()
    mock_hdf5_file.__contains__.return_value = True

    # Test append mode
    reduced_table, triggered_table, file_names_table = lookup_table_generator._tables(
        mock_hdf5_file, mock_data_group, mode="a"
    )

    # Verify existing tables were retrieved in append mode
    assert mock_hdf5_file.get_node.call_count == 3
    assert reduced_table == mock_table
    assert triggered_table == mock_table
    assert file_names_table == mock_table

    # Verify correct table paths were used
    expected_paths = ["/data/reduced_data", "/data/triggered_data", "/data/file_info"]
    get_node_calls = [call[0][0] for call in mock_hdf5_file.get_node.call_args_list]
    assert get_node_calls == expected_paths


def test_process_array_event(lookup_table_generator):
    """Test _process_array_event method."""
    # Create mock array event with multiple objects
    mock_array_event = MagicMock()

    # Create mock trigger information
    mock_trigger = MagicMock(spec=TriggerInformation)
    mock_trigger.parse.return_value = {"triggered_telescopes": [1, 2, 3]}

    # Create mock tracking position
    mock_tracking = MagicMock(spec=TrackingPosition)
    mock_tracking.parse.return_value = {"altitude_raw": 0.5, "azimuth_raw": 1.2}

    # Setup mock array event to return trigger and tracking objects
    mock_array_event.__iter__.return_value = [
        mock_trigger,
        mock_tracking,
        mock_trigger,  # Second event
        mock_tracking,
    ]

    # Set initial shower info for triggered events
    lookup_table_generator.shower = {"shower": 1}
    lookup_table_generator.shower_id_offset = 0

    # Process the mock array event
    lookup_table_generator._process_array_event(mock_array_event)

    # Verify trigger information was processed
    assert len(lookup_table_generator.triggered_data.triggered_id) == 2
    assert lookup_table_generator.triggered_data.triggered_id == [1, 1]
    assert len(lookup_table_generator.triggered_data.trigger_telescope_list_list) == 2
    assert all(
        (tels == [1, 2, 3]).all()
        for tels in lookup_table_generator.triggered_data.trigger_telescope_list_list
    )

    # Verify tracking positions were processed (taking mean)
    assert len(lookup_table_generator.triggered_data.array_altitudes) == 1
    assert len(lookup_table_generator.triggered_data.array_azimuths) == 1
    assert all(alt == 0.5 for alt in lookup_table_generator.triggered_data.array_altitudes)
    assert all(az == 1.2 for az in lookup_table_generator.triggered_data.array_azimuths)


def test_process_array_event_empty(lookup_table_generator):
    """Test _process_array_event method with empty array event."""
    mock_array_event = MagicMock()
    mock_array_event.__iter__.return_value = []

    lookup_table_generator._process_array_event(mock_array_event)

    # Verify no data was added
    assert len(lookup_table_generator.triggered_data.triggered_id) == 0
    assert len(lookup_table_generator.triggered_data.array_altitudes) == 0
    assert len(lookup_table_generator.triggered_data.array_azimuths) == 0
    assert len(lookup_table_generator.triggered_data.trigger_telescope_list_list) == 0


def test_process_array_event_with_multiple_telescopes(lookup_table_generator):
    """Test _process_array_event method with multiple telescopes and positions."""
    mock_array_event = MagicMock()

    # Create mock trigger information objects with different telescope lists
    mock_trigger1 = MagicMock(spec=TriggerInformation)
    mock_trigger1.parse.return_value = {"triggered_telescopes": [1, 2]}
    mock_trigger2 = MagicMock(spec=TriggerInformation)
    mock_trigger2.parse.return_value = {"triggered_telescopes": [3, 4, 5]}

    # Create mock tracking position objects with different alt/az values
    mock_tracking1 = MagicMock(spec=TrackingPosition)
    mock_tracking1.parse.return_value = {"altitude_raw": 0.3, "azimuth_raw": 1.5}
    mock_tracking2 = MagicMock(spec=TrackingPosition)
    mock_tracking2.parse.return_value = {"altitude_raw": 0.4, "azimuth_raw": 1.7}
    mock_tracking3 = MagicMock(spec=TrackingPosition)
    mock_tracking3.parse.return_value = {"altitude_raw": 0.5, "azimuth_raw": 1.9}

    # Setup mock array event to return objects in specific order
    mock_array_event.__iter__.return_value = [
        mock_trigger1,  # First event
        mock_tracking1,
        mock_tracking2,
        mock_trigger2,  # Second event
        mock_tracking3,
    ]

    # Set initial shower info for triggered events
    lookup_table_generator.shower = {"shower": 42}
    lookup_table_generator.shower_id_offset = 1000

    # Process the mock array event
    lookup_table_generator._process_array_event(mock_array_event)

    # Verify trigger information was processed correctly
    assert len(lookup_table_generator.triggered_data.triggered_id) == 2
    assert lookup_table_generator.triggered_data.triggered_id == [1042, 1042]  # 42 + 1000

    # Verify telescope lists were stored correctly
    assert len(lookup_table_generator.triggered_data.trigger_telescope_list_list) == 2
    np.testing.assert_array_equal(
        lookup_table_generator.triggered_data.trigger_telescope_list_list[0], np.array([1, 2])
    )
    np.testing.assert_array_equal(
        lookup_table_generator.triggered_data.trigger_telescope_list_list[1], np.array([3, 4, 5])
    )

    # Verify tracking positions were processed correctly (using mean values)
    assert len(lookup_table_generator.triggered_data.array_altitudes) == 1
    assert len(lookup_table_generator.triggered_data.array_azimuths) == 1
    np.testing.assert_almost_equal(
        lookup_table_generator.triggered_data.array_altitudes[0],
        0.4,  # mean of [0.3, 0.4, 0.5]
    )
    # Test circular mean of azimuths
    expected_azimuth1 = calculate_circular_mean([1.5, 1.7, 1.9])
    np.testing.assert_almost_equal(
        lookup_table_generator.triggered_data.array_azimuths[0], expected_azimuth1
    )


def test_get_event_data(lookup_table_generator):
    """Test get_event_data method returns correct event data objects."""
    # Get initial data objects
    event_data, triggered_data = lookup_table_generator.get_event_data()

    # Verify returned objects are the correct type
    assert event_data == lookup_table_generator.event_data
    assert triggered_data == lookup_table_generator.triggered_data

    # Add some test data
    lookup_table_generator.event_data.simulated_energy = [1.0, 2.0]
    lookup_table_generator.triggered_data.triggered_id = [1, 2]

    # Get updated data objects
    event_data, triggered_data = lookup_table_generator.get_event_data()

    # Verify data is accessible and correct
    assert event_data.simulated_energy == [1.0, 2.0]
    assert triggered_data.triggered_id == [1, 2]

    # Verify the returned objects are references to the instance attributes
    assert id(event_data) == id(lookup_table_generator.event_data)
    assert id(triggered_data) == id(lookup_table_generator.triggered_data)


def test_get_preliminary_nsb_level(lookup_table_generator, caplog):
    """Test get_preliminary_nsb_level method."""

    with pytest.raises(AttributeError, match="Invalid file name."):
        lookup_table_generator._get_preliminary_nsb_level(None)

    with caplog.at_level(logging.WARNING):
        _nsb = lookup_table_generator._get_preliminary_nsb_level("test.simtel.gz")
    assert pytest.approx(_nsb) == 1.0
    assert "No NSB level found in file name" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _nsb = lookup_table_generator._get_preliminary_nsb_level("mostly_half_moon")
    assert pytest.approx(_nsb) == 2.0
    assert "NSB level set to hardwired value" in caplog.text


@patch("simtools.simtel.simtel_io_event_writer.filenode", autospec=True)
def test_write_metadata(mock_filenode, lookup_table_generator):
    """Test _write_metadata method."""
    mock_file = MagicMock()
    mock_node = MagicMock()
    mock_filenode.new_node.return_value = mock_node

    lookup_table_generator._write_metadata(mock_file, metadata=None)
    mock_filenode.new_node.assert_not_called()
    mock_node.write.assert_not_called()

    mock_filenode.reset_mock()
    mock_node.reset_mock()
    test_metadata = {"simulation_type": "gamma", "energy_range": [0.1, 100], "zenith": 20.0}
    lookup_table_generator._write_metadata(mock_file, metadata=test_metadata)
    mock_filenode.new_node.assert_called_once_with(mock_file, where="/", name="metadata")
    mock_node.write.assert_called_once_with(json.dumps(test_metadata).encode("utf-8"))
