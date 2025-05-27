from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from astropy.table import Table
from eventio.simtel import (
    ArrayEvent,
    MCEvent,
    MCRunHeader,
    MCShower,
    TrackingPosition,
    TriggerInformation,
)

from simtools.simtel.simtel_io_event_writer import SimtelIOEventDataWriter

OUTPUT_FILE_NAME = "output.fits"
one_two_three = "1,2,3"


@pytest.fixture
def mock_eventio_file(tmp_path):
    """Create a mock EventIO file path."""
    file_path = tmp_path / "mock_eventio_file.simtel.zst"
    file_path.touch()  # Create an empty file
    return str(file_path)


@pytest.fixture
def lookup_table_generator(mock_eventio_file):
    """Create SimtelIOEventDataWriter instance."""
    return SimtelIOEventDataWriter(input_files=[mock_eventio_file], max_files=1)


@pytest.fixture
def mock_corsika_run_header(mocker):
    # Mock the get_corsika_run_header function
    mock_get_header = mocker.patch("simtools.simtel.simtel_io_event_writer.get_corsika_run_header")
    mock_get_header.return_value = {
        "direction": [0.0, 70.0 / 57.3],
        "particle_id": 1,
    }
    return mock_get_header


def create_mc_run_header():
    """Create mock MC run header."""
    mock_header = MagicMock(spec=MCRunHeader)
    mock_header.parse.return_value = {
        "n_use": 2,  # Important: Must be >= 1
        "viewcone": [0.0, 10.0],
    }
    return mock_header


def create_mc_shower(shower_id=1):
    """Create mock MC shower."""
    mock_shower = MagicMock(spec=MCShower)
    mock_shower.parse.return_value = {
        "energy": 1.0,
        "azimuth": 0.1,
        "altitude": 0.1,
        "shower": shower_id,  # Must match shower_num in mc_event
    }
    return mock_shower


def create_mc_event(shower_num=1, event_id=42):
    """Create mock MC event."""
    mock_event = MagicMock(spec=MCEvent)
    mock_event.parse.return_value = {
        "shower_num": shower_num,  # Must match shower in mc_shower
        "event_id": event_id,
        "xcore": 0.1,
        "ycore": 0.1,
        "aweight": 1.0,
    }
    return mock_event


def create_array_event():
    """Create mock array event."""
    mock_event = MagicMock(spec=ArrayEvent)
    mock_trigger = MagicMock(spec=TriggerInformation)
    mock_trigger.parse.return_value = {"triggered_telescopes": [1, 2, 3]}
    mock_tracking = MagicMock(spec=TrackingPosition)
    mock_tracking.parse.return_value = {"altitude_raw": 0.5, "azimuth_raw": 1.2}
    mock_event.__iter__.return_value = [mock_trigger, mock_tracking]
    mock_event.event_id = 42
    return mock_event


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
    """Test processing of files and creation of tables."""
    # Create sequence that matches SimtelIOEventDataWriter expectations
    mock_eventio_class.return_value.__enter__.return_value.__iter__.return_value = [
        create_mc_run_header(),
        create_mc_shower(shower_id=1),  # First shower
        create_mc_event(shower_num=1, event_id=0),  # First event of shower 1
        create_mc_event(shower_num=1, event_id=1),  # Second event of shower 1
        create_array_event(),  # Array event matching shower 1
    ]

    tables = lookup_table_generator.process_files()

    # Verify tables structure and content
    assert len(tables) == 3
    assert tables[0].meta["EXTNAME"] == "SHOWERS"
    assert tables[1].meta["EXTNAME"] == "TRIGGERS"
    assert tables[2].meta["EXTNAME"] == "FILE_INFO"

    # Verify shower data - should have 2 events with IDs 0 and 1
    assert len(tables[0]) == 2
    assert tables[0]["shower_id"][0] == 1
    assert tables[0]["event_id"][0] == 0  # First event ID
    assert tables[0]["event_id"][1] == 1  # Second event ID

    # Verify trigger data
    assert len(tables[1]) > 0
    assert "array_altitude" in tables[1].colnames
    assert "telescope_list" in tables[1].colnames
    assert one_two_three in tables[1]["telescope_list"]


def test_no_input_files():
    with pytest.raises(TypeError, match="No input files provided."):
        SimtelIOEventDataWriter(None, None)


@patch("simtools.simtel.simtel_io_event_writer.EventIOFile", autospec=True)
def test_multiple_files(mock_eventio_class, tmp_path, mock_corsika_run_header):
    """Test processing multiple input files."""
    # Create mock events for each file
    mock_eventio_class.return_value.__enter__.return_value.__iter__.return_value = [
        create_mc_run_header(),
        create_mc_shower(shower_id=1),
        create_mc_event(shower_num=1, event_id=10001),
        create_array_event(),
    ]

    # Create test files
    input_files = [str(tmp_path / f"mock_eventio_file_{i}.simtel.zst") for i in range(3)]
    for file in input_files:
        Path(file).touch()

    writer = SimtelIOEventDataWriter(input_files=input_files, max_files=3)
    tables = writer.process_files()

    assert len(tables) == 3
    assert len(tables[2]) == 3  # file_info table should have 3 entries


def create_test_data():
    """Create a complete set of test data matching schemas."""
    return {
        "shower_id": 1,
        "event_id": 42,
        "file_id": 0,
        "simulated_energy": 1.0,
        "x_core": 100.0,
        "y_core": 200.0,
        "shower_azimuth": 0.1,
        "shower_altitude": 1.2,
        "area_weight": 1.0,
    }


def test_write_fits(tmp_path, lookup_table_generator):
    """Test writing tables to FITS file."""
    output_file = tmp_path / "test.fits"

    # Add test data matching schema
    lookup_table_generator.shower_data.append(create_test_data())
    lookup_table_generator.trigger_data.append(
        {
            "shower_id": 1,
            "event_id": 42,
            "file_id": 0,
            "array_altitude": 1.2,
            "array_azimuth": 0.5,
            "telescope_list": one_two_three,
        }
    )
    lookup_table_generator.file_info.append(
        {
            "file_name": "test.simtel.gz",
            "file_id": 0,
            "particle_id": 1,
            "zenith": 20.0,
            "azimuth": 180.0,
            "nsb_level": 1.0,
        }
    )

    tables = lookup_table_generator._create_tables()
    lookup_table_generator.write(output_file, tables)

    assert output_file.exists()


def test_write_invalid_format(tmp_path, lookup_table_generator):
    """Test writing tables with invalid file format."""
    output_file = tmp_path / "test.txt"

    with pytest.raises(ValueError, match="Unsupported file format: .txt"):
        lookup_table_generator.write(output_file, [])


@patch("astropy.io.fits.HDUList.writeto")
def test_write_fits_gz(mock_write, tmp_path, lookup_table_generator):
    """Test writing tables to compressed FITS file."""
    output_file = tmp_path / "test.fits.gz"

    # Add test data matching schema
    lookup_table_generator.shower_data.append(create_test_data())
    lookup_table_generator.trigger_data.append(
        {
            "shower_id": 1,
            "event_id": 42,
            "file_id": 0,
            "array_altitude": 1.2,
            "array_azimuth": 0.5,
            "telescope_list": one_two_three,
        }
    )
    lookup_table_generator.file_info.append(
        {
            "file_name": "test.simtel.gz",
            "file_id": 0,
            "particle_id": 1,
            "zenith": 20.0,
            "azimuth": 180.0,
            "nsb_level": 1.0,
        }
    )

    tables = lookup_table_generator._create_tables()
    lookup_table_generator.write(output_file, tables)

    mock_write.assert_called_once()
    args = mock_write.call_args[0]
    assert str(args[0]) == str(output_file)


def test_process_array_event(lookup_table_generator):
    """Test array event processing."""
    mock_array_event = MagicMock(spec=ArrayEvent)
    mock_array_event.event_id = 42

    # Create mock trigger and tracking information
    mock_trigger = MagicMock(spec=TriggerInformation)
    mock_trigger.parse.return_value = {"triggered_telescopes": [1, 2, 3]}
    mock_tracking = MagicMock(spec=TrackingPosition)
    mock_tracking.parse.return_value = {"altitude_raw": 0.5, "azimuth_raw": 1.2}

    mock_array_event.__iter__.return_value = [mock_trigger, mock_tracking]

    # Add required shower data
    lookup_table_generator.shower_data.append({"shower_id": 1, "event_id": 42, "file_id": 0})

    # Process the event
    lookup_table_generator._process_array_event(mock_array_event, 0)

    # Verify data was added correctly
    assert len(lookup_table_generator.trigger_data) == 1
    trigger_event = lookup_table_generator.trigger_data[0]
    assert trigger_event["shower_id"] == 1
    assert trigger_event["event_id"] == 42
    assert trigger_event["telescope_list"] == one_two_three


def test_process_array_event_empty(lookup_table_generator):
    """Test _process_array_event method with empty array event."""
    mock_array_event = MagicMock(spec=ArrayEvent)
    mock_array_event.__iter__.return_value = []

    # Initial length of trigger data
    initial_len = len(lookup_table_generator.trigger_data)
    lookup_table_generator._process_array_event(mock_array_event, 0)

    # Verify no data was added
    assert len(lookup_table_generator.trigger_data) == initial_len


def test_process_array_event_with_trigger_data(lookup_table_generator):
    """Test processing array events and updating trigger data."""
    # Setup test event
    mock_array_event = create_array_event()

    # Add required shower data
    lookup_table_generator.shower_data.append({"shower_id": 1, "event_id": 42, "file_id": 0})

    # Process event
    lookup_table_generator._process_array_event(mock_array_event, 0)

    # Verify trigger data
    assert len(lookup_table_generator.trigger_data) == 1
    trigger_event = lookup_table_generator.trigger_data[0]
    assert trigger_event["shower_id"] == 1
    assert trigger_event["event_id"] == 42
    assert trigger_event["telescope_list"] == one_two_three


def test_get_preliminary_nsb_level(lookup_table_generator):
    """Test parsing NSB levels from filenames."""
    assert lookup_table_generator._get_preliminary_nsb_level("dark_file.simtel.zst") == 1.0

    assert lookup_table_generator._get_preliminary_nsb_level("half_nsb_file.simtel.zst") == 2.0

    assert (
        lookup_table_generator._get_preliminary_nsb_level("gamma_full_moon_file.simtel.zst") == 5.0
    )

    assert lookup_table_generator._get_preliminary_nsb_level("file.simtel.zst") == 1.0

    assert lookup_table_generator._get_preliminary_nsb_level("DARK_FILE.simtel.zst") == 1.0


def test_get_preliminary_nsb_level_invalid_input(lookup_table_generator):
    """Test NSB level parsing with invalid input."""
    with pytest.raises(AttributeError, match="Invalid file name."):
        lookup_table_generator._get_preliminary_nsb_level(None)


def test_process_mc_event(lookup_table_generator):
    """Test processing of MC events."""
    lookup_table_generator.n_use = 2
    lookup_table_generator.shower_data = [
        {"shower_id": 1, "event_id": None},
        {"shower_id": 1, "event_id": None},
    ]

    # Create mock MC event
    mock_event = MagicMock(spec=MCEvent)
    mock_event.parse.return_value = {
        "event_id": 1001,
        "shower_num": 1,
        "xcore": 100.0,
        "ycore": 200.0,
        "aweight": 1.5,
    }

    # Process event
    lookup_table_generator._process_mc_event(mock_event)

    # Verify data was updated correctly
    updated_event = lookup_table_generator.shower_data[1]  # event_id is 10001
    assert updated_event["event_id"] == 1001
    assert updated_event["x_core"] == 100.0
    assert updated_event["y_core"] == 200.0
    assert updated_event["area_weight"] == 1.5


def test_process_mc_event_inconsistent_shower(lookup_table_generator):
    """Test processing MC event with inconsistent shower ID."""
    # Setup test data
    lookup_table_generator.n_use = 2
    lookup_table_generator.shower_data = [
        {"shower_id": 1, "event_id": None},
        {"shower_id": 1, "event_id": None},
    ]

    # Create mock MC event with mismatched shower number
    mock_event = MagicMock(spec=MCEvent)
    mock_event.parse.return_value = {
        "event_id": 1001,
        "shower_num": 2,  # Different from shower_id in data
        "xcore": 100.0,
        "ycore": 200.0,
        "aweight": 1.5,
    }

    with pytest.raises(IndexError, match="Inconsistent shower and MC event data for shower id 2"):
        lookup_table_generator._process_mc_event(mock_event)

    mock_event.parse.return_value = {
        "event_id": 109999,
        "shower_num": 2,  # Different from shower_id in data
        "xcore": 100.0,
        "ycore": 200.0,
        "aweight": 1.5,
    }

    with pytest.raises(IndexError, match="Inconsistent shower and MC event data for shower id 2"):
        lookup_table_generator._process_mc_event(mock_event)


@patch("importlib.util.find_spec")
def test_write_hdf5_h5py_not_installed(mock_find_spec, tmp_path, lookup_table_generator):
    """Test _write_hdf5 when h5py is not installed."""
    mock_find_spec.return_value = None
    output_file = tmp_path / "test_1.hdf5"

    with pytest.raises(ImportError, match="h5py is required to write HDF5 files with Astropy."):
        lookup_table_generator._write_hdf5([], output_file)

    with pytest.raises(ImportError, match="h5py is required to write HDF5 files with Astropy."):
        lookup_table_generator.write(output_file, [])


@patch("importlib.util.find_spec")
@patch("astropy.table.Table.write")
def test_write_hdf5_single_table(mock_write, mock_find_spec, tmp_path, lookup_table_generator):
    """Test _write_hdf5 with a single table."""
    mock_find_spec.return_value = True
    output_file = tmp_path / "test_2.hdf5"

    table = Table()
    table.meta["EXTNAME"] = "TEST"
    tables = [table]

    lookup_table_generator._write_hdf5(tables, output_file)

    mock_write.assert_called_once_with(
        output_file,
        path="/TEST",
        format="hdf5",
        overwrite=True,
        serialize_meta=True,
        compression=True,
    )


@patch("importlib.util.find_spec")
@patch("astropy.table.Table.write")
def test_write_hdf5_multiple_tables(mock_write, mock_find_spec, tmp_path, lookup_table_generator):
    """Test _write_hdf5 with multiple tables."""
    mock_find_spec.return_value = True
    output_file = tmp_path / "test_3.hdf5"

    table1 = Table()
    table1.meta["EXTNAME"] = "TEST1"
    table2 = Table()
    table2.meta["EXTNAME"] = "TEST2"
    tables = [table1, table2]

    lookup_table_generator._write_hdf5(tables, output_file)

    assert mock_write.call_count == 2

    # First table called with overwrite=True
    mock_write.assert_any_call(
        output_file,
        path="/TEST1",
        format="hdf5",
        overwrite=True,
        serialize_meta=True,
        compression=True,
    )

    # Second table called with append=True
    mock_write.assert_any_call(
        output_file,
        path="/TEST2",
        format="hdf5",
        append=True,
        serialize_meta=True,
        compression=True,
    )
