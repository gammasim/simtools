from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from eventio.simtel import (
    ArrayEvent,
    MCEvent,
    MCRunHeader,
    MCShower,
    TrackingPosition,
    TriggerInformation,
)

from simtools.sim_events.writer import IOEventDataWriter

OUTPUT_FILE_NAME = "output.fits"
one_two_three = "LSTN-01,LSTN-02,MSTN-01"


@pytest.fixture
def mock_eventio_file(tmp_path):
    """Create a mock EventIO file path."""
    file_path = tmp_path / "mock_eventio_file.simtel.zst"
    file_path.touch()  # Create an empty file
    return str(file_path)


@pytest.fixture
def lookup_table_generator(mock_eventio_file):
    """Create IOEventDataWriter instance."""
    return IOEventDataWriter(input_files=[mock_eventio_file], max_files=1)


@pytest.fixture
def mock_corsika_run_header(mocker):
    """Mock the get_combined_eventio_run_header."""
    mock_get_header = mocker.patch(
        "simtools.simtel.simtel_io_event_writer.get_combined_eventio_run_header"
    )
    mock_get_header.return_value = {
        "direction": [0.0, 70.0 / 57.3],
        "particle_id": 1,
        "E_range": [0.003, 330.0],
        "viewcone": [0.0, 10.0],
        "core_range": [0.0, 1000.0],
    }
    return mock_get_header


@pytest.fixture
def mock_get_sim_telarray_telescope_id_to_telescope_name_mapping(mocker):
    """Mock the get_sim_telarray_telescope_id_to_telescope_name_mapping."""
    mock_get_mapping = mocker.patch(
        "simtools.simtel.simtel_io_event_writer."
        "get_sim_telarray_telescope_id_to_telescope_name_mapping"
    )
    mock_get_mapping.return_value = {
        1: "LSTN-01",
        2: "LSTN-02",
        3: "MSTN-01",
        4: "MSTN-02",
    }
    return mock_get_mapping


@pytest.fixture
def mock_read_sim_telarray_metadata(mocker):
    """Mock the read_sim_telarray_metadata function."""
    mock_metadata = mocker.patch(
        "simtools.simtel.simtel_io_event_writer.read_sim_telarray_metadata"
    )
    mock_metadata.return_value = {"nsb_integrated_flux": 22.24}, {}
    return mock_metadata


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
def test_process_files(
    mock_eventio_class,
    lookup_table_generator,
    mock_corsika_run_header,
    mock_get_sim_telarray_telescope_id_to_telescope_name_mapping,
    mock_read_sim_telarray_metadata,
):
    """Test processing of files and creation of tables."""
    # Create sequence that matches IOEventDataWriter expectations
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
    with pytest.raises(TypeError, match=r"No input files provided."):
        IOEventDataWriter(None, None)


@patch("simtools.simtel.simtel_io_event_writer.EventIOFile", autospec=True)
def test_multiple_files(
    mock_eventio_class,
    tmp_path,
    mock_corsika_run_header,
    mock_get_sim_telarray_telescope_id_to_telescope_name_mapping,
    mock_read_sim_telarray_metadata,
):
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

    writer = IOEventDataWriter(input_files=input_files, max_files=3)
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


def test_process_array_event(lookup_table_generator):
    """Test array event processing."""
    mock_array_event = MagicMock(spec=ArrayEvent)
    mock_array_event.event_id = 42

    mock_trigger = MagicMock(spec=TriggerInformation)
    mock_trigger.parse.return_value = {"triggered_telescopes": [1, 2, 3]}
    mock_tracking = MagicMock(spec=TrackingPosition)
    mock_tracking.parse.return_value = {"altitude_raw": 0.5, "azimuth_raw": 1.2}

    mock_array_event.__iter__.return_value = [mock_trigger, mock_tracking]

    lookup_table_generator.shower_data.append({"shower_id": 1, "event_id": 42, "file_id": 0})

    with patch.object(
        lookup_table_generator, "_map_telescope_names", return_value=one_two_three.split(",")
    ):
        lookup_table_generator._process_array_event(mock_array_event, 0)

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
    mock_array_event = create_array_event()
    lookup_table_generator.shower_data.append({"shower_id": 1, "event_id": 42, "file_id": 0})

    with patch.object(
        lookup_table_generator, "_map_telescope_names", return_value=one_two_three.split(",")
    ):
        lookup_table_generator._process_array_event(mock_array_event, 0)

    assert len(lookup_table_generator.trigger_data) == 1
    trigger_event = lookup_table_generator.trigger_data[0]
    assert trigger_event["shower_id"] == 1
    assert trigger_event["event_id"] == 42
    assert trigger_event["telescope_list"] == one_two_three


def test_get_nsb_level_from_file_name(lookup_table_generator):
    """Test parsing NSB levels from filenames."""
    assert lookup_table_generator._get_nsb_level_from_file_name(
        "dark_file.simtel.zst"
    ) == pytest.approx(0.24)

    assert lookup_table_generator._get_nsb_level_from_file_name(
        "half_nsb_file.simtel.zst"
    ) == pytest.approx(0.835)

    assert lookup_table_generator._get_nsb_level_from_file_name(
        "gamma_full_moon_file.simtel.zst"
    ) == pytest.approx(1.2)

    assert lookup_table_generator._get_nsb_level_from_file_name("file.simtel.zst") is None

    assert lookup_table_generator._get_nsb_level_from_file_name(
        "DARK_FILE.simtel.zst"
    ) == pytest.approx(0.24)


def test_get_nsb_level_from_file_name_invalid_input(lookup_table_generator):
    """Test NSB level parsing with invalid input."""
    with pytest.raises(AttributeError, match=r"Invalid file name."):
        lookup_table_generator._get_nsb_level_from_file_name(None)


def test_process_mc_event(lookup_table_generator):
    """Test processing of MC events."""
    lookup_table_generator.n_use = 2
    lookup_table_generator.shower_data = [
        {"shower_id": 1, "event_id": None},
        {"shower_id": 1, "event_id": None},
    ]

    mock_event = MagicMock(spec=MCEvent)
    mock_event.parse.return_value = {
        "event_id": 1001,
        "shower_num": 1,
        "xcore": 100.0,
        "ycore": 200.0,
        "aweight": 1.5,
    }

    lookup_table_generator._process_mc_event(mock_event)

    updated_event = lookup_table_generator.shower_data[1]  # event_id is 10001
    assert updated_event["event_id"] == 1001
    assert updated_event["x_core"] == pytest.approx(100.0)
    assert updated_event["y_core"] == pytest.approx(200.0)
    assert updated_event["area_weight"] == pytest.approx(1.5)


def test_process_mc_event_inconsistent_shower(lookup_table_generator):
    """Test processing MC event with inconsistent shower ID."""
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


def test_map_telescope_names(lookup_table_generator):
    """Test mapping of telescope IDs to names."""
    # Set up test mapping
    lookup_table_generator.telescope_id_to_name = {
        1: "LSTN-01",
        2: "LSTN-02",
        3: "MSTN-01",
        4: "MSTN-02",
    }

    # Test with known IDs
    telescope_ids = [1, 2, 3]
    expected = ["LSTN-01", "LSTN-02", "MSTN-01"]
    assert lookup_table_generator._map_telescope_names(telescope_ids) == expected

    # Test with unknown ID
    telescope_ids = [1, 99]
    expected = ["LSTN-01", "Unknown_99"]
    assert lookup_table_generator._map_telescope_names(telescope_ids) == expected

    # Test with empty list
    assert lookup_table_generator._map_telescope_names([]) == []

    # Test with all unknown IDs
    telescope_ids = [98, 99]
    expected = ["Unknown_98", "Unknown_99"]
    assert lookup_table_generator._map_telescope_names(telescope_ids) == expected


def test_process_mc_shower_from_iact_simple(lookup_table_generator):
    """Very simple test for _process_mc_shower_from_iact."""
    mock_eventio_object = MagicMock()
    mock_eventio_object.parse.return_value = {
        "n_reuse": 2,
        "event_number": 7,
        "total_energy": 42.0,
        "reuse_x": [100.0, 200.0],
        "reuse_y": [300.0, 400.0],
        "azimuth": 0.1,
        "zenith": 0.2,
    }

    lookup_table_generator._process_mc_shower_from_iact(mock_eventio_object, 1)

    assert len(lookup_table_generator.shower_data) == 2
    assert lookup_table_generator.shower_data[0]["shower_id"] == 7
    assert lookup_table_generator.shower_data[0]["event_id"] == 700
    assert lookup_table_generator.shower_data[1]["event_id"] == 701
    assert lookup_table_generator.shower_data[0]["simulated_energy"] == pytest.approx(42.0)
    assert lookup_table_generator.shower_data[0]["x_core"] == pytest.approx(1.0)
    assert lookup_table_generator.shower_data[1]["x_core"] == pytest.approx(2.0)
    assert lookup_table_generator.shower_data[0]["y_core"] == pytest.approx(3.0)
    assert lookup_table_generator.shower_data[1]["y_core"] == pytest.approx(4.0)
    assert lookup_table_generator.shower_data[0]["file_id"] == 1
    assert lookup_table_generator.shower_data[1]["file_id"] == 1
    assert lookup_table_generator.shower_data[0]["area_weight"] == pytest.approx(1.0)
    assert lookup_table_generator.shower_data[1]["area_weight"] == pytest.approx(1.0)


def test_process_file_info_else(monkeypatch, tmp_path):
    """Test _process_file_info for CORSIKA IACT file (run_info is None)."""
    file_path = tmp_path / "test.iact"
    file_path.touch()

    fake_run_header = {"x_scatter": 10000.0}
    fake_event_header = {
        "particle_id": 3,
        "energy_min": 0.5,
        "energy_max": 5.0,
        "zenith": 0.5,
        "azimuth": 1.0,
        "viewcone_inner_angle": 0.1,
        "viewcone_outer_angle": 0.2,
    }

    monkeypatch.setattr(
        "simtools.simtel.simtel_io_event_writer.get_combined_eventio_run_header",
        lambda f: None,
    )
    monkeypatch.setattr(
        "simtools.simtel.simtel_io_event_writer.get_corsika_run_and_event_headers",
        lambda f: (fake_run_header, fake_event_header),
    )

    writer = IOEventDataWriter([str(file_path)])
    writer._process_file_info(1, str(file_path))

    assert len(writer.file_info) == 1
    info = writer.file_info[0]
    assert info["file_name"] == str(file_path)
    assert info["file_id"] == 1
    assert info["particle_id"] == 3
    assert info["energy_min"] == pytest.approx(0.5)
    assert info["energy_max"] == pytest.approx(5.0)
    assert info["viewcone_min"] == pytest.approx(0.1)
    assert info["viewcone_max"] == pytest.approx(0.2)
    assert info["core_scatter_min"] == pytest.approx(0.0)
    assert info["core_scatter_max"] == pytest.approx(100.0)
    assert info["zenith"] == pytest.approx(28.64788975654116)
    assert info["azimuth"] == pytest.approx(57.29577951308232)
    assert info["nsb_level"] == pytest.approx(0.0)
