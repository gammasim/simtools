from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest

from simtools.io import table_handler
from simtools.sim_events.writer import EventDataWriter

one_two_three = "LSTN-01,LSTN-02,MSTN-01"


@pytest.fixture
def mock_eventio_file(tmp_test_directory):
    """Create a mock EventIO file path."""
    file_path = Path(tmp_test_directory) / "mock_eventio_file.simtel.zst"
    file_path.touch()  # Create an empty file
    return str(file_path)


@pytest.fixture
def lookup_table_generator(mock_eventio_file):
    """Create EventDataWriter instance."""
    return EventDataWriter(input_files=[mock_eventio_file], max_files=1)


@pytest.fixture
def mock_get_sim_telarray_telescope_id_to_telescope_name_mapping(mocker):
    """Mock the get_sim_telarray_telescope_id_to_telescope_name_mapping."""
    mock_get_mapping = mocker.patch(
        "simtools.sim_events.formats.eventio_reader.get_sim_telarray_telescope_id_to_telescope_name_mapping"
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
        "simtools.sim_events.formats.eventio_reader.read_sim_telarray_metadata"
    )
    mock_metadata.return_value = {"nsb_integrated_flux": 22.24}, {}
    return mock_metadata


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


@patch("simtools.sim_events.formats.eventio_reader.EventIOFile", autospec=True)
def test_process_files(
    mock_eventio_class,
    lookup_table_generator,
    mock_get_sim_telarray_telescope_id_to_telescope_name_mapping,
    mock_read_sim_telarray_metadata,
    eventio_objects,
):
    # Create sequence that matches EventDataWriter expectations
    mock_eventio_class.return_value.__enter__.return_value.__iter__.return_value = [
        eventio_objects["run_header"](),
        eventio_objects["shower"](shower_id=1),  # First shower
        eventio_objects["event"](shower_num=1, event_id=0),  # First event of shower 1
        eventio_objects["event"](shower_num=1, event_id=1),  # Second event of shower 1
        eventio_objects["array_event"](),  # Array event matching shower 1
    ]

    tables = lookup_table_generator.process_files()

    assert mock_eventio_class.call_count == 1
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
    assert tables[2]["run_number"][0] == 123
    assert lookup_table_generator.get_simulation_input_metadata()[0]["run_number"] == 123


def test_no_input_files():
    with pytest.raises(TypeError, match=r"No input files provided."):
        EventDataWriter(None, None)


def test_default_processes_all_input_files():
    writer = EventDataWriter([f"input_{index}" for index in range(101)])

    assert writer.max_files == 101


def test_max_files_rejects_negative_values():
    with pytest.raises(ValueError, match="max_files must be non-negative"):
        EventDataWriter(["input"], max_files=-1)


def test_chunked_output_matches_non_chunked_output(
    lookup_table_generator,
    mock_get_sim_telarray_telescope_id_to_telescope_name_mapping,
    mock_read_sim_telarray_metadata,
    eventio_objects,
    mocker,
    tmp_test_directory,
):
    input_file = lookup_table_generator.input_files[0]
    reference_file = Path(tmp_test_directory) / "reference.hdf5"
    chunked_file = Path(tmp_test_directory) / "chunked.hdf5"
    eventio_file = mocker.patch("simtools.sim_events.formats.eventio_reader.EventIOFile")
    eventio_file.return_value.__enter__.return_value = [
        eventio_objects["run_header"](),
        eventio_objects["shower"](shower_id=1),
        eventio_objects["event"](shower_num=1, event_id=0),
        eventio_objects["event"](shower_num=1, event_id=1),
        eventio_objects["array_event"](),
    ]

    reference_tables = EventDataWriter([input_file]).process_files()
    table_handler.write_tables(reference_tables, reference_file)
    chunked_writer = EventDataWriter([input_file])
    table_handler.write_table_chunks(
        chunked_writer.iter_table_chunks(chunk_size=3),
        chunked_file,
    )

    assert chunked_writer.shower_data == []
    assert chunked_writer.trigger_data == []
    assert chunked_writer.file_info == []
    with h5py.File(reference_file) as reference, h5py.File(chunked_file) as chunked:
        assert set(reference) == set(chunked)
        assert dict(reference.attrs) == dict(chunked.attrs)
        for table_name in reference:
            assert reference[table_name].dtype == chunked[table_name].dtype
            np.testing.assert_array_equal(reference[table_name][:], chunked[table_name][:])
            assert dict(reference[table_name].attrs) == dict(chunked[table_name].attrs)


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


def test_create_chunk_rejects_unset_shower_fields(lookup_table_generator):
    shower = create_test_data()
    shower["x_core"] = None

    with pytest.raises(
        ValueError,
        match=r"Incomplete reduced event data.*SHOWERS.*unset required field.*x_core",
    ):
        lookup_table_generator._create_chunk("SHOWERS", [shower])


@pytest.fixture
def example_reader(monkeypatch):
    """A second format supplying known physical values without eventio objects."""
    from types import SimpleNamespace

    from simtools.sim_events.formats import registry

    shower = {
        "shower_id": 9,
        "event_id": 901,
        "simulated_energy": 2.5,
        "x_core": 10.0,
        "y_core": -20.0,
        "shower_azimuth": 180.0,
        "shower_altitude": 70.0,
        "area_weight": 0.5,
    }
    run = {
        "file_name": "input.example",
        "run_number": 7,
        "particle_id": 1,
        "spectral_index": -2.0,
        "energy_min": 1.0,
        "energy_max": 10.0,
        "viewcone_min": 0.0,
        "viewcone_max": 5.0,
        "core_scatter_min": 0.0,
        "core_scatter_max": 200.0,
        "zenith": 20.0,
        "azimuth": 180.0,
        "nsb_level": 0.24,
    }
    monkeypatch.setattr(registry, "_READERS", registry._READERS.copy())

    def factory(file):
        return SimpleNamespace(
            iter_records=lambda file_id: iter(
                [
                    ("SHOWERS", {**shower, "file_id": file_id}),
                    ("FILE_INFO", {**run, "file_name": str(file), "file_id": file_id}),
                ]
            ),
            read_metadata=lambda: {"software": "example", "file_name": str(file)},
        )

    registry.register_reader("example", factory)
    return factory


def test_second_format_preserves_common_tables(example_reader):
    generator = EventDataWriter(["first.example", "second.example"], file_format="example")
    tables = generator.process_files()
    showers, triggers, files = tables
    assert list(showers["file_id"]) == [0, 1]
    assert list(showers["shower_id"]) == [9, 9]
    assert list(showers["simulated_energy"]) == pytest.approx([2.5, 2.5])
    assert str(showers["simulated_energy"].unit) == "TeV"
    assert list(showers["x_core"]) == pytest.approx([10, 10])
    assert str(showers["x_core"].unit) == "m"
    assert len(triggers) == 0
    assert list(files["run_number"]) == [7, 7]
    assert list(files["file_name"]) == ["first.example", "second.example"]
    assert generator.get_simulation_input_metadata() == [
        {"software": "example", "file_name": "first.example"},
        {"software": "example", "file_name": "second.example"},
    ]


def test_selected_format_chunk_size(example_reader):
    generator = EventDataWriter(["input.example"], file_format="example")
    chunks = list(generator.iter_table_chunks(chunk_size=1))
    assert all(len(table) <= 1 for chunk in chunks for table in chunk)
    assert (
        sum(len(table) for chunk in chunks for table in chunk if table.meta["EXTNAME"] == "SHOWERS")
        == 1
    )
    assert generator.shower_data == generator.file_info == []


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_invalid_chunk_size(chunk_size):
    with pytest.raises(ValueError, match="chunk_size must be greater"):
        list(EventDataWriter([]).iter_table_chunks(chunk_size))


@pytest.mark.parametrize("rows", [[], [("FILE_INFO", {})], [("SHOWERS", create_test_data())]])
def test_reader_with_incomplete_file(rows, mocker):
    from types import SimpleNamespace

    mocker.patch(
        "simtools.sim_events.writer.get_reader",
        return_value=SimpleNamespace(iter_records=lambda file_id: iter(rows)),
    )
    with pytest.raises(ValueError, match="Incomplete reduced event data"):
        EventDataWriter(["input"]).process_files()
