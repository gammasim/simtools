import logging
from unittest.mock import MagicMock, patch

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

from simtools.production_configuration.extract_mc_event_data import MCEventExtractor

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
    return MCEventExtractor([mock_eventio_file], output_file, max_files=1)


def create_mock_eventio_objects(alt_range, az_range):
    """
    Helper function to create mock EventIO objects with specified alt_range and az_range.
    """
    mc_run_header = MagicMock(spec=MCRunHeader)
    mc_run_header.parse.return_value = {
        "n_use": 1,
    }

    mc_shower = MagicMock(spec=MCShower)
    mc_shower.parse.return_value = {"energy": 1.0, "azimuth": 0.1, "altitude": 0.1, "shower": 0}

    mc_event = MagicMock(spec=MCEvent)
    mc_event.parse.return_value = {"xcore": 0.1, "ycore": 0.1}

    trigger_info = MagicMock(spec=TriggerInformation)
    trigger_info.parse.return_value = {"telescopes_with_data": [1, 2, 3]}

    tracking_position = MagicMock(spec=TrackingPosition)
    tracking_position.parse.return_value = {
        "altitude_raw": alt_range,
        "azimuth_raw": az_range,
    }

    array_event = MagicMock(spec=ArrayEvent)
    array_event.__iter__.return_value = iter([trigger_info, tracking_position])

    return [mc_run_header, mc_shower, mc_event, array_event]


def validate_datasets(reduced_data, triggered_data, file_names, trigger_telescope_list_list):
    """
    Helper function to validate that datasets are not empty.
    """
    assert len(reduced_data.col("simulated")) > 0
    assert len(triggered_data.col("shower_id_triggered")) > 0
    assert len(triggered_data.col("triggered_energies")) > 0
    assert len(triggered_data.col("array_altitudes")) > 0
    assert len(triggered_data.col("array_azimuths")) > 0
    assert len(trigger_telescope_list_list) > 0
    assert len(reduced_data.col("core_x")) > 0
    assert len(reduced_data.col("core_y")) > 0
    assert len(file_names.col("file_names")) > 0
    assert len(reduced_data.col("shower_sim_azimuth")) > 0
    assert len(reduced_data.col("shower_sim_altitude")) > 0


@patch("simtools.production_configuration.extract_mc_event_data.EventIOFile", autospec=True)
def test_process_files(mock_eventio_class, lookup_table_generator):
    mock_eventio_class.return_value.__enter__.return_value.__iter__.return_value = (
        create_mock_eventio_objects([0.1, 0.1], [0.1, 0.1])
    )
    lookup_table_generator.process_files()

    with tables.open_file(lookup_table_generator.output_file, mode="r") as hdf:
        data_group = hdf.root.data
        reduced_data = data_group.reduced_data
        triggered_data = data_group.triggered_data
        file_names = data_group.file_names
        trigger_telescope_list_list = data_group.trigger_telescope_list_list

        # Validate datasets using the helper function
        validate_datasets(reduced_data, triggered_data, file_names, trigger_telescope_list_list)


@patch("simtools.production_configuration.extract_mc_event_data.EventIOFile", autospec=True)
def test_print_dataset_information(mock_eventio_class, lookup_table_generator, capsys):
    mock_eventio_class.return_value.__enter__.return_value.__iter__.return_value = (
        create_mock_eventio_objects([0.1, 0.1], [0.1, 0.1])
    )
    lookup_table_generator.process_files()
    lookup_table_generator.print_dataset_information()

    captured = capsys.readouterr()
    assert "Datasets in file:" in captured.out
    assert "- file_names: shape=" in captured.out
    assert "- reduced_data: shape=" in captured.out
    assert "- triggered_data: shape=" in captured.out
    assert "- trigger_telescope_list_list: shape=" in captured.out
    assert "simulated" in captured.out
    assert "shower_sim_azimuth" in captured.out
    assert "shower_sim_altitude" in captured.out
    assert "array_altitudes" in captured.out
    assert "array_azimuths" in captured.out
    assert "shower_id_triggered" in captured.out
    assert "triggered_energies" in captured.out


@patch("simtools.production_configuration.extract_mc_event_data.EventIOFile", autospec=True)
def test_no_input_files(mock_eventio_class, tmp_path):
    output_file = tmp_path / OUTPUT_FILE_NAME
    lookup_table_generator = MCEventExtractor([], output_file, max_files=1)
    lookup_table_generator.process_files()

    assert not output_file.exists()


@patch("simtools.production_configuration.extract_mc_event_data.EventIOFile", autospec=True)
def test_multiple_files(mock_eventio_class, tmp_path):
    mock_eventio_class.return_value.__enter__.return_value.__iter__.return_value = (
        create_mock_eventio_objects([0.1, 0.1], [0.1, 0.1])
    )
    input_files = [tmp_path / f"mock_eventio_file_{i}.simtel.zst" for i in range(3)]
    for file in input_files:
        file.touch()

    output_file = tmp_path / OUTPUT_FILE_NAME
    lookup_table_generator = MCEventExtractor(input_files, output_file, max_files=3)
    lookup_table_generator.process_files()

    with tables.open_file(lookup_table_generator.output_file, mode="r") as hdf:
        data_group = hdf.root.data
        reduced_data = data_group.reduced_data
        triggered_data = data_group.triggered_data
        file_names = data_group.file_names
        trigger_telescope_list_list = data_group.trigger_telescope_list_list

        # Validate datasets using the helper function
        validate_datasets(reduced_data, triggered_data, file_names, trigger_telescope_list_list)


@patch("simtools.production_configuration.extract_mc_event_data.EventIOFile", autospec=True)
def test_process_files_with_different_alt_az_ranges(
    mock_eventio_class, lookup_table_generator, caplog
):
    """
    Test processing files where alt_range and az_range are different and ensure logger info is generated.
    """
    mock_eventio_class.return_value.__enter__.return_value.__iter__.return_value = (
        create_mock_eventio_objects([0.1, 0.2], [0.3, 0.4])
    )

    with caplog.at_level(logging.INFO):
        lookup_table_generator.process_files()

    assert "Telescopes have different tracking positions, applying mean." in caplog.text

    with tables.open_file(lookup_table_generator.output_file, mode="r") as hdf:
        data_group = hdf.root.data
        reduced_data = data_group.reduced_data
        triggered_data = data_group.triggered_data
        file_names = data_group.file_names
        trigger_telescope_list_list = data_group.trigger_telescope_list_list

        # Validate datasets using the helper function
        validate_datasets(reduced_data, triggered_data, file_names, trigger_telescope_list_list)
