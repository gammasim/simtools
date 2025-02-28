from unittest.mock import MagicMock, patch

import h5py
import pytest
from eventio.simtel import ArrayEvent, MCEvent, MCRunHeader, MCShower, TriggerInformation

from simtools.production_configuration.generate_reduced_datasets import ReducedDatasetGenerator


@pytest.fixture
def mock_eventio_file(tmp_path):
    # Create a mock EventIO file path
    file_path = tmp_path / "mock_eventio_file.simtel.zst"
    file_path.touch()  # Create an empty file
    return str(file_path)


@pytest.fixture
def lookup_table_generator(mock_eventio_file, tmp_path):
    output_file = tmp_path / "output.h5"
    return ReducedDatasetGenerator([mock_eventio_file], output_file, max_files=1)


def mock_eventio_objects():
    mc_run_header = MagicMock(spec=MCRunHeader)
    mc_run_header.parse.return_value = {
        "n_use": 1,
        "alt_range": [0.1, 0.2, 0.3],
        "az_range": [0.1, 0.2, 0.3],
    }

    mc_shower = MagicMock(spec=MCShower)
    mc_shower.parse.return_value = {"energy": 1.0, "azimuth": 0.1, "altitude": 0.1, "shower": 0}

    mc_event = MagicMock(spec=MCEvent)
    mc_event.parse.return_value = {"xcore": 0.1, "ycore": 0.1}

    trigger_info = MagicMock(spec=TriggerInformation)
    trigger_info.parse.return_value = {"telescopes_with_data": [1, 2, 3]}

    array_event = MagicMock(spec=ArrayEvent)
    array_event.__iter__.return_value = iter([trigger_info])

    return [mc_run_header, mc_shower, mc_event, array_event]


@patch("simtools.production_configuration.generate_lookup_tables.EventIOFile", autospec=True)
def test_process_files(mock_eventio_class, lookup_table_generator):
    mock_eventio_class.return_value.__enter__.return_value.__iter__.return_value = (
        mock_eventio_objects()
    )
    lookup_table_generator.process_files()

    with h5py.File(lookup_table_generator.output_file, "r") as hdf:
        data_group = hdf["data"]
        print('data_group["simulated"]', data_group["simulated"])
        assert "simulated" in data_group
        assert "shower_id_triggered" in data_group
        assert "triggered_energies" in data_group
        assert "num_triggered_telescopes" in data_group
        assert "trigger_telescope_list_list" in data_group
        assert "core_x" in data_group
        assert "core_y" in data_group
        assert "file_names" in data_group
        assert "shower_sim_azimuth" in data_group
        assert "shower_sim_altitude" in data_group
        assert "array_altitude" in data_group
        assert "array_azimuth" in data_group

        # Check that datasets are not empty
        assert len(data_group["simulated"]) > 0
        assert len(data_group["shower_id_triggered"]) > 0
        assert len(data_group["triggered_energies"]) > 0
        assert len(data_group["num_triggered_telescopes"]) > 0
        assert len(data_group["trigger_telescope_list_list"]) > 0
        assert len(data_group["core_x"]) > 0
        assert len(data_group["core_y"]) > 0
        assert len(data_group["file_names"]) > 0
        assert len(data_group["shower_sim_azimuth"]) > 0
        assert len(data_group["shower_sim_altitude"]) > 0
        assert len(data_group["array_altitude"]) > 0
        assert len(data_group["array_azimuth"]) > 0


@patch("simtools.production_configuration.generate_lookup_tables.EventIOFile", autospec=True)
def test_print_hdf5_file(mock_eventio_class, lookup_table_generator, capsys):
    mock_eventio_class.return_value.__enter__.return_value.__iter__.return_value = (
        mock_eventio_objects()
    )
    lookup_table_generator.process_files()
    lookup_table_generator.print_hdf5_file()

    captured = capsys.readouterr()
    assert "Datasets in file:" in captured.out
    assert "- simulated: shape=" in captured.out
    assert "- shower_id_triggered: shape=" in captured.out
    assert "- triggered_energies: shape=" in captured.out
    assert "- num_triggered_telescopes: shape=" in captured.out
    assert "- trigger_telescope_list_list: shape=" in captured.out
    assert "- core_x: shape=" in captured.out
    assert "- core_y: shape=" in captured.out
    assert "- file_names: shape=" in captured.out
    assert "- shower_sim_azimuth: shape=" in captured.out
    assert "- shower_sim_altitude: shape=" in captured.out
    assert "- array_altitude: shape=" in captured.out
    assert "- array_azimuth: shape=" in captured.out
