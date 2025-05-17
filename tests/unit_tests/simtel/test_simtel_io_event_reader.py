from unittest.mock import Mock, patch

import numpy as np
import pytest

from simtools.simtel.simtel_io_event_reader import SimtelIOEventDataReader
from simtools.simtel.simtel_io_event_writer import ShowerEventData, TriggeredEventData


@pytest.fixture
def mock_hdf5_file():
    with patch("tables.open_file") as mock_open:
        mock_file = Mock()

        # Mock reduced data
        mock_reduced = Mock()
        mock_reduced.nrows = 2
        mock_reduced.col.side_effect = lambda x: {
            "simulated_energy": np.array([1.0, 2.0]),
            "x_core": np.array([100.0, 200.0]),
            "y_core": np.array([150.0, 250.0]),
            "shower_azimuth": np.array([0.1, 0.2]),
            "shower_altitude": np.array([1.0, 1.1]),
            "shower_id": np.array([1, 2]),
            "area_weight": np.array([1.0, 1.0]),
        }[x]
        # Mock triggered data
        mock_triggered = Mock()
        mock_triggered.nrows = 2
        mock_triggered.col.side_effect = lambda x: {
            "triggered_id": np.array([0, 1]),
            "array_altitudes": np.array([1.1, 1.2]),
            "array_azimuths": np.array([0.2, 0.3]),
            "telescope_list_index": np.array([0, 1]),
        }[x]

        mock_file.root.data.reduced_data = mock_reduced
        mock_file.root.data.triggered_data = mock_triggered
        mock_file.root.data.trigger_telescope_list_list = Mock()
        mock_file.root.data.trigger_telescope_list_list.__len__ = lambda x: 2
        mock_file.root.data.trigger_telescope_list_list.nrows = 2
        mock_file.root.data.trigger_telescope_list_list.__getitem__ = (
            lambda x, i: np.array([1, 2]) if i == 0 else np.array([2, 3])
        )
        mock_open.return_value.__enter__.return_value = mock_file

        yield mock_open


def test_init(mock_hdf5_file):
    mock_file = mock_hdf5_file.return_value.__enter__.return_value
    reader = SimtelIOEventDataReader(mock_file)
    assert isinstance(reader.shower_data, ShowerEventData)
    assert isinstance(reader.triggered_shower_data, ShowerEventData)
    assert isinstance(reader.triggered_data, TriggeredEventData)


def test_read_event_data(mock_hdf5_file):
    reader = SimtelIOEventDataReader(None)
    _, shower, triggered_shower, triggered = reader.read_event_data(mock_hdf5_file)

    assert len(shower.simulated_energy) == 2
    assert len(triggered_shower.simulated_energy) == 2
    assert len(triggered.array_azimuths) == 2


def test_get_mask_triggered_telescopes_no_filter(mock_hdf5_file):
    reader = SimtelIOEventDataReader(mock_hdf5_file)
    triggered_id = np.array([1, 2, 3])
    trigger_list = [np.array([1, 2]), np.array([2, 3]), np.array([1, 3])]

    filtered_id, indices = reader._get_mask_triggered_telescopes(None, triggered_id, trigger_list)

    np.testing.assert_array_equal(filtered_id, triggered_id)
    np.testing.assert_array_equal(indices, np.array([0, 1, 2]))


def test_get_mask_triggered_telescopes_with_filter(mock_hdf5_file):
    reader = SimtelIOEventDataReader(mock_hdf5_file)
    triggered_id = np.array([1, 2, 3])
    trigger_list = [np.array([1, 2]), np.array([2, 3]), np.array([1, 3])]

    filtered_id, indices = reader._get_mask_triggered_telescopes([2], triggered_id, trigger_list)

    np.testing.assert_array_equal(filtered_id, np.array([1, 2]))
    np.testing.assert_array_equal(indices, np.array([0, 1]))


def test_get_mask_triggered_telescopes_no_matches(mock_hdf5_file):
    reader = SimtelIOEventDataReader(mock_hdf5_file)
    triggered_id = np.array([1, 2, 3])
    trigger_list = [np.array([1, 2]), np.array([2, 3]), np.array([1, 3])]

    filtered_id, indices = reader._get_mask_triggered_telescopes([4], triggered_id, trigger_list)

    np.testing.assert_array_equal(filtered_id, np.array([]))
    np.testing.assert_array_equal(indices, np.array([]))


def test_print_dataset_information(mock_hdf5_file, capsys):
    reader = SimtelIOEventDataReader(mock_hdf5_file)
    reader.print_dataset_information(n_events=1)

    captured = capsys.readouterr()
    output = captured.out

    assert "Simulated energy (TeV)" in output
    assert "Core x (m)" in output
    assert "Core y (m)" in output
    assert "Shower azimuth (rad)" in output
    assert "Array azimuth (rad)" in output
    assert "Array altitude (rad)" in output
    assert "Triggered telescopes" in output

    assert str(int(reader.triggered_shower_data.x_core[0])) in output
    assert str(reader.triggered_data.array_azimuths[0]) in output


def test_print_event_table(mock_hdf5_file, capsys):
    reader = SimtelIOEventDataReader(mock_hdf5_file)

    # Call with small lines_per_page to trigger pagination
    reader.print_event_table()

    captured = capsys.readouterr()
    output = captured.out

    # Check header is present
    assert "Counter" in output
    assert "Simulated Energy (TeV)" in output
    assert "Triggered Telescopes" in output
    assert "Core distance shower (m)" in output

    # Check data is present (use 'int' to avoid floating point precision issues)
    assert str(int(reader.triggered_shower_data.simulated_energy[0])) in output
    assert str(reader.triggered_data.trigger_telescope_list_list[0]) in output
    assert str(int(reader.triggered_shower_data.core_distance_shower[0])) in output


def test_read_event_data_loads_all_fields(mock_hdf5_file):
    reader = SimtelIOEventDataReader(None)
    _, shower, triggered_shower, triggered = reader.read_event_data(mock_hdf5_file)

    # Test shower data fields
    assert len(shower.simulated_energy) == 2
    np.testing.assert_array_equal(shower.simulated_energy, [1.0, 2.0])
    np.testing.assert_array_equal(shower.x_core, [100.0, 200.0])
    np.testing.assert_array_equal(shower.y_core, [150.0, 250.0])
    np.testing.assert_array_equal(shower.shower_azimuth, [0.1, 0.2])
    np.testing.assert_array_equal(shower.shower_altitude, [1.0, 1.1])
    np.testing.assert_array_equal(shower.shower_id, [1, 2])
    np.testing.assert_array_equal(shower.area_weight, [1.0, 1.0])

    # Test triggered data fields
    assert len(triggered.array_azimuths) == 2
    np.testing.assert_array_equal(triggered.array_azimuths, [0.2, 0.3])
    np.testing.assert_array_equal(triggered.array_altitudes, [1.1, 1.2])
    np.testing.assert_array_equal(triggered.trigger_telescope_list_list[0], [1, 2])
    np.testing.assert_array_equal(triggered.trigger_telescope_list_list[1], [2, 3])


def test_read_event_data_handles_invalid_telescope_index(mock_hdf5_file):
    mock_file = mock_hdf5_file.return_value.__enter__.return_value
    mock_triggered = mock_file.root.data.triggered_data
    # Set an invalid telescope list index
    mock_triggered.col.side_effect = lambda x: {
        "triggered_id": np.array([0, 1]),
        "array_altitudes": np.array([1.1, 1.2]),
        "array_azimuths": np.array([0.2, 0.3]),
        "telescope_list_index": np.array([99, 1]),  # Invalid index 99
    }[x]

    reader = SimtelIOEventDataReader(None)
    _, _, _, triggered = reader.read_event_data(mock_hdf5_file)

    # First telescope list should be empty due to invalid index
    np.testing.assert_array_equal(triggered.trigger_telescope_list_list[0], [])
    # Second telescope list should be normal
    np.testing.assert_array_equal(triggered.trigger_telescope_list_list[1], [2, 3])
