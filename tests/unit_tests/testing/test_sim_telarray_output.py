from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from simtools.testing import sim_telarray_output


@pytest.fixture
def mock_simtel_file():
    mock_file = MagicMock()
    mock_file.mc_run_headers = [{"n_showers": 100, "E_range": [0.1, 100]}]
    mock_file.__iter__.return_value = [
        {"mc_shower": {"energy": energy}} for energy in np.linspace(0.1, 100, 100)
    ]
    return mock_file


@pytest.fixture
def valid_sim_telarray_file_content():
    return {
        "photoelectron_sums": {
            "n_pe": np.array([10, 20, 30, 0, 0]),
            "photons_atm_qe": np.array([100, 200, 300, 0, 0]),
            "photons": np.array([200, 300, 400, 0, 0]),
        },
        "trigger_information": {"trigger_times": [1.0, 2.0, 3.0]},
        "mc_shower": {"energy": 10.0},
    }


@patch("simtools.testing.sim_telarray_output.SimTelFile")
def test_assert_n_showers_and_energy_range(mock_simtelfile_class, mock_simtel_file):
    mock_simtelfile_class.return_value.__enter__.return_value = mock_simtel_file

    assert sim_telarray_output.assert_n_showers_and_energy_range(Path("dummy_path"))


@patch("simtools.testing.sim_telarray_output.SimTelFile")
def test_assert_n_showers_and_energy_range_inconsistent_showers(
    mock_simtelfile_class, mock_simtel_file
):
    mock_simtel_file.mc_run_headers[0]["n_showers"] = 200  # Set an inconsistent number of showers
    mock_simtelfile_class.return_value.__enter__.return_value = mock_simtel_file

    assert not sim_telarray_output.assert_n_showers_and_energy_range(Path("dummy_path"))


@patch("simtools.testing.sim_telarray_output.SimTelFile")
def test_assert_n_showers_and_energy_range_out_of_range_energy(
    mock_simtelfile_class, mock_simtel_file
):
    mock_simtel_file.__iter__.return_value = [
        {"mc_shower": {"energy": energy}} for energy in np.linspace(0.05, 100.05, 100)
    ]  # Set energies slightly out of range
    mock_simtelfile_class.return_value.__enter__.return_value = mock_simtel_file

    assert not sim_telarray_output.assert_n_showers_and_energy_range(Path("dummy_path"))


@patch("simtools.testing.sim_telarray_output.SimTelFile")
def test_assert_expected_sim_telarray_output(
    mock_simtelfile_class, mock_simtel_file, valid_sim_telarray_file_content
):
    mock_simtel_file.__iter__.return_value = [valid_sim_telarray_file_content]
    mock_simtelfile_class.return_value.__enter__.return_value = mock_simtel_file

    expected_output = {"pe_sum": [5, 35], "trigger_time": [0.5, 3.5], "photons": [50, 350]}

    assert sim_telarray_output.assert_expected_sim_telarray_output(
        Path("dummy_path"), expected_output
    )


@patch("simtools.testing.sim_telarray_output.SimTelFile")
def test_assert_expected_sim_telarray_output_no_data(mock_simtelfile_class, mock_simtel_file):
    mock_simtel_file.__iter__.return_value = [
        {
            "photoelectron_sums": {
                "n_pe": np.array([0, 0, 0, 0, 0]),
                "photons_atm_qe": np.array([0, 0, 0, 0, 0]),
                "photons": np.array([0, 0, 0, 0, 0]),
            },
            "trigger_information": {"trigger_times": []},
        }
    ]
    mock_simtelfile_class.return_value.__enter__.return_value = mock_simtel_file

    expected_output = {"pe_sum": [5, 35], "trigger_time": [0.5, 3.5], "photons": [50, 350]}

    assert not sim_telarray_output.assert_expected_sim_telarray_output(
        Path("dummy_path"), expected_output
    )


@patch("simtools.testing.sim_telarray_output.SimTelFile")
def test_assert_expected_sim_telarray_output_out_of_range(mock_simtelfile_class, mock_simtel_file):
    mock_simtel_file.__iter__.return_value = [
        {
            "photoelectron_sums": {
                "n_pe": np.array([1, 2, 3, 4, 5]),
                "photons_atm_qe": np.array([10, 20, 30, 40, 50]),
                "photons": np.array([10, 20, 30, 40, 50]),
            },
            "trigger_information": {"trigger_times": [0.1, 0.2, 0.3]},
        }
    ]
    mock_simtelfile_class.return_value.__enter__.return_value = mock_simtel_file

    expected_output = {"pe_sum": [10, 20], "trigger_time": [1.0, 2.0], "photons": [100, 200]}

    assert not sim_telarray_output.assert_expected_sim_telarray_output(
        Path("dummy_path"), expected_output
    )


@patch("simtools.testing.sim_telarray_output.SimTelFile")
def test_check_output_from_sim_telarray(
    mock_simtelfile_class, mock_simtel_file, valid_sim_telarray_file_content
):
    mock_simtel_file.__iter__.return_value = [valid_sim_telarray_file_content]
    mock_simtel_file.mc_run_headers = [{"n_showers": 1, "E_range": [5.0, 15.0]}]
    mock_simtelfile_class.return_value.__enter__.return_value = mock_simtel_file

    expected_output = {"pe_sum": [5, 35], "trigger_time": [0.5, 3.5], "photons": [50, 350]}

    file = Path("dummy_path.zst")

    assert sim_telarray_output.assert_expected_sim_telarray_output(file, expected_output)


@patch("simtools.testing.sim_telarray_output.read_sim_telarray_metadata")
def test_assert_expected_simtel_metadata(mock_read_metadata):
    mock_read_metadata.return_value = (
        {"site": "North", "array_name": "test_array"},
        {"telescope_1": {"mirror_area": 100.0}},
    )

    expected_metadata = {"site": "North", "array_name": "test_array"}

    assert sim_telarray_output.assert_expected_sim_telarray_metadata(
        Path("dummy_path"), expected_metadata
    )


@patch("simtools.testing.sim_telarray_output.read_sim_telarray_metadata")
def test_assert_expected_simtel_metadata_mismatch(mock_read_metadata):
    mock_read_metadata.return_value = (
        {"site": "North", "array_name": "test_array"},
        {"telescope_1": {"mirror_area": 100.0}},
    )

    expected_metadata = {"site": "South", "array_name": "test_array"}

    assert not sim_telarray_output.assert_expected_sim_telarray_metadata(
        Path("dummy_path"), expected_metadata
    )


@patch("simtools.testing.sim_telarray_output.read_sim_telarray_metadata")
def test_assert_expected_simtel_metadata_missing_key(mock_read_metadata):
    mock_read_metadata.return_value = (
        {"site": "North"},
        {"telescope_1": {"mirror_area": 100.0}},
    )

    expected_metadata = {"site": "North", "missing_key": "value"}

    assert not sim_telarray_output.assert_expected_sim_telarray_metadata(
        Path("dummy_path"), expected_metadata
    )
