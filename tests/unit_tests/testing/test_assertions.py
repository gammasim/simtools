#!/usr/bin/python3

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from simtools.constants import MODEL_PARAMETER_SCHEMA_PATH
from simtools.testing import assertions

logging.getLogger().setLevel(logging.DEBUG)


@pytest.fixture
def test_json_file():
    return Path("tests/resources/reference_point_altitude.json")


@pytest.fixture
def test_yaml_file():
    return MODEL_PARAMETER_SCHEMA_PATH / "num_gains.schema.yml"


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


def test_assert_file_type_json(test_json_file, test_yaml_file):
    assert assertions.assert_file_type("json", test_json_file)
    assert not assertions.assert_file_type("json", "tests/resources/does_not_exist.json")
    assert not assertions.assert_file_type("json", test_yaml_file)

    assert assertions.assert_file_type("json", Path(test_json_file))


def test_assert_file_type_yaml(test_json_file, test_yaml_file, caplog):
    assert assertions.assert_file_type("yaml", test_yaml_file)
    assert assertions.assert_file_type("yml", test_yaml_file)
    assert not assertions.assert_file_type("yml", "tests/resources/does_not_exit.schema.yml")

    assert not assertions.assert_file_type(
        "yaml", "tests/resources/telescope_positions-South-ground.ecsv"
    )


def test_assert_file_type_others(caplog):
    with caplog.at_level(logging.INFO):
        assert assertions.assert_file_type(
            "ecsv", "tests/resources/telescope_positions-South-ground.ecsv"
        )
    assert (
        "File type test is checking suffix only for tests/resources/"
        "telescope_positions-South-ground.ecsv (suffix: ecsv)" in caplog.text
    )


def test_assert_no_suffix():
    assert not assertions.assert_file_type("yml", "tests/resources/does_not_exit_yml")


@patch("eventio.simtel.simtelfile.SimTelFile")
def test_assert_n_showers_and_energy_range(mock_simtelfile_class, mock_simtel_file):
    mock_simtelfile_class.return_value.__enter__.return_value = mock_simtel_file

    assert assertions.assert_n_showers_and_energy_range(Path("dummy_path"))


@patch("eventio.simtel.simtelfile.SimTelFile")
def test_assert_n_showers_and_energy_range_inconsistent_showers(
    mock_simtelfile_class, mock_simtel_file
):
    mock_simtel_file.mc_run_headers[0]["n_showers"] = 200  # Set an inconsistent number of showers
    mock_simtelfile_class.return_value.__enter__.return_value = mock_simtel_file

    assert not assertions.assert_n_showers_and_energy_range(Path("dummy_path"))


@patch("eventio.simtel.simtelfile.SimTelFile")
def test_assert_n_showers_and_energy_range_out_of_range_energy(
    mock_simtelfile_class, mock_simtel_file
):
    mock_simtel_file.__iter__.return_value = [
        {"mc_shower": {"energy": energy}} for energy in np.linspace(0.05, 100.05, 100)
    ]  # Set energies slightly out of range
    mock_simtelfile_class.return_value.__enter__.return_value = mock_simtel_file

    assert not assertions.assert_n_showers_and_energy_range(Path("dummy_path"))


@patch("eventio.simtel.simtelfile.SimTelFile")
def test_assert_expected_output(
    mock_simtelfile_class, mock_simtel_file, valid_sim_telarray_file_content
):
    mock_simtel_file.__iter__.return_value = [valid_sim_telarray_file_content]
    mock_simtelfile_class.return_value.__enter__.return_value = mock_simtel_file

    expected_output = {"pe_sum": [5, 35], "trigger_time": [0.5, 3.5], "photons": [50, 350]}

    assert assertions.assert_expected_output(Path("dummy_path"), expected_output)


@patch("eventio.simtel.simtelfile.SimTelFile")
def test_assert_expected_output_no_data(mock_simtelfile_class, mock_simtel_file):
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

    assert not assertions.assert_expected_output(Path("dummy_path"), expected_output)


@patch("eventio.simtel.simtelfile.SimTelFile")
def test_assert_expected_output_out_of_range(mock_simtelfile_class, mock_simtel_file):
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

    assert not assertions.assert_expected_output(Path("dummy_path"), expected_output)


@patch("eventio.simtel.simtelfile.SimTelFile")
def test_check_output_from_sim_telarray(
    mock_simtelfile_class, mock_simtel_file, valid_sim_telarray_file_content
):
    mock_simtel_file.__iter__.return_value = [valid_sim_telarray_file_content]
    mock_simtel_file.mc_run_headers = [{"n_showers": 1, "E_range": [5.0, 15.0]}]
    mock_simtelfile_class.return_value.__enter__.return_value = mock_simtel_file

    expected_output = {"pe_sum": [5, 35], "trigger_time": [0.5, 3.5], "photons": [50, 350]}

    file = Path("dummy_path.zst")

    assert assertions.check_output_from_sim_telarray(file, expected_output)


@patch("eventio.simtel.simtelfile.SimTelFile")
def test_check_output_from_sim_telarray_invalid_file_extension(mock_simtelfile_class):
    file = Path("dummy_path.txt")
    expected_output = {"pe_sum": [5, 35], "trigger_time": [0.5, 3.5], "photons": [50, 350]}

    with pytest.raises(
        ValueError, match=r"Expected output file dummy_path.txt is not a zstd compressed file"
    ):
        assertions.check_output_from_sim_telarray(file, expected_output)
