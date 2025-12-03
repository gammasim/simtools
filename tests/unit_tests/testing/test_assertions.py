import gzip
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from simtools.constants import MODEL_PARAMETER_SCHEMA_PATH
from simtools.testing import assertions
from simtools.testing.assertions import check_plain_log

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


@patch("simtools.testing.assertions.read_sim_telarray_metadata")
def test_assert_expected_simtel_metadata(mock_read_metadata):
    mock_read_metadata.return_value = (
        {"site": "North", "array_name": "test_array"},
        {"telescope_1": {"mirror_area": 100.0}},
    )

    expected_metadata = {"site": "North", "array_name": "test_array"}

    assert assertions.assert_expected_simtel_metadata(Path("dummy_path"), expected_metadata)


@patch("simtools.testing.assertions.read_sim_telarray_metadata")
def test_assert_expected_simtel_metadata_mismatch(mock_read_metadata):
    mock_read_metadata.return_value = (
        {"site": "North", "array_name": "test_array"},
        {"telescope_1": {"mirror_area": 100.0}},
    )

    expected_metadata = {"site": "South", "array_name": "test_array"}

    assert not assertions.assert_expected_simtel_metadata(Path("dummy_path"), expected_metadata)


@patch("simtools.testing.assertions.read_sim_telarray_metadata")
def test_assert_expected_simtel_metadata_missing_key(mock_read_metadata):
    mock_read_metadata.return_value = (
        {"site": "North"},
        {"telescope_1": {"mirror_area": 100.0}},
    )

    expected_metadata = {"site": "North", "missing_key": "value"}

    assert not assertions.assert_expected_simtel_metadata(Path("dummy_path"), expected_metadata)


@patch("simtools.testing.assertions.assert_n_showers_and_energy_range")
@patch("simtools.testing.assertions.assert_expected_output")
@patch("simtools.testing.assertions.assert_expected_simtel_metadata")
def test_check_output_from_sim_telarray_with_both_checks(
    mock_assert_metadata, mock_assert_output, mock_assert_n_showers
):
    mock_assert_n_showers.return_value = True
    mock_assert_output.return_value = True
    mock_assert_metadata.return_value = True

    file = Path("dummy_path.zst")
    file_test = {
        "expected_output": {"pe_sum": [5, 35]},
        "expected_simtel_metadata": {"site": "North"},
    }

    assert assertions.check_output_from_sim_telarray(file, file_test)


@patch("simtools.testing.assertions.assert_n_showers_and_energy_range")
@patch("simtools.testing.assertions.assert_expected_output")
def test_check_output_from_sim_telarray_output_only(mock_assert_output, mock_assert_n_showers):
    mock_assert_n_showers.return_value = True
    mock_assert_output.return_value = True

    file = Path("dummy_path.zst")
    file_test = {"expected_output": {"pe_sum": [5, 35]}}

    assert assertions.check_output_from_sim_telarray(file, file_test)


@patch("simtools.testing.assertions.assert_n_showers_and_energy_range")
@patch("simtools.testing.assertions.assert_expected_simtel_metadata")
def test_check_output_from_sim_telarray_metadata_only(mock_assert_metadata, mock_assert_n_showers):
    mock_assert_n_showers.return_value = True
    mock_assert_metadata.return_value = True

    file = Path("dummy_path.zst")
    file_test = {"expected_simtel_metadata": {"site": "North"}}

    assert assertions.check_output_from_sim_telarray(file, file_test)


@patch("simtools.testing.assertions.assert_n_showers_and_energy_range")
def test_check_output_from_sim_telarray_no_checks(mock_assert_n_showers):
    mock_assert_n_showers.return_value = True

    file = Path("dummy_path.zst")
    file_test = {}

    assert assertions.check_output_from_sim_telarray(file, file_test)


@patch("simtools.testing.assertions.assert_n_showers_and_energy_range")
@patch("simtools.testing.assertions.assert_expected_output")
def test_check_output_from_sim_telarray_failed_output(mock_assert_output, mock_assert_n_showers):
    mock_assert_n_showers.return_value = True
    mock_assert_output.return_value = False

    file = Path("dummy_path.zst")
    file_test = {"expected_output": {"pe_sum": [5, 35]}}

    assert not assertions.check_output_from_sim_telarray(file, file_test)


def test_check_simulation_logs_no_patterns():
    file_test = {}
    result = assertions.check_simulation_logs(Path("dummy.tar.gz"), file_test)
    assert result


def test_check_simulation_logs_not_tar_file(tmp_test_directory):
    not_tar = Path(tmp_test_directory) / "not_a_tar.txt"
    not_tar.write_text("not a tar file", encoding="utf-8")
    file_test = {"expected_log_output": {"pattern": ["test"]}}

    with pytest.raises(ValueError, match=r"is not a tar file"):
        assertions.check_simulation_logs(not_tar, file_test)


def test_check_simulation_logs_success(tmp_test_directory, safe_tar_open):
    tmp_path = Path(tmp_test_directory)
    tar_path = tmp_path / "test_logs.tar.gz"
    log_content = b"Log line with pattern_A\nAnother line\nLine with pattern_B\n"

    with safe_tar_open(tar_path, "w:gz") as tar:
        log_gz = tmp_path / "test.log.gz"
        with gzip.open(log_gz, "wb") as gz:
            gz.write(log_content)
        tar.add(log_gz, arcname="test.log.gz")

    file_test = {"expected_log_output": {"pattern": ["pattern_A", "pattern_B"]}}
    result = assertions.check_simulation_logs(tar_path, file_test)
    assert result


def test_check_simulation_logs_missing_pattern(tmp_test_directory, safe_tar_open):
    tmp_path = Path(tmp_test_directory)
    tar_path = tmp_path / "test_logs.tar.gz"
    log_content = b"Log line with pattern_A\nAnother line\n"

    with safe_tar_open(tar_path, "w:gz") as tar:
        log_gz = tmp_path / "test.log.gz"
        with gzip.open(log_gz, "wb") as gz:
            gz.write(log_content)
        tar.add(log_gz, arcname="test.log.gz")

    file_test = {"expected_log_output": {"pattern": ["pattern_A", "missing_pattern"]}}
    result = assertions.check_simulation_logs(tar_path, file_test)
    assert not result


def test_check_simulation_logs_skip_non_log_files(tmp_test_directory, safe_tar_open):
    tmp_path = Path(tmp_test_directory)
    tar_path = tmp_path / "test_logs.tar.gz"

    with safe_tar_open(tar_path, "w:gz") as tar:
        not_log = tmp_path / "readme.txt"
        not_log.write_text("This is not a log file", encoding="utf-8")
        tar.add(not_log, arcname="readme.txt")

    file_test = {"expected_log_output": {"pattern": ["pattern"]}}
    result = assertions.check_simulation_logs(tar_path, file_test)
    assert not result


def test_read_log(tmp_test_directory, safe_tar_open):
    tmp_path = Path(tmp_test_directory)
    tar_path = tmp_path / "test.tar.gz"
    log_content = b"Test log content\nSecond line\n"

    with safe_tar_open(tar_path, "w:gz") as tar:
        log_gz = tmp_path / "test.log.gz"
        with gzip.open(log_gz, "wb") as gz:
            gz.write(log_content)
        tar.add(log_gz, arcname="test.log.gz")

    with safe_tar_open(tar_path, "r:gz") as tar:
        member = tar.getmembers()[0]
        result = assertions._read_log(member, tar)

    assert result == "Test log content\nSecond line\n"


def test_find_patterns():
    text = "This is a test line with pattern1 and pattern2"
    patterns = ["pattern1", "pattern2", "pattern3"]

    found = assertions._find_patterns(text, patterns)

    assert "pattern1" in found
    assert "pattern2" in found
    assert "pattern3" not in found
    assert len(found) == 2


def test_check_simulation_logs_forbidden_patterns_found(tmp_test_directory, safe_tar_open):
    tmp_path = Path(tmp_test_directory)
    tar_path = tmp_path / "test_logs.tar.gz"
    log_content = b"Log line with CURVED VERSION WITH SLIDING PLANAR ATMOSPHERE\nAnother line\n"

    with safe_tar_open(tar_path, "w:gz") as tar:
        log_gz = tmp_path / "test.log.gz"
        with gzip.open(log_gz, "wb") as gz:
            gz.write(log_content)
        tar.add(log_gz, arcname="test.log.gz")

    file_test = {
        "expected_log_output": {
            "forbidden_pattern": ["CURVED VERSION WITH SLIDING PLANAR ATMOSPHERE"]
        }
    }
    result = assertions.check_simulation_logs(tar_path, file_test)
    assert not result


def test_check_simulation_logs_forbidden_patterns_not_found(tmp_test_directory, safe_tar_open):
    tmp_path = Path(tmp_test_directory)
    tar_path = tmp_path / "test_logs.tar.gz"
    log_content = b"Log line with normal content\nAnother line\n"

    with safe_tar_open(tar_path, "w:gz") as tar:
        log_gz = tmp_path / "test.log.gz"
        with gzip.open(log_gz, "wb") as gz:
            gz.write(log_content)
        tar.add(log_gz, arcname="test.log.gz")

    file_test = {"expected_log_output": {"forbidden_pattern": ["CURVED VERSION", "FATAL ERROR"]}}
    result = assertions.check_simulation_logs(tar_path, file_test)
    assert result


def test_check_simulation_logs_wanted_and_forbidden_both_ok(tmp_test_directory, safe_tar_open):
    tmp_path = Path(tmp_test_directory)
    tar_path = tmp_path / "test_logs.tar.gz"
    log_content = b"Log line with expected_pattern\nAnother line with good content\n"

    with safe_tar_open(tar_path, "w:gz") as tar:
        log_gz = tmp_path / "test.log.gz"
        with gzip.open(log_gz, "wb") as gz:
            gz.write(log_content)
        tar.add(log_gz, arcname="test.log.gz")

    file_test = {
        "expected_log_output": {
            "pattern": ["expected_pattern"],
            "forbidden_pattern": ["CURVED VERSION", "ERROR"],
        }
    }
    result = assertions.check_simulation_logs(tar_path, file_test)
    assert result


def test_check_simulation_logs_wanted_ok_but_forbidden_found(tmp_test_directory, safe_tar_open):
    tmp_path = Path(tmp_test_directory)
    tar_path = tmp_path / "test_logs.tar.gz"
    log_content = b"Log line with expected_pattern\nAnother line with ERROR\n"

    with safe_tar_open(tar_path, "w:gz") as tar:
        log_gz = tmp_path / "test.log.gz"
        with gzip.open(log_gz, "wb") as gz:
            gz.write(log_content)
        tar.add(log_gz, arcname="test.log.gz")

    file_test = {
        "expected_log_output": {
            "pattern": ["expected_pattern"],
            "forbidden_pattern": ["ERROR", "FATAL"],
        }
    }
    result = assertions.check_simulation_logs(tar_path, file_test)
    assert not result


def test_check_simulation_logs_multiple_forbidden_patterns_found(tmp_test_directory, safe_tar_open):
    tmp_path = Path(tmp_test_directory)
    tar_path = tmp_path / "test_logs.tar.gz"
    log_content = b"Log with ERROR\nAnother line with FATAL\nThird line with WARNING\n"

    with safe_tar_open(tar_path, "w:gz") as tar:
        log_gz = tmp_path / "test.log.gz"
        with gzip.open(log_gz, "wb") as gz:
            gz.write(log_content)
        tar.add(log_gz, arcname="test.log.gz")

    file_test = {"expected_log_output": {"forbidden_pattern": ["ERROR", "FATAL", "CRITICAL"]}}
    result = assertions.check_simulation_logs(tar_path, file_test)
    assert not result


def test_check_simulation_logs_forbidden_only_empty_list(tmp_test_directory, safe_tar_open):
    tmp_path = Path(tmp_test_directory)
    tar_path = tmp_path / "test_logs.tar.gz"
    log_content = b"Log line with any content\n"

    with safe_tar_open(tar_path, "w:gz") as tar:
        log_gz = tmp_path / "test.log.gz"
        with gzip.open(log_gz, "wb") as gz:
            gz.write(log_content)
        tar.add(log_gz, arcname="test.log.gz")

    file_test = {"expected_log_output": {"forbidden_pattern": []}}
    result = assertions.check_simulation_logs(tar_path, file_test)
    assert result


def test_check_plain_log_patterns_found(tmp_path: Path):
    log_file = tmp_path / "run.log"
    log_file.write_text("start\nAll good\nOK done\n", encoding="utf-8")

    file_test = {"expected_log_output": {"pattern": ["OK"], "forbidden_pattern": []}}

    assert check_plain_log(log_file, file_test) is True


def test_check_plain_log_forbidden_pattern(tmp_path: Path):
    log_file = tmp_path / "run.log"
    log_file.write_text("ERROR: failure happened\n", encoding="utf-8")

    file_test = {"expected_log_output": {"pattern": [], "forbidden_pattern": ["ERROR"]}}

    assert check_plain_log(log_file, file_test) is False


def test_check_plain_log_missing_file_returns_false(tmp_path: Path):
    log_file = tmp_path / "missing.log"
    file_test = {"expected_log_output": {"pattern": ["hello"], "forbidden_pattern": []}}

    assert check_plain_log(log_file, file_test) is False


def test_check_plain_log_top_level_keys_fallback(tmp_path: Path):
    # When expected_log_output is not a dict, fallback to top-level keys
    log_file = tmp_path / "run.log"
    log_file.write_text("pipeline finished successfully\n", encoding="utf-8")

    file_test = {"expected_log_output": None, "pattern": ["finished"], "forbidden_pattern": []}

    assert check_plain_log(log_file, file_test) is True


def test_check_plain_log_no_patterns_returns_true(tmp_path: Path, caplog):
    # expected_log_output has no patterns; function should return True and log debug
    log_file = tmp_path / "run.log"
    log_file.write_text("some content\n", encoding="utf-8")

    file_test = {"expected_log_output": {}}

    with caplog.at_level(logging.DEBUG):
        assert check_plain_log(log_file, file_test) is True
        assert "No expected log output provided" in caplog.text


def test_check_plain_log_missing_expected_patterns(tmp_path: Path, caplog):
    # wanted pattern not present in log should log error and return False
    log_file = tmp_path / "run.log"
    log_file.write_text("only info lines here\n", encoding="utf-8")

    file_test = {"expected_log_output": {"pattern": ["NEEDED"], "forbidden_pattern": []}}

    with caplog.at_level(logging.ERROR):
        assert check_plain_log(log_file, file_test) is False
        assert "Missing expected patterns" in caplog.text


def test_check_plain_log_case_insensitive(tmp_path: Path):
    log_file = tmp_path / "run.log"
    log_file.write_text("Error: something went wrong\n", encoding="utf-8")

    # "error" (lowercase) should match "Error" (mixed case)
    file_test = {"expected_log_output": {"forbidden_pattern": ["error"]}}
    assert check_plain_log(log_file, file_test) is False

    log_file.write_text("Success: all good\n", encoding="utf-8")
    # "success" (lowercase) should match "Success" (mixed case)
    file_test = {"expected_log_output": {"pattern": ["success"]}}
    assert check_plain_log(log_file, file_test) is True
