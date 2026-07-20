"""Unit tests for simtel_log_reader."""

import gzip
from unittest.mock import patch

import pytest

from simtools.simtel import simtel_log_reader

LOG_TEXT = """
Intermediate block
Tel. triggered: 12
Run(s) completed as expected after 100 events
Final block
Tel. triggered: 34
Run(s) completed as expected after 200 events
"""


def test_read_log_file_reads_plain_text(tmp_path):
    log_file = tmp_path / "run000001.simtel.log"
    log_file.write_text(LOG_TEXT, encoding="utf-8")

    assert simtel_log_reader.read_log_file(log_file) == LOG_TEXT


def test_read_log_file_reads_gzip(tmp_path):
    log_file = tmp_path / "run000001.simtel.log.gz"
    with gzip.open(log_file, mode="wt", encoding="utf-8") as file_handle:
        file_handle.write(LOG_TEXT)

    assert simtel_log_reader.read_log_file(log_file) == LOG_TEXT


def test_read_log_file_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Log file not found"):
        simtel_log_reader.read_log_file(tmp_path / "missing.log")


def test_extract_trigger_count_uses_last_match():
    assert simtel_log_reader.extract_trigger_count(LOG_TEXT) == 34


def test_extract_trigger_count_returns_none_when_missing():
    assert simtel_log_reader.extract_trigger_count("no trigger summary") is None


def test_extract_event_count_uses_last_match():
    assert simtel_log_reader.extract_event_count(LOG_TEXT) == 200


def test_extract_event_count_returns_none_when_missing():
    assert simtel_log_reader.extract_event_count("no event summary") is None


def test_extract_run_number_from_file_name():
    assert simtel_log_reader.extract_run_number("gamma_run000015.simtel.log.gz") == 15
    assert simtel_log_reader.extract_run_number("gamma_run15.simtel.log.gz") == 15


def test_extract_run_number_from_parent_directory(tmp_path):
    file_path = tmp_path / "run000042" / "logs" / "gamma.simtel.log.gz"

    assert simtel_log_reader.extract_run_number(file_path) == 42


def test_extract_run_number_returns_none_when_missing(tmp_path):
    assert simtel_log_reader.extract_run_number(tmp_path / "gamma.simtel.log.gz") is None


def test_extract_threshold_from_directory_part(tmp_path):
    file_path = tmp_path / "300" / "sim_telarray" / "gamma_run000001.simtel.log.gz"

    assert simtel_log_reader.extract_threshold(file_path) == 300


def test_extract_threshold_ignores_values_outside_expected_range(tmp_path):
    file_path = tmp_path / "5" / "2000" / "gamma_run000001.simtel.log.gz"

    assert simtel_log_reader.extract_threshold(file_path) is None


def test_parse_log_file_returns_parsed_data(tmp_path):
    log_dir = tmp_path / "300"
    log_dir.mkdir()
    log_file = log_dir / "gamma_run000001.simtel.log"
    log_file.write_text(LOG_TEXT, encoding="utf-8")

    result = simtel_log_reader.parse_log_file(log_file)

    assert result == {
        "run": 1,
        "threshold": 300,
        "triggers": 34,
        "events": 200,
        "file_path": str(log_file),
    }


def test_parse_log_file_returns_none_when_file_cannot_be_read(tmp_path):
    with patch(
        "simtools.simtel.simtel_log_reader.read_log_file",
        side_effect=OSError("cannot read"),
    ):
        assert simtel_log_reader.parse_log_file(tmp_path / "300" / "run000001.log") is None


def test_parse_log_file_returns_none_when_critical_info_is_missing(tmp_path):
    log_file = tmp_path / "gamma.simtel.log"
    log_file.write_text(LOG_TEXT, encoding="utf-8")

    assert simtel_log_reader.parse_log_file(log_file) is None


def test_find_log_files_finds_matching_logs(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    log_file = log_dir / "gamma_run000001.simtel.log.gz"
    log_file.touch()

    assert simtel_log_reader.find_log_files(tmp_path) == [log_file]


def test_find_log_files_raises_for_missing_root(tmp_path):
    with pytest.raises(FileNotFoundError, match="Root directory not found"):
        simtel_log_reader.find_log_files(tmp_path / "missing")


def test_find_log_files_raises_when_no_files_match(tmp_path):
    with pytest.raises(FileNotFoundError, match="No log files found"):
        simtel_log_reader.find_log_files(tmp_path)


def test_parse_log_files_filters_failed_parses(tmp_path):
    files = [tmp_path / "good.log", tmp_path / "bad.log"]

    with patch(
        "simtools.simtel.simtel_log_reader.parse_log_file",
        side_effect=[
            {"run": 1, "threshold": 300, "triggers": 10, "events": 100},
            None,
        ],
    ):
        assert simtel_log_reader.parse_log_files(files) == [
            {"run": 1, "threshold": 300, "triggers": 10, "events": 100}
        ]


def test_parse_log_files_raises_when_all_parses_fail(tmp_path):
    with patch("simtools.simtel.simtel_log_reader.parse_log_file", return_value=None):
        with pytest.raises(ValueError, match="No log files could be parsed successfully"):
            simtel_log_reader.parse_log_files([tmp_path / "bad.log"])
