import gzip
import logging
from pathlib import Path

import pytest

from simtools.testing import log_inspector

ERROR_MSG_LINE_1 = "Error or warning found in log at line 1"


@pytest.fixture
def mock_logger(caplog):
    caplog.set_level(logging.ERROR)
    return caplog


@pytest.fixture
def tar_with_log(tmp_test_directory, safe_tar_open):
    def _create_tar(log_content):
        tmp_path = Path(tmp_test_directory)
        tar_path = tmp_path / "test_logs.tar.gz"
        with safe_tar_open(tar_path, "w:gz") as tar:
            log_gz = tmp_path / "test.log.gz"
            with gzip.open(log_gz, "wb") as gz:
                gz.write(log_content)
            tar.add(log_gz, arcname="test.log.gz")
        return tar_path

    return _create_tar


def test_inspect_no_issues(mock_logger):
    log_text = ["INFO:: This is an informational message.", "DEBUG:: Debugging details here."]
    result = log_inspector.inspect(log_text)
    assert result is True
    assert not mock_logger.records


def test_inspect_with_errors(mock_logger):
    log_text = [
        "INFO:: This is an informational message.",
        "ERROR:: An error occurred in the system.",
        "RuntimeWarning: Something went wrong.",
    ]
    result = log_inspector.inspect(log_text)
    assert result is False
    assert len(mock_logger.records) == 2
    assert ERROR_MSG_LINE_1 in mock_logger.text
    assert "Error or warning found in log at line 1" in mock_logger.text


def test_inspect_ignore_info_lines(mock_logger):
    log_text = [
        "INFO:: Job error stream contains no issues.",
        "INFO:: Another informational message.",
    ]
    result = log_inspector.inspect(log_text)
    assert result is True
    assert not mock_logger.records


def test_inspect_mixed_input(mock_logger):
    log_text = [
        "INFO:: All systems operational.\nException: A critical failure occurred.\n"
        "INFO:: This is fine.\nFailed to connect to the database."
    ]
    result = log_inspector.inspect(log_text)
    assert result is False
    assert len(mock_logger.records) == 2
    assert "Error or warning found in log at line 2" in mock_logger.text
    assert "Error or warning found in log at line 4" in mock_logger.text


def test_inspect_single_string_input(mock_logger):
    log_text = "INFO:: All good.\nERROR:: Something broke.\nINFO:: Still good."
    result = log_inspector.inspect(log_text)
    assert result is False
    assert len(mock_logger.records) == 1
    assert "Error or warning found in log at line 2" in mock_logger.text


def test_inspect_ignore_patterns(mock_logger):
    log_text = (
        "WARNING::simtel_io_metadata(l80)::_decode_dictionary::Unable to decode metadata "
        "with encoding utf8: 'utf-8' codec can't decode byte 0x80 in position 128: invalid "
        "start byte. Falling back to 'utf-8' with errors='ignore'."
    )
    result = log_inspector.inspect(log_text)
    assert result is True

    log_text = (
        "Hello!"
        "WARNING::metadata_collector(l273)::_fill_contact_meta::Failed to get user name: 'getpwuid(): uid not found: 1000', setting it to UNKNOWN_USER"
    )
    result = log_inspector.inspect(log_text)
    assert result is True

    log_text = "DEBUG::__init__(l748)::adjust_text::Error: 52.640601388888854"
    result = log_inspector.inspect(log_text)
    assert result is True


def test_check_plain_logs_skip_non_log_files(tmp_test_directory, safe_tar_open):
    tmp_path = Path(tmp_test_directory)
    tar_path = tmp_path / "test_logs.tar.gz"

    with safe_tar_open(tar_path, "w:gz") as tar:
        not_log = tmp_path / "readme.txt"
        not_log.write_text("This is not a log file", encoding="utf-8")
        tar.add(not_log, arcname="readme.txt")

    file_test = {"expected_log_output": {"pattern": ["pattern"]}}
    assert log_inspector.check_plain_logs(tar_path, file_test) is False


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
        result = log_inspector._read_log(member, tar)

    assert result == "Test log content\nSecond line\n"


def test_find_patterns():
    text = "This is a test line with pattern1 and pattern2"
    patterns = ["pattern1", "pattern2", "pattern3"]

    found = log_inspector._find_patterns(text, patterns)

    assert "pattern1" in found
    assert "pattern2" in found
    assert "pattern3" not in found
    assert len(found) == 2


@pytest.mark.parametrize(
    ("log_content", "expected_log_output", "should_pass"),
    [
        (
            b"Log line with CURVED VERSION WITH SLIDING PLANAR ATMOSPHERE\nAnother line\n",
            {"forbidden_pattern": ["CURVED VERSION WITH SLIDING PLANAR ATMOSPHERE"]},
            False,
        ),
        (
            b"Log line with normal content\nAnother line\n",
            {"forbidden_pattern": ["CURVED VERSION", "FATAL ERROR"]},
            True,
        ),
        (
            b"Log line with expected_pattern\nAnother line with good content\n",
            {"pattern": ["expected_pattern"], "forbidden_pattern": ["CURVED VERSION", "ERROR"]},
            True,
        ),
        (
            b"Log line with expected_pattern\nAnother line with ERROR\n",
            {"pattern": ["expected_pattern"], "forbidden_pattern": ["ERROR", "FATAL"]},
            False,
        ),
        (
            b"Log with ERROR\nAnother line with FATAL\nThird line with WARNING\n",
            {"forbidden_pattern": ["ERROR", "FATAL", "CRITICAL"]},
            False,
        ),
        (b"Log line with any content\n", {"forbidden_pattern": []}, True),
        (
            b"Log line with ERROR\nAnother line\n",
            {"pattern": [], "forbidden_pattern": ["ERROR"]},
            False,
        ),
        (b"Any content\n", {}, True),
        (
            b"Log line with pattern_A\nAnother line\nLine with pattern_B\n",
            {"pattern": ["pattern_A", "pattern_B"]},
            True,
        ),
        (
            b"Log line with pattern_A\nAnother line\n",
            {"pattern": ["pattern_A", "missing_pattern"]},
            False,
        ),
    ],
)
def test_check_tar_logs(tar_with_log, log_content, expected_log_output, should_pass):
    tar_path = tar_with_log(log_content)
    file_test = {"expected_log_output": expected_log_output}
    assert log_inspector.check_tar_logs(tar_path, file_test) is should_pass


@pytest.mark.parametrize(
    ("content", "file_test", "should_pass"),
    [
        (
            "start\nAll good\nOK done\n",
            {"expected_log_output": {"pattern": ["OK"], "forbidden_pattern": []}},
            True,
        ),
        (
            "ERROR: failure happened\n",
            {"expected_log_output": {"pattern": [], "forbidden_pattern": ["ERROR"]}},
            False,
        ),
        (
            "Error: something went wrong\n",
            {"expected_log_output": {"forbidden_pattern": ["error"]}},
            False,
        ),
        ("Success: all good\n", {"expected_log_output": {"pattern": ["success"]}}, True),
    ],
)
def test_check_plain_logs(tmp_path: Path, content, file_test, should_pass):
    log_file = tmp_path / "run.log"
    log_file.write_text(content, encoding="utf-8")
    assert log_inspector.check_plain_logs(log_file, file_test) is should_pass


def test_check_plain_logs_missing_file_returns_false(tmp_path: Path):
    log_file = tmp_path / "missing.log"
    file_test = {"expected_log_output": {"pattern": ["hello"], "forbidden_pattern": []}}
    assert log_inspector.check_plain_logs(log_file, file_test) is False


def test_check_plain_logs_top_level_keys_fallback(tmp_path: Path):
    log_file = tmp_path / "run.log"
    log_file.write_text("pipeline finished successfully\n", encoding="utf-8")
    file_test = {"expected_log_output": None, "pattern": ["finished"], "forbidden_pattern": []}
    assert log_inspector.check_plain_logs(log_file, file_test) is True


def test_check_plain_logs_no_patterns_returns_true(tmp_path: Path, caplog):
    log_file = tmp_path / "run.log"
    log_file.write_text("some content\n", encoding="utf-8")
    file_test = {"expected_log_output": {}}
    with caplog.at_level(logging.DEBUG):
        assert log_inspector.check_plain_logs(log_file, file_test) is True
        assert "No expected log output provided" in caplog.text


def test_check_plain_logs_missing_expected_patterns(tmp_path: Path, caplog):
    log_file = tmp_path / "run.log"
    log_file.write_text("only info lines here\n", encoding="utf-8")
    file_test = {"expected_log_output": {"pattern": ["NEEDED"], "forbidden_pattern": []}}
    with caplog.at_level(logging.ERROR):
        assert log_inspector.check_plain_logs(log_file, file_test) is False
        assert "Missing expected patterns" in caplog.text


def test_check_tar_logs_invalid_tar_raises(tmp_path: Path):
    not_tar = tmp_path / "not_a_tar.txt"
    not_tar.write_text("This is not a tar file", encoding="utf-8")
    file_test = {"expected_log_output": {"pattern": ["test"]}}
    with pytest.raises(ValueError, match="is not a tar file"):
        log_inspector.check_tar_logs(not_tar, file_test)


def test_get_expected_patterns_with_expected_log_output():
    file_test = {
        "expected_log_output": {
            "pattern": ["success"],
            "forbidden_pattern": ["error"],
        }
    }
    wanted, forbidden = log_inspector._get_expected_patterns(file_test)
    assert wanted == ["success"]
    assert forbidden == ["error"]


def test_get_expected_patterns_top_level():
    file_test = {
        "pattern": ["done"],
        "forbidden_pattern": ["failed"],
    }
    wanted, forbidden = log_inspector._get_expected_patterns(file_test)
    assert wanted == ["done"]
    assert forbidden == ["failed"]


def test_get_expected_patterns_no_patterns():
    file_test = {"expected_log_output": {}}
    wanted, forbidden = log_inspector._get_expected_patterns(file_test)
    assert wanted is None
    assert forbidden is None


def test_validate_patterns_forbidden_found(caplog):
    with caplog.at_level(logging.ERROR):
        result = log_inspector._validate_patterns({"ERROR"}, {"ERROR"}, ["success"])
        assert result is False
        assert "Forbidden patterns found" in caplog.text


def test_validate_patterns_missing_expected(caplog):
    with caplog.at_level(logging.ERROR):
        result = log_inspector._validate_patterns(set(), set(), ["success", "done"])
        assert result is False
        assert "Missing expected patterns" in caplog.text


def test_validate_patterns_all_found(caplog):
    with caplog.at_level(logging.DEBUG):
        result = log_inspector._validate_patterns({"success"}, set(), ["success"])
        assert result is True
        assert "All expected patterns found" in caplog.text


def test_check_plain_logs_multiple_files(tmp_path: Path):
    log1 = tmp_path / "run1.log"
    log2 = tmp_path / "run2.log"
    log1.write_text("First log with success\n", encoding="utf-8")
    log2.write_text("Second log with done\n", encoding="utf-8")
    file_test = {"expected_log_output": {"pattern": ["success", "done"], "forbidden_pattern": []}}
    assert log_inspector.check_plain_logs([log1, log2], file_test) is True


def test_check_plain_logs_gzipped_file(tmp_path: Path):
    log_gz = tmp_path / "run.log.gz"
    with gzip.open(log_gz, "wt", encoding="utf-8") as gz:
        gz.write("Compressed log with pattern\n")
    file_test = {"expected_log_output": {"pattern": ["pattern"], "forbidden_pattern": []}}
    assert log_inspector.check_plain_logs(log_gz, file_test) is True


def test_check_tar_logs_no_patterns_without_file():
    file_test = {}
    result = log_inspector.check_tar_logs(Path("dummy.tar.gz"), file_test)
    assert result
