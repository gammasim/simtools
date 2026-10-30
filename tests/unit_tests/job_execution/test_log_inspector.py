import gzip
import logging
from pathlib import Path

import pytest

from simtools.job_execution import log_inspector

PATTERN_KEY = "pattern"
FORBIDDEN_PATTERN_KEY = "forbidden_pattern"
UTF8_ENCODING = "utf-8"
STDERR_DIAGNOSTIC = "stderr diagnostic"

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

    log_text = (
        "DEBUG: Setting environment variables for job execution: "
        "{'GITHUB_HEAD_REF': 'corsika-limits-error'}"
    )
    result = log_inspector.inspect(log_text)
    assert result is True

    log_text = (
        "DEBUG: Loading schema from "
        "/workdir/external/simtools/src/simtools/schemas/model_parameters/"
        "transit_time_error.schema.yml for schema version latest"
    )
    result = log_inspector.inspect(log_text)
    assert result is True


def test_check_plain_logs_skip_non_log_files(tmp_test_directory, safe_tar_open):
    tar_path = Path(str(tmp_test_directory)) / "test_logs.tar.gz"

    with safe_tar_open(tar_path, "w:gz") as tar:
        not_log = Path(str(tmp_test_directory)) / "readme.txt"
        not_log.write_text("This is not a log file", encoding=UTF8_ENCODING)
        tar.add(not_log, arcname="readme.txt")

    file_test = {"expected_log_output": {PATTERN_KEY: [PATTERN_KEY]}}
    assert log_inspector.check_plain_logs(tar_path, file_test) is False


def test_read_log(tmp_test_directory, safe_tar_open):
    tar_path = Path(str(tmp_test_directory)) / "test.tar.gz"
    log_content = b"Test log content\nSecond line\n"

    with safe_tar_open(tar_path, "w:gz") as tar:
        log_gz = Path(str(tmp_test_directory)) / "test.log.gz"
        with gzip.open(log_gz, "wb") as gz:
            gz.write(log_content)
        tar.add(log_gz, arcname="test.log.gz")

    with safe_tar_open(tar_path, "r:gz") as tar:
        member = tar.getmembers()[0]
        result = log_inspector._read_log(member, tar)

    assert result == "Test log content\nSecond line\n"


@pytest.mark.parametrize(
    ("log_content", "expected_log_output", "should_pass"),
    [
        (
            b"Log line with CURVED VERSION WITH SLIDING PLANAR ATMOSPHERE\nAnother line\n",
            {FORBIDDEN_PATTERN_KEY: ["CURVED VERSION WITH SLIDING PLANAR ATMOSPHERE"]},
            False,
        ),
        (
            b"Log line with normal content\nAnother line\n",
            {FORBIDDEN_PATTERN_KEY: ["CURVED VERSION", "FATAL ERROR"]},
            True,
        ),
        (
            b"Log line with expected_pattern\nAnother line with good content\n",
            {PATTERN_KEY: ["expected_pattern"], FORBIDDEN_PATTERN_KEY: ["CURVED VERSION", "ERROR"]},
            True,
        ),
        (
            b"Log line with expected_pattern\nAnother line with ERROR\n",
            {PATTERN_KEY: ["expected_pattern"], FORBIDDEN_PATTERN_KEY: ["ERROR", "FATAL"]},
            False,
        ),
        (
            b"Log with ERROR\nAnother line with FATAL\nThird line with WARNING\n",
            {FORBIDDEN_PATTERN_KEY: ["ERROR", "FATAL", "CRITICAL"]},
            False,
        ),
        (b"Log line with any content\n", {FORBIDDEN_PATTERN_KEY: []}, True),
        (
            b"Log line with ERROR\nAnother line\n",
            {PATTERN_KEY: [], FORBIDDEN_PATTERN_KEY: ["ERROR"]},
            False,
        ),
        (b"Any content\n", {}, True),
        (
            b"Log line with pattern_A\nAnother line\nLine with pattern_B\n",
            {PATTERN_KEY: ["pattern_A", "pattern_B"]},
            True,
        ),
        (
            b"Log line with pattern_A\nAnother line\n",
            {PATTERN_KEY: ["pattern_A", "missing_pattern"]},
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
            {"expected_log_output": {PATTERN_KEY: ["OK"], FORBIDDEN_PATTERN_KEY: []}},
            True,
        ),
        (
            "ERROR: failure happened\n",
            {"expected_log_output": {PATTERN_KEY: [], FORBIDDEN_PATTERN_KEY: ["ERROR"]}},
            False,
        ),
        (
            "Error: something went wrong\n",
            {"expected_log_output": {FORBIDDEN_PATTERN_KEY: ["error"]}},
            False,
        ),
        ("Success: all good\n", {"expected_log_output": {PATTERN_KEY: ["success"]}}, True),
    ],
)
def test_check_plain_logs(tmp_test_directory, content, file_test, should_pass):
    log_file = Path(str(tmp_test_directory)) / "run.log"
    log_file.write_text(content, encoding=UTF8_ENCODING)
    assert log_inspector.check_plain_logs(log_file, file_test) is should_pass


def test_check_plain_logs_missing_file_returns_false(tmp_test_directory):
    log_file = Path(str(tmp_test_directory)) / "missing.log"
    file_test = {"expected_log_output": {PATTERN_KEY: ["hello"], FORBIDDEN_PATTERN_KEY: []}}
    assert log_inspector.check_plain_logs(log_file, file_test) is False


def test_check_plain_logs_top_level_keys_fallback(tmp_test_directory):
    log_file = Path(str(tmp_test_directory)) / "run.log"
    log_file.write_text("pipeline finished successfully\n", encoding=UTF8_ENCODING)
    file_test = {"expected_log_output": None, PATTERN_KEY: ["finished"], FORBIDDEN_PATTERN_KEY: []}
    assert log_inspector.check_plain_logs(log_file, file_test) is True


def test_check_tar_logs_invalid_tar_raises(tmp_test_directory):
    not_tar = Path(str(tmp_test_directory)) / "not_a_tar.txt"
    not_tar.write_text("This is not a tar file", encoding=UTF8_ENCODING)
    file_test = {"expected_log_output": {PATTERN_KEY: ["test"]}}
    with pytest.raises(ValueError, match="is not a tar file"):
        log_inspector.check_tar_logs(not_tar, file_test)


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("forbidden", [False, True])
def test_check_plain_logs_includes_stderr_sibling(tmp_test_directory, compressed, forbidden):
    suffix = ".gz" if compressed else ""
    stdout = Path(tmp_test_directory) / f"simulation.log{suffix}"
    stderr = Path(tmp_test_directory) / f"simulation.err{suffix}"
    for path, content in ((stdout, "simulation started"), (stderr, STDERR_DIAGNOSTIC)):
        if compressed:
            with gzip.open(path, "wt", encoding=UTF8_ENCODING) as stream:
                stream.write(content)
        else:
            path.write_text(content, encoding=UTF8_ENCODING)
    pattern_key = FORBIDDEN_PATTERN_KEY if forbidden else PATTERN_KEY
    assert log_inspector.check_plain_logs(stdout, {pattern_key: [STDERR_DIAGNOSTIC]}) is (
        not forbidden
    )


def test_check_tar_logs_includes_stderr(tmp_test_directory, safe_tar_open):
    stderr = Path(tmp_test_directory) / "simulation.err.gz"
    with gzip.open(stderr, "wt", encoding=UTF8_ENCODING) as stream:
        stream.write(STDERR_DIAGNOSTIC)
    archive = Path(tmp_test_directory) / "split_logs.tar.gz"
    with safe_tar_open(archive, "w:gz") as tar:
        tar.add(stderr, arcname=stderr.name)
    assert log_inspector.check_tar_logs(archive, {PATTERN_KEY: [STDERR_DIAGNOSTIC]})
    assert not log_inspector.check_tar_logs(archive, {FORBIDDEN_PATTERN_KEY: [STDERR_DIAGNOSTIC]})
