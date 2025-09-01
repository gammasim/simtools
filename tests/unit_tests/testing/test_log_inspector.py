import logging

import pytest

from simtools.testing.log_inspector import inspect

ERROR_MSG_LINE_1 = "Error or warning found in log at line 1"


@pytest.fixture
def mock_logger(caplog):
    caplog.set_level(logging.ERROR)
    return caplog


def test_inspect_no_issues(mock_logger):
    log_text = ["INFO:: This is an informational message.", "DEBUG:: Debugging details here."]
    result = inspect(log_text)
    assert result is True
    assert not mock_logger.records


def test_inspect_with_errors(mock_logger):
    log_text = [
        "INFO:: This is an informational message.",
        "ERROR:: An error occurred in the system.",
        "RuntimeWarning: Something went wrong.",
    ]
    result = inspect(log_text)
    assert result is False
    assert len(mock_logger.records) == 2
    assert ERROR_MSG_LINE_1 in mock_logger.text
    assert "Error or warning found in log at line 1" in mock_logger.text


def test_inspect_ignore_info_lines(mock_logger):
    log_text = [
        "INFO:: Job error stream contains no issues.",
        "INFO:: Another informational message.",
    ]
    result = inspect(log_text)
    assert result is True
    assert not mock_logger.records


def test_inspect_mixed_input(mock_logger):
    log_text = [
        "INFO:: All systems operational.\nException: A critical failure occurred.\n"
        "INFO:: This is fine.\nFailed to connect to the database."
    ]
    result = inspect(log_text)
    assert result is False
    assert len(mock_logger.records) == 2
    assert "Error or warning found in log at line 2" in mock_logger.text
    assert "Error or warning found in log at line 4" in mock_logger.text


def test_inspect_single_string_input(mock_logger):
    log_text = "INFO:: All good.\nERROR:: Something broke.\nINFO:: Still good."
    result = inspect(log_text)
    assert result is False
    assert len(mock_logger.records) == 1
    assert "Error or warning found in log at line 2" in mock_logger.text


def test_inspect_ignore_patterns(mock_logger):
    log_text = (
        "WARNING::simtel_io_metadata(l80)::_decode_dictionary::Unable to decode metadata "
        "with encoding utf8: 'utf-8' codec can't decode byte 0x80 in position 128: invalid "
        "start byte. Falling back to 'utf-8' with errors='ignore'."
    )
    result = inspect(log_text)
    assert result is True
