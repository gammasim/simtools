import logging

import pytest

from simtools.testing.log_inspector import inspect

ERROR_MSG_LINE_2 = "Error or warning found in log at line 2"


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
        "RuntimeWarning: Something went wrong."
    ]
    result = inspect(log_text)
    assert result is False
    assert len(mock_logger.records) == 2
    assert ERROR_MSG_LINE_2 in mock_logger.text
    assert "Error or warning found in log at line 3" in mock_logger.text


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
        "INFO:: All systems operational.\nException: A critical failure occurred.\nINFO:: This is fine.\nFailed to connect to the database."
    ]
    result = inspect(log_text)
    assert result is False
    assert len(mock_logger.records) == 2
    assert ERROR_MSG_LINE_2 in mock_logger.text
    assert "Error or warning found in log at line 4" in mock_logger.text


def test_inspect_single_string_input(mock_logger):
    log_text = "INFO:: All good.\nERROR:: Something broke.\nINFO:: Still good."
    result = inspect(log_text)
    assert result is False
    assert len(mock_logger.records) == 1
    assert ERROR_MSG_LINE_2 in mock_logger.text
