"""Tests for shared execution backend contracts and exceptions."""

from simtools.job_execution.backends.base import (
    BackendConfigurationError,
    BackendError,
    BackendExecutionError,
    BackendSubmissionError,
)


def test_backend_errors_share_one_exception_hierarchy():
    """All backend failures can be handled through the common base class."""
    assert issubclass(BackendConfigurationError, BackendError)
    assert issubclass(BackendSubmissionError, BackendError)
    assert issubclass(BackendExecutionError, BackendError)
