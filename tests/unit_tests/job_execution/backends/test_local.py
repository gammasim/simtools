"""Tests for local process execution."""

import pytest

from simtools.job_execution.backends.base import BackendExecutionError
from simtools.job_execution.backends.local import LocalBackend, determine_max_workers
from simtools.job_execution.job import ExecutionOptions, JobSpec


def _square(value):
    """Return the square of one value."""
    return value * value


def _fail(_value):
    """Raise an error for failure-path testing."""
    raise RuntimeError("expected failure")


def test_determine_max_workers_handles_defaults_and_explicit_values(monkeypatch):
    """Worker-count selection handles CPU discovery and explicit limits."""
    monkeypatch.setattr("simtools.job_execution.backends.local.os.cpu_count", lambda: 10)

    assert determine_max_workers() == 6
    assert determine_max_workers(3) == 3
    assert determine_max_workers(0) == 10


def test_local_backend_executes_direct_jobs_in_order(tmp_test_directory):
    """Small local submissions return ordered results without a process pool."""
    jobs = [
        JobSpec("job-000000", 0, function=_square, item=3),
        JobSpec("job-000001", 1, function=_square, item=2),
    ]
    options = ExecutionOptions(max_workers=1, work_dir=tmp_test_directory)
    backend = LocalBackend()

    submission = backend.submit(jobs, options)

    assert [result.value for result in backend.wait(submission)] == [9, 4]


def test_local_backend_reports_direct_job_failures():
    """Exceptions from direct jobs become backend execution errors."""
    job = JobSpec("job-000000", 0, function=_fail, item=1)

    with pytest.raises(BackendExecutionError, match="expected failure"):
        LocalBackend().submit([job], ExecutionOptions(max_workers=1))
