"""Tests for local process execution."""

from pathlib import Path

import pytest

from simtools.job_execution.backends.base import BackendConfigurationError, BackendExecutionError
from simtools.job_execution.backends.local import LocalBackend, determine_max_workers
from simtools.job_execution.job import ExecutionOptions, JobSpec, SubmissionHandle


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


def test_determine_max_workers_handles_missing_cpu_count(monkeypatch):
    """CPU discovery always leaves at least one usable worker."""
    monkeypatch.setattr("simtools.job_execution.backends.local.os.cpu_count", lambda: None)

    assert determine_max_workers() == 1
    assert determine_max_workers(-1) == 1


@pytest.mark.parametrize(
    "config", [{"unknown": True}, {"mp_start_method": "invalid"}, {"mp_start_method": 1}]
)
def test_local_backend_rejects_invalid_configuration_for_direct_jobs(config):
    """Local backend configuration is validated even when no pool is created."""
    job = JobSpec("job-000000", 0, function=_square, item=1)

    with pytest.raises(BackendConfigurationError):
        LocalBackend().submit([job], ExecutionOptions(max_workers=1, backend_config=config))


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


def test_local_backend_uses_pool_and_collects_out_of_order_results(mocker, tmp_test_directory):
    """Pooled jobs are collected in input order and the executor is closed."""
    calls = {"shutdown": []}

    class _Future:
        def __init__(self, value):
            self.value = value

        def result(self):
            return self.value

        def cancel(self):
            calls["cancelled"] = calls.get("cancelled", 0) + 1

    class _Executor:
        def __init__(self, **kwargs):
            calls["kwargs"] = kwargs

        def submit(self, _function, job):
            return _Future(job.item * job.item)

        def shutdown(self, **kwargs):
            calls["shutdown"].append(kwargs)

    mocker.patch("simtools.job_execution.backends.local.get_context", return_value="context")
    mocker.patch("simtools.job_execution.backends.local.ProcessPoolExecutor", _Executor)
    mocker.patch(
        "simtools.job_execution.backends.local.as_completed",
        side_effect=lambda futures: reversed(list(futures)),
    )
    jobs = [
        JobSpec("job-000000", 0, function=_square, item=3),
        JobSpec("job-000001", 1, function=_square, item=2),
    ]

    submission = LocalBackend().submit(
        jobs,
        ExecutionOptions(
            max_workers=2,
            work_dir=tmp_test_directory,
            backend_config={"mp_start_method": "spawn"},
        ),
    )

    assert [result.value for result in LocalBackend().wait(submission)] == [9, 4]
    assert calls["kwargs"]["mp_context"] == "context"
    assert calls["shutdown"] == [{"wait": True}]


def test_local_backend_pool_reports_failures_and_cancels_pending_jobs(mocker):
    """Pool failures are reported and cancellation stops outstanding work."""

    class _Future:
        def result(self):
            raise RuntimeError("pool failure")

        def cancel(self):
            self.cancelled = True

    class _Executor:
        def __init__(self):
            self.shutdown_calls = []

        def shutdown(self, **kwargs):
            self.shutdown_calls.append(kwargs)

    future = _Future()
    executor = _Executor()
    submission = SubmissionHandle(
        backend="local",
        work_dir=Path("run"),
        job_ids=("job",),
        metadata={
            "executor": executor,
            "futures": {future: JobSpec("job", 0, function=_square, item=1)},
        },
    )
    mocker.patch("simtools.job_execution.backends.local.as_completed", return_value=[future])

    with pytest.raises(BackendExecutionError, match="pool failure"):
        LocalBackend().wait(submission)

    LocalBackend().cancel(submission)
    assert future.cancelled
    assert executor.shutdown_calls == [{"wait": True}, {"wait": False, "cancel_futures": True}]
