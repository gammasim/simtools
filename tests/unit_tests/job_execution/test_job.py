"""Tests for backend-neutral execution data structures."""

from pathlib import Path

import pytest

from simtools.job_execution.job import ExecutionOptions, JobResult, JobSpec, SubmissionHandle


def _identity(value):
    """Return the input value."""
    return value


def test_job_spec_requires_exactly_one_execution_form():
    """A job cannot contain both a function and a command."""
    with pytest.raises(ValueError, match="exactly one"):
        JobSpec("job-000000", 0)
    with pytest.raises(ValueError, match="exactly one"):
        JobSpec("job-000000", 0, function=_identity, command=("command",))


def test_job_spec_rejects_empty_commands():
    """A command job must contain an executable or command argument."""
    with pytest.raises(ValueError, match="at least one"):
        JobSpec("job-000000", 0, command=())


def test_job_result_and_execution_options_have_stable_defaults():
    """Result and option defaults are suitable for backend construction."""
    result = JobResult("job-000000", 0)
    options = ExecutionOptions()

    assert result.status == "completed"
    assert result.return_code == 0
    assert options.backend == "local"
    assert options.backend_config == {}


def test_submission_handle_round_trip_preserves_paths_and_process_ids():
    """Submission handles can be serialized and restored without loss."""
    handle = SubmissionHandle(
        backend="htcondor",
        work_dir=Path("run"),
        job_ids=("job-000000",),
        scheduler_id=17,
        process_ids={"job-000000": 3},
        metadata={"state": "submitted"},
    )

    assert SubmissionHandle.from_dict(handle.as_dict()) == handle
