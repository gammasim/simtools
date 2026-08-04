"""Generic execution backends for independent simtools jobs."""

from simtools.job_execution.execution import (
    execute_jobs,
    load_submission,
    map_ordered,
    options_from_args,
    submit_jobs,
    wait_for_submission,
)
from simtools.job_execution.job import ExecutionOptions, JobResult, JobSpec, SubmissionHandle
from simtools.job_execution.backends.registry import available_backends, register_backend

__all__ = [
    "ExecutionOptions",
    "JobResult",
    "JobSpec",
    "SubmissionHandle",
    "execute_jobs",
    "load_submission",
    "map_ordered",
    "options_from_args",
    "submit_jobs",
    "wait_for_submission",
    "available_backends",
    "register_backend",
]
