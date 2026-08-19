"""Protocol and exceptions shared by execution backends."""

from typing import Protocol

from simtools.job_execution.job import ExecutionOptions, JobResult, JobSpec, SubmissionHandle


class BackendError(RuntimeError):
    """Base class for backend failures."""


class BackendConfigurationError(BackendError):
    """Raised when backend configuration is invalid or unavailable."""


class BackendSubmissionError(BackendError):
    """Raised when a backend cannot submit a job set."""


class BackendExecutionError(BackendError):
    """Raised when one or more submitted jobs fail."""


class ExecutionBackend(Protocol):
    """Protocol implemented by all job-execution backends."""

    def submit(self, job_specs: list[JobSpec], options: ExecutionOptions) -> SubmissionHandle:
        """Submit jobs and return a handle."""

    def wait(self, submission: SubmissionHandle) -> list[JobResult]:
        """Wait for completion and return ordered results."""

    def cancel(self, submission: SubmissionHandle) -> None:
        """Cancel an active submission."""
