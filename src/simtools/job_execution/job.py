"""Backend-neutral job and execution data structures."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class JobSpec:
    """Describe one independent unit of work."""

    job_id: str
    index: int
    function: Any = None
    item: Any = None
    command: tuple[str, ...] | None = None
    initializer: Any = None
    initargs: tuple[Any, ...] = ()
    runtime_args: dict[str, Any] | None = None
    runtime_db_config: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    resources: dict[str, Any] = field(default_factory=dict)
    mount_paths: tuple[Path, ...] = ()
    output_paths: tuple[Path, ...] = ()

    def __post_init__(self):
        """Validate the mutually exclusive job execution forms."""
        has_function = self.function is not None
        has_command = self.command is not None
        if has_function == has_command:
            raise ValueError("A JobSpec must define exactly one of function or command.")
        if has_command and not self.command:
            raise ValueError("A command JobSpec must contain at least one argument.")


@dataclass
class ExecutionOptions:
    """Options shared by execution backends."""

    backend: str = "local"
    max_workers: int | None = None
    work_dir: Path | None = None
    backend_config: dict[str, Any] = field(default_factory=dict)
    initializer: Any = None
    initargs: tuple[Any, ...] = ()
    request_cpus: int = 1
    request_memory: str | None = None
    request_disk: str | None = None
    priority: int | None = None
    container_image: str | None = None
    environment_file: Path | None = None
    poll_interval: float = 60
    timeout: float | None = None
    cancel_on_interrupt: bool = False
    keep_successful_artifacts: bool = False
    extra_submit_attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class JobResult:
    """Result and diagnostic information for one job."""

    job_id: str
    index: int
    value: Any = None
    status: str = "completed"
    return_code: int = 0
    stdout: Path | None = None
    stderr: Path | None = None
    error: str | None = None


@dataclass
class SubmissionHandle:
    """Persistable description of a submitted execution."""

    backend: str
    work_dir: Path
    job_ids: tuple[str, ...]
    scheduler_id: int | None = None
    process_ids: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self):
        """Return a JSON-serializable manifest representation."""
        return {
            "backend": self.backend,
            "work_dir": str(self.work_dir),
            "job_ids": list(self.job_ids),
            "scheduler_id": self.scheduler_id,
            "process_ids": self.process_ids,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload):
        """Restore a submission handle from a manifest mapping."""
        return cls(
            backend=payload["backend"],
            work_dir=Path(payload["work_dir"]),
            job_ids=tuple(payload.get("job_ids", ())),
            scheduler_id=payload.get("scheduler_id"),
            process_ids={key: int(value) for key, value in payload.get("process_ids", {}).items()},
            metadata=dict(payload.get("metadata", {})),
        )
