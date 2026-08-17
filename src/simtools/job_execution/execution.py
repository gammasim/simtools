"""Public execution facade used by long-running applications."""

import json
import logging
from pathlib import Path

import yaml

from simtools.job_execution.backends.registry import get_backend
from simtools.job_execution.job import ExecutionOptions, JobSpec, SubmissionHandle
from simtools.settings import config

logger = logging.getLogger(__name__)


def execute_jobs(job_specs, options=None):
    """Execute ordered jobs through the selected backend."""
    options = options or ExecutionOptions()
    jobs = list(job_specs)
    _validate_jobs(jobs)
    if not jobs:
        return []
    backend = get_backend(options.backend)
    logger.info("Executing %d job(s) with backend %s", len(jobs), options.backend)
    submission = _submit_validated(jobs, options, backend)
    results = wait_for_submission(submission, options=options, backend=backend)
    logger.info("Completed %d job(s) with backend %s", len(results), options.backend)
    return sorted(results, key=lambda result: result.index)


def submit_jobs(job_specs, options=None):
    """Submit independent jobs, detaching when the backend supports it.

    Scheduler-backed implementations return without waiting. Backends that do
    not advertise submit-only support complete synchronously.
    """
    options = options or ExecutionOptions()
    jobs = list(job_specs)
    _validate_jobs(jobs)
    if not jobs:
        return SubmissionHandle(
            backend=options.backend,
            work_dir=Path(options.work_dir or Path.cwd()),
            job_ids=(),
        )
    backend = get_backend(options.backend)
    submission = _submit_validated(jobs, options, backend)
    supports_submit_only = getattr(backend, "supports_submit_only", False)
    if not supports_submit_only:
        wait_for_submission(submission, options=options, backend=backend)
    if jobs:
        if supports_submit_only:
            logger.info(
                "Submitted %d job(s) with backend %s; manifest: %s",
                len(jobs),
                options.backend,
                submission.work_dir / "submission.json",
            )
        else:
            logger.info(
                "Completed %d job(s) synchronously with backend %s",
                len(jobs),
                options.backend,
            )
    return submission


def wait_for_submission(submission, *, options=None, backend=None):
    """Wait for a previously submitted job set and validate its outputs."""
    backend = backend or get_backend(submission.backend)
    try:
        results = backend.wait(submission)
        _validate_manifest_outputs(submission)
    except KeyboardInterrupt:
        _mark_submission(submission, "interrupted")
        _cancel_if_requested(backend, submission, options)
        raise
    except Exception:
        _mark_submission(submission, "failed")
        raise
    _mark_submission(submission, "completed")
    return sorted(results, key=lambda result: result.index)


def load_submission(path):
    """Load a submission handle from a JSON manifest."""
    manifest = Path(path)
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except OSError as exc:
        raise FileNotFoundError(f"Submission manifest not found: {manifest}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid submission manifest: {manifest}") from exc
    return SubmissionHandle.from_dict(payload)


def _submit_validated(jobs, options, backend):
    """Submit validated jobs and persist a remote manifest."""
    submission = backend.submit(jobs, options)
    submission.metadata["expected_outputs"] = {
        job.job_id: [str(Path(path).expanduser().resolve()) for path in job.output_paths]
        for job in jobs
        if job.output_paths
    }
    _mark_submission(submission, "submitted")
    return submission


def _validate_jobs(jobs):
    """Validate stable identifiers and expected outputs before submission."""
    if len({job.job_id for job in jobs}) != len(jobs):
        raise ValueError("Execution jobs must have unique job IDs.")
    if len({job.index for job in jobs}) != len(jobs):
        raise ValueError("Execution jobs must have unique input-order indices.")
    expected_outputs = [
        Path(path).expanduser().resolve() for job in jobs for path in job.output_paths
    ]
    if len(expected_outputs) != len(set(expected_outputs)):
        raise ValueError("Execution jobs contain duplicate expected output paths.")


def _mark_submission(submission, state):
    """Update the remote submission manifest when applicable."""
    if submission.backend == "local":
        return
    submission.metadata["state"] = state
    _write_manifest(submission)


def _cancel_if_requested(backend, submission, options):
    """Cancel a remote submission when interruption policy requests it."""
    configured = submission.metadata.get("cancel_on_interrupt", False)
    if options is not None:
        configured = configured or (options.backend_config or {}).get("cancel_on_interrupt", False)
    if configured:
        backend.cancel(submission)


def _validate_manifest_outputs(submission):
    """Validate output paths recorded in a durable submission manifest."""
    expected = submission.metadata.get("expected_outputs", {})
    if isinstance(expected, list):
        expected = {"unknown": expected}
    missing_outputs = [
        path for paths in expected.values() for path in paths if not Path(path).exists()
    ]
    if missing_outputs:
        raise FileNotFoundError(
            "Execution completed without expected output(s): " + ", ".join(missing_outputs)
        )


def map_ordered(
    function,
    items,
    *,
    backend="local",
    max_workers=None,
    backend_config=None,
    initializer=None,
    initargs=(),
    mp_start_method="fork",
    work_dir=None,
):
    """Apply ``function`` to items and return values in input order."""
    runtime_args = dict(config.args) if backend != "local" else None
    runtime_db_config = dict(config.db_config) if backend != "local" else None
    job_specs = [
        JobSpec(
            job_id=f"job-{index:06d}",
            index=index,
            function=function,
            item=item,
            initializer=initializer if backend != "local" else None,
            initargs=tuple(initargs) if backend != "local" else (),
            runtime_args=runtime_args,
            runtime_db_config=runtime_db_config,
        )
        for index, item in enumerate(items)
    ]
    execution_config = _load_backend_config(backend_config)
    if backend == "local":
        execution_config.setdefault("mp_start_method", mp_start_method)
    options = ExecutionOptions(
        backend=backend,
        max_workers=max_workers,
        work_dir=Path(work_dir) if work_dir else None,
        backend_config=execution_config,
        initializer=initializer,
        initargs=tuple(initargs),
    )
    return [result.value for result in execute_jobs(job_specs, options)]


def options_from_args(args, *, max_workers=None, work_dir=None):
    """Build execution options from an application argument dictionary."""
    config_value = args.get("backend_config")
    backend_config = _load_backend_config(config_value)
    return ExecutionOptions(
        backend=args.get("backend", "local"),
        max_workers=max_workers,
        work_dir=Path(work_dir) if work_dir else None,
        backend_config=backend_config,
    )


def _load_backend_config(config_value):
    """Load an inline or file-based backend configuration."""
    if isinstance(config_value, dict):
        return dict(config_value)
    if config_value:
        config_path = Path(config_value)
        if not config_path.is_file():
            raise FileNotFoundError(f"Backend configuration file not found: {config_path}")
        with config_path.open(encoding="utf-8") as handle:
            backend_config = yaml.safe_load(handle) or {}
        if not isinstance(backend_config, dict):
            raise ValueError("Backend configuration must contain a mapping.")
    else:
        return {}
    return backend_config


def _write_manifest(submission):
    """Write a durable submission manifest."""
    submission.work_dir.mkdir(parents=True, exist_ok=True)
    manifest = submission.work_dir / "submission.json"
    temporary = manifest.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(submission.as_dict(), indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest)
