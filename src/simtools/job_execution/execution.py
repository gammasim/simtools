"""Public execution facade used by long-running applications."""

import json
import logging
from pathlib import Path

import yaml

from simtools.job_execution.backends.registry import get_backend
from simtools.job_execution.job import ExecutionOptions, JobSpec

logger = logging.getLogger(__name__)


def execute_jobs(job_specs, options=None):
    """Execute ordered jobs through the selected backend."""
    options = options or ExecutionOptions()
    jobs = list(job_specs)
    _validate_jobs(jobs)
    backend = get_backend(options.backend)
    if jobs:
        logger.info("Executing %d job(s) with backend %s", len(jobs), options.backend)
    submission = backend.submit(jobs, options)
    _mark_submission(submission, options, "submitted")
    try:
        results = backend.wait(submission)
    except KeyboardInterrupt:
        _mark_submission(submission, options, "interrupted")
        _cancel_if_requested(backend, submission, options)
        raise
    except Exception:
        _mark_submission(submission, options, "failed")
        raise
    _mark_submission(submission, options, "completed")
    _validate_outputs(jobs)
    if jobs:
        logger.info("Completed %d job(s) with backend %s", len(results), options.backend)
    return sorted(results, key=lambda result: result.index)


def _validate_jobs(jobs):
    """Validate stable identifiers and expected outputs before submission."""
    if len({job.job_id for job in jobs}) != len(jobs):
        raise ValueError("Execution jobs must have unique job IDs.")
    if len({job.index for job in jobs}) != len(jobs):
        raise ValueError("Execution jobs must have unique input-order indices.")
    expected_outputs = [Path(path) for job in jobs for path in job.output_paths]
    if len(expected_outputs) != len(set(expected_outputs)):
        raise ValueError("Execution jobs contain duplicate expected output paths.")


def _mark_submission(submission, options, state):
    """Update the remote submission manifest when applicable."""
    if options.backend == "local":
        return
    submission.metadata["state"] = state
    _write_manifest(submission)


def _cancel_if_requested(backend, submission, options):
    """Cancel a remote submission when interruption policy requests it."""
    configured = (options.backend_config or {}).get("cancel_on_interrupt", False)
    if options.cancel_on_interrupt or configured:
        backend.cancel(submission)


def _validate_outputs(jobs):
    """Verify all declared output paths after execution."""
    missing_outputs = [
        str(path) for job in jobs for path in job.output_paths if not Path(path).is_file()
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
    job_specs = [
        JobSpec(
            job_id=f"job-{index:06d}",
            index=index,
            function=function,
            item=item,
            initializer=initializer if backend != "local" else None,
            initargs=tuple(initargs) if backend != "local" else (),
        )
        for index, item in enumerate(items)
    ]
    config = _load_backend_config(backend_config)
    if backend == "local":
        config.setdefault("mp_start_method", mp_start_method)
    options = ExecutionOptions(
        backend=backend,
        max_workers=max_workers,
        work_dir=Path(work_dir) if work_dir else None,
        backend_config=config,
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
    manifest.write_text(json.dumps(submission.as_dict(), indent=2, default=str) + "\n")
