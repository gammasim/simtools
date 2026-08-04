"""Local process execution backend."""

import logging
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

from simtools.job_execution.backends.base import BackendExecutionError
from simtools.job_execution.job import JobResult, SubmissionHandle

logger = logging.getLogger(__name__)


def determine_max_workers(max_workers=None, default_fraction=0.6):
    """Determine the local worker count."""
    cpu_count = os.cpu_count() or 1
    if max_workers is None:
        return max(1, int(cpu_count * default_fraction))
    return max_workers if max_workers > 0 else cpu_count


def execute_job_spec(job_spec):
    """Execute one job specification in a worker process or HTCondor worker."""
    if job_spec.initializer is not None:
        job_spec.initializer(*job_spec.initargs)
    if job_spec.function is not None:
        return job_spec.function(job_spec.item)
    completed = subprocess.run(
        list(job_spec.command),
        check=True,
    )
    return completed.returncode


class LocalBackend:
    """Execute job specifications with local worker processes."""

    supports_submit_only = False

    def submit(self, job_specs, options):
        """Submit jobs to a local process pool."""
        item_list = list(job_specs)
        workers = determine_max_workers(options.max_workers)
        if workers == 1 or len(item_list) < 2:
            return self._submit_direct(item_list, options)
        return self._submit_pool(item_list, workers, options)

    @staticmethod
    def _submit_direct(item_list, options):
        """Execute a small job set in the controller process."""
        if options.initializer is not None:
            options.initializer(*options.initargs)
        results = []
        failures = []
        for job_spec in item_list:
            try:
                results.append(
                    JobResult(job_spec.job_id, job_spec.index, execute_job_spec(job_spec))
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                failures.append(f"{job_spec.job_id} (index {job_spec.index}): {exc}")
        if failures:
            raise BackendExecutionError("Local job failure(s): " + "; ".join(failures))
        return SubmissionHandle(
            backend="local",
            work_dir=Path(options.work_dir or Path.cwd()),
            job_ids=tuple(job.job_id for job in item_list),
            metadata={"direct_results": results},
        )

    @staticmethod
    def _submit_pool(item_list, workers, options):
        """Submit a job set to a local process pool."""
        start_method = options.backend_config.get("mp_start_method", "fork")
        context = get_context(str(start_method)) if start_method else None
        executor_kwargs = {
            "max_workers": int(workers),
            "initializer": options.initializer,
            "initargs": tuple(options.initargs),
        }
        if context is not None:
            executor_kwargs["mp_context"] = context
        executor = ProcessPoolExecutor(**executor_kwargs)
        futures = {executor.submit(execute_job_spec, job_spec): job_spec for job_spec in item_list}
        return SubmissionHandle(
            backend="local",
            work_dir=Path(options.work_dir or Path.cwd()),
            job_ids=tuple(job.job_id for job in item_list),
            metadata={"executor": executor, "futures": futures},
        )

    def wait(self, submission):
        """Collect local results in input order."""
        if "direct_results" in submission.metadata:
            return sorted(submission.metadata["direct_results"], key=lambda result: result.index)
        executor = submission.metadata["executor"]
        futures = submission.metadata["futures"]
        results = {}
        failures = []
        try:
            for future in as_completed(futures):
                job_spec = futures[future]
                try:
                    results[job_spec.index] = JobResult(
                        job_id=job_spec.job_id,
                        index=job_spec.index,
                        value=future.result(),
                    )
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    failures.append(f"{job_spec.job_id} (index {job_spec.index}): {exc}")
        finally:
            executor.shutdown(wait=True)
        if failures:
            raise BackendExecutionError("Local job failure(s): " + "; ".join(failures))
        return [results[index] for index in sorted(results)]

    def cancel(self, submission):
        """Cancel pending local jobs."""
        if "direct_results" in submission.metadata:
            return
        for future in submission.metadata["futures"]:
            future.cancel()
        submission.metadata["executor"].shutdown(wait=False, cancel_futures=True)
