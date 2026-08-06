"""Execute one serialized job specification for a remote backend."""

import argparse
import logging
import pickle
import re
import subprocess
import traceback
from pathlib import Path

from simtools.io import io_handler
from simtools.job_execution.job import JobSpec
from simtools.settings import config

logger = logging.getLogger(__name__)

_PAYLOAD_VERSION = 1
_JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--log-file", type=Path)
    return parser


def _configure_worker_logging(log_file):
    """Configure INFO-level logging for one remote worker."""
    if log_file is None:
        return None
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in root_logger.handlers:
        handler.setLevel(logging.WARNING)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    root_logger.addHandler(file_handler)
    return file_handler


def execute_job_spec(job_spec):
    """Execute one Python or command job specification."""
    _initialize_runtime(job_spec)
    if job_spec.initializer is not None:
        job_spec.initializer(*job_spec.initargs)
    if job_spec.function is not None:
        return job_spec.function(job_spec.item)
    return subprocess.run(list(job_spec.command), check=True).returncode


def _initialize_runtime(job_spec):
    """Restore the submitting application's runtime for function jobs."""
    if job_spec.runtime_args is None:
        return
    config.load(job_spec.runtime_args, job_spec.runtime_db_config)
    io_handler.IOHandler().set_paths(
        output_path=job_spec.runtime_args.get("output_path"),
        model_path=job_spec.runtime_args.get("model_path"),
    )


def write_job_payload(job_spec, path):
    """Serialize a versioned worker payload."""
    payload = {
        "payload_version": _PAYLOAD_VERSION,
        "job_spec": job_spec,
    }
    with Path(path).open("wb") as handle:
        pickle.dump(payload, handle)


def _load_job_payload(path):
    """Load and validate one trusted worker payload."""
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or payload.get("payload_version") != _PAYLOAD_VERSION:
        raise ValueError("Unsupported worker payload format.")
    job_spec = payload.get("job_spec")
    if not isinstance(job_spec, JobSpec):
        raise TypeError("Worker payload does not contain a JobSpec.")
    return job_spec


def _job_file(run_directory, directory, job_id):
    """Resolve a job-owned path below the private run directory."""
    if not _JOB_ID_PATTERN.fullmatch(job_id):
        raise ValueError(f"Invalid job ID: {job_id!r}.")
    return Path(run_directory) / directory / f"{job_id}.pkl"


def _write_result(path, payload):
    """Write a worker result atomically."""
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle)
    temporary.replace(path)


def run(run_directory, job_id, log_file=None):
    """Run one serialized job and write its result or failure record."""
    run_directory = Path(run_directory)
    input_path = _job_file(run_directory, "inputs", job_id)
    result_path = _job_file(run_directory, "results", job_id)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = None
    try:
        file_handler = _configure_worker_logging(log_file)
        job_spec = _load_job_payload(input_path)
        value = execute_job_spec(job_spec)
        _write_result(result_path, {"ok": True, "value": value})
        return 0
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.exception("Job %s failed", job_id)
        payload = {
            "ok": False,
            "exception": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_result(result_path, payload)
        return 1
    finally:
        if file_handler is not None:
            logging.getLogger().removeHandler(file_handler)
            file_handler.close()


def main():
    """Run the worker CLI."""
    args = _parser().parse_args()
    raise SystemExit(run(args.run_directory, args.job_id, args.log_file))


if __name__ == "__main__":
    main()
