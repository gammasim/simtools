"""Execute one serialized job specification for a remote backend."""

import argparse
import logging
import pickle
import traceback
from pathlib import Path

from simtools.job_execution.backends.local import execute_job_spec

logger = logging.getLogger(__name__)


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


def run(run_directory, job_id, log_file=None):
    """Run one serialized job and write its result or failure record."""
    run_directory = Path(run_directory)
    input_path = run_directory / "inputs" / f"{job_id}.pkl"
    result_path = run_directory / "results" / f"{job_id}.pkl"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = None
    try:
        file_handler = _configure_worker_logging(log_file)
        with input_path.open("rb") as handle:
            job_spec = pickle.load(handle)
        value = execute_job_spec(job_spec)
        temporary = result_path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            pickle.dump({"ok": True, "value": value}, handle)
        temporary.replace(result_path)
        return 0
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.exception("Job %s failed", job_id)
        payload = {
            "ok": False,
            "exception": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        with result_path.open("wb") as handle:
            pickle.dump(payload, handle)
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
