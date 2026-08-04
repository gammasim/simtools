"""Execute one serialized job specification for a remote backend."""

import argparse
import pickle
import traceback
from pathlib import Path

from simtools.job_execution.backends.local import execute_job_spec


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    return parser


def run(run_directory, job_id):
    """Run one serialized job and write its result or failure record."""
    run_directory = Path(run_directory)
    input_path = run_directory / "inputs" / f"{job_id}.pkl"
    result_path = run_directory / "results" / f"{job_id}.pkl"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with input_path.open("rb") as handle:
            job_spec = pickle.load(handle)
        value = execute_job_spec(job_spec)
        temporary = result_path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            pickle.dump({"ok": True, "value": value}, handle)
        temporary.replace(result_path)
        return 0
    except Exception as exc:  # pylint: disable=broad-exception-caught
        payload = {
            "ok": False,
            "exception": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        with result_path.open("wb") as handle:
            pickle.dump(payload, handle)
        return 1


def main():
    """Run the worker CLI."""
    args = _parser().parse_args()
    raise SystemExit(run(args.run_directory, args.job_id))


if __name__ == "__main__":
    main()
