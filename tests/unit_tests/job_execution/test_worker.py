"""Tests for serialized remote worker execution."""

import pickle
import platform
from pathlib import Path

from simtools.job_execution.job import JobSpec
from simtools.job_execution.worker import execute_job_spec, run, write_job_payload


def _square(value):
    """Return the square of one value."""
    return value * value


def test_execute_job_spec_runs_commands(mocker):
    """Command jobs invoke their executable and return its status."""
    completed = mocker.patch("simtools.job_execution.worker.subprocess.run")
    completed.return_value.returncode = 0
    job = JobSpec("job-000000", 0, command=("example", "--flag"))

    assert execute_job_spec(job) == 0
    completed.assert_called_once_with(["example", "--flag"], check=True)


def test_worker_writes_failure_result_for_invalid_payload(tmp_test_directory):
    """Malformed payloads produce a durable failure record."""
    run_directory = Path(tmp_test_directory)
    (run_directory / "inputs").mkdir()
    (run_directory / "results").mkdir()
    with (run_directory / "inputs" / "job-000000.pkl").open("wb") as handle:
        pickle.dump({"payload_version": -1}, handle)

    assert run(run_directory, "job-000000") == 1
    with (run_directory / "results" / "job-000000.pkl").open("rb") as handle:
        result = pickle.load(handle)
    assert result["ok"] is False
    assert result["exception"] == "ValueError"


def test_write_job_payload_is_readable_by_worker(tmp_test_directory):
    """Payload serialization preserves the complete JobSpec."""
    path = Path(tmp_test_directory) / "job.pkl"
    job = JobSpec("job-000000", 0, function=_square, item=4)

    write_job_payload(job, path)

    with path.open("rb") as handle:
        payload = pickle.load(handle)
    assert payload["job_spec"] == job
    assert payload["payload_version"] == 1


def test_worker_accepts_python_patch_version_difference(tmp_test_directory):
    """Python patch releases remain compatible for serialized job payloads."""
    run_directory = Path(tmp_test_directory)
    (run_directory / "inputs").mkdir()
    (run_directory / "results").mkdir()
    payload_path = run_directory / "inputs" / "job-000000.pkl"
    write_job_payload(JobSpec("job-000000", 0, function=_square, item=4), payload_path)
    with payload_path.open("rb") as handle:
        payload = pickle.load(handle)
    payload["python_version"] = f"{platform.python_version().rsplit('.', 1)[0]}.0"
    with payload_path.open("wb") as handle:
        pickle.dump(payload, handle)

    assert run(run_directory, "job-000000") == 0


def test_worker_rejects_python_major_minor_version_mismatch(tmp_test_directory):
    """Different Python major/minor versions remain unsupported."""
    run_directory = Path(tmp_test_directory)
    (run_directory / "inputs").mkdir()
    (run_directory / "results").mkdir()
    payload_path = run_directory / "inputs" / "job-000000.pkl"
    write_job_payload(JobSpec("job-000000", 0, function=_square, item=4), payload_path)
    with payload_path.open("rb") as handle:
        payload = pickle.load(handle)
    major, minor = platform.python_version().split(".")[:2]
    payload["python_version"] = f"{major}.{int(minor) - 1}.0"
    with payload_path.open("wb") as handle:
        pickle.dump(payload, handle)

    assert run(run_directory, "job-000000") == 1
    with (run_directory / "results" / "job-000000.pkl").open("rb") as handle:
        result = pickle.load(handle)
    assert "Python major/minor version" in result["message"]
