"""Tests for serialized remote worker execution."""

import pickle
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


def test_execute_job_spec_initializes_serialized_runtime(mocker):
    """Function jobs restore settings and IO paths in the worker interpreter."""
    config = mocker.patch("simtools.job_execution.worker.config")
    io_handler = mocker.patch("simtools.job_execution.worker.io_handler.IOHandler")
    function = mocker.Mock(return_value=4)
    args = {"output_path": "output", "model_path": "models", "sim_telarray_path": "simtel"}
    db_config = {"db_url": "mongodb://example"}
    job = JobSpec(
        "job-000000",
        0,
        function=function,
        item=2,
        runtime_args=args,
        runtime_db_config=db_config,
    )

    assert execute_job_spec(job) == 4

    config.load.assert_called_once_with(args, db_config)
    io_handler.return_value.set_paths.assert_called_once_with(
        output_path="output", model_path="models"
    )
    function.assert_called_once_with(2)


def test_execute_job_spec_restores_serialized_model_source(mocker):
    """Workers reopen the immutable model source recorded at submission time."""
    config = mocker.patch("simtools.job_execution.worker.config")
    io_handler = mocker.patch("simtools.job_execution.worker.io_handler.IOHandler")
    source_config = {"type": "git", "repository": "/models.git", "commit": "a" * 40}
    reader = mocker.Mock()
    restore_reader = mocker.patch(
        "simtools.job_execution.worker.create_model_reader_from_source_config",
        return_value=reader,
    )
    job = JobSpec(
        "job-000000",
        0,
        function=lambda value: value,
        item=2,
        runtime_args={},
        model_source_config=source_config,
    )

    assert execute_job_spec(job) == 2

    restore_reader.assert_called_once_with(source_config)
    config.set_model_reader.assert_called_once_with(reader)
    io_handler.return_value.set_paths.assert_called_once_with(output_path=None, model_path=None)


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
