"""Tests for serialized remote worker execution."""

import pickle
from pathlib import Path

from simtools.job_execution.job import CommandOptions, JobSpec
from simtools.job_execution.worker import execute_command, execute_job_spec, run, write_job_payload


def _square(value):
    """Return the square of one value."""
    return value * value


def test_execute_job_spec_runs_commands(mocker):
    """Command jobs invoke their executable and return its status."""
    completed = mocker.patch("simtools.job_execution.worker.subprocess.run")
    completed.return_value.returncode = 0
    job = JobSpec("job-000000", 0, command=("example", "--flag"))

    return_code, runtime, _ = execute_command(job)

    assert return_code == 0
    assert runtime >= 0
    call_args = completed.call_args
    assert call_args.args == (("example", "--flag"),)
    assert call_args.kwargs["check"] is True
    assert call_args.kwargs["stdout"] == -1
    assert call_args.kwargs["stderr"] == -1


def test_execute_command_applies_runtime_streams_environment_and_configuration(
    mocker, tmp_test_directory
):
    """Command jobs preserve their runtime, streams, environment, and options."""
    script = Path(tmp_test_directory) / "run.sh"
    script.write_text("#!/bin/sh\n", encoding="utf-8")
    stdin = (Path(tmp_test_directory) / "input.txt").open("w", encoding="utf-8")
    stdout = Path(tmp_test_directory) / "stdout.log"
    stderr = Path(tmp_test_directory) / "stderr.log"
    run = mocker.patch("simtools.job_execution.worker.subprocess.run")
    run.return_value.returncode = 7
    job = JobSpec(
        "job-000001",
        1,
        command=str(script),
        options=CommandOptions(
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            env={"SIMTOOLS_TEST": "yes"},
            runtime_environment=("container", "exec"),
            configuration={
                "enabled": True,
                "disabled": False,
                "count": 3,
                "items": ["a", "b"],
            },
            check=False,
        ),
    )

    return_code, runtime, _ = execute_command(job)

    assert return_code == 7
    assert runtime >= 0
    assert script.stat().st_mode & 0o100
    assert run.call_args.args == (
        ["container", "exec", str(script), "--enabled", "--count", "3", "--items", "a", "b"],
    )
    assert run.call_args.kwargs["stdin"].closed is False
    assert run.call_args.kwargs["stdout"].name == str(stdout)
    assert run.call_args.kwargs["stderr"].name == str(stderr)
    assert run.call_args.kwargs["env"]["SIMTOOLS_TEST"] == "yes"
    assert run.call_args.kwargs["check"] is False
    stdin.close()


def test_execute_job_spec_initializes_serialized_runtime(mocker):
    """Function jobs restore settings and IO paths in the worker interpreter."""
    config = mocker.patch("simtools.job_execution.worker.config")
    io_handler = mocker.patch("simtools.job_execution.worker.io_handler.IOHandler")
    function = mocker.Mock(return_value=4)
    args = {"output_path": "output", "model_path": "models", "sim_telarray_path": "simtel"}
    job = JobSpec(
        "job-000000",
        0,
        function=function,
        item=2,
        runtime_args=args,
    )

    assert execute_job_spec(job) == 4

    config.load.assert_called_once_with(args)
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
