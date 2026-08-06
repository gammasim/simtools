"""Tests for the generic execution facade."""

import logging
import pickle
import sys
import types
from pathlib import Path

import pytest
import yaml

from simtools.job_execution import (
    ExecutionOptions,
    JobSpec,
    SubmissionHandle,
    execute_jobs,
    load_submission,
    map_ordered,
    options_from_args,
    submit_jobs,
    wait_for_submission,
)
from simtools.job_execution.backends.base import BackendConfigurationError, BackendExecutionError
from simtools.job_execution.backends.htcondor import HTCondorBackend
from simtools.job_execution.worker import run as run_worker
from simtools.job_execution.worker import write_job_payload
from simtools.settings import config


def _square(value):
    """Return the square of one value."""
    return value * value


def _log_square(value):
    """Log one worker message and return the square of one value."""
    logging.getLogger(__name__).info("processing value %s", value)
    return value * value


def _raise_keyboard_interrupt(_submission):
    """Raise an interrupt from a backend wait operation."""
    raise KeyboardInterrupt


class _RemoteBackend:
    """Small scheduler-like backend for facade tests."""

    supports_submit_only = True

    def __init__(self, results=()):
        self.results = list(results)
        self.cancelled = []

    def submit(self, jobs, options):
        return SubmissionHandle(
            backend="remote",
            work_dir=Path(options.work_dir),
            job_ids=tuple(job.job_id for job in jobs),
        )

    def wait(self, _submission):
        return self.results

    def cancel(self, submission):
        self.cancelled.append(submission)


def test_map_ordered_returns_input_order():
    """Local execution returns values in input order."""
    assert map_ordered(_square, [3, 1, 2], max_workers=2) == [9, 1, 4]


def test_map_ordered_serializes_runtime_for_remote_jobs(mocker):
    """Remote function jobs receive the submitting application's runtime configuration."""
    execute = mocker.patch("simtools.job_execution.execution.execute_jobs", return_value=[])
    args = {"output_path": "output", "sim_telarray_path": "simtel"}
    db_config = {"db_url": "mongodb://example"}
    config.load(args, db_config, resolve_sim_software_executables=False)

    try:
        assert map_ordered(_square, [2], backend="htcondor") == []
    finally:
        config.load(resolve_sim_software_executables=False)

    job = execute.call_args.args[0][0]
    assert job.runtime_args == args
    assert job.runtime_db_config == db_config


def test_execute_jobs_empty_input_does_not_resolve_backend():
    """An empty execution plan has no backend dependency or side effects."""
    assert execute_jobs([], ExecutionOptions(backend="not-installed")) == []


def test_execute_jobs_persists_remote_manifest_and_sorts_results(mocker, tmp_test_directory):
    """Remote execution records expected outputs and completes its manifest."""
    output = Path(tmp_test_directory) / "output.txt"
    output.touch()
    backend = _RemoteBackend(
        [
            types.SimpleNamespace(index=1, value="second"),
            types.SimpleNamespace(index=0, value="first"),
        ]
    )
    mocker.patch("simtools.job_execution.execution.get_backend", return_value=backend)
    jobs = [
        JobSpec("job-000000", 0, function=_square, item=1, output_paths=(output,)),
        JobSpec("job-000001", 1, function=_square, item=2),
    ]

    results = execute_jobs(jobs, ExecutionOptions(backend="remote", work_dir=tmp_test_directory))

    assert [result.value for result in results] == ["first", "second"]
    manifest = load_submission(Path(tmp_test_directory) / "submission.json")
    assert manifest.metadata["state"] == "completed"
    assert manifest.metadata["expected_outputs"] == {"job-000000": [str(output.resolve())]}


def test_submit_jobs_returns_immediately_for_scheduler_backend(mocker, tmp_test_directory):
    """Submit-only backends leave a submitted manifest for later waiting."""
    backend = _RemoteBackend()
    mocker.patch("simtools.job_execution.execution.get_backend", return_value=backend)

    submission = submit_jobs(
        [JobSpec("job-000000", 0, function=_square, item=1)],
        ExecutionOptions(backend="remote", work_dir=tmp_test_directory),
    )

    assert submission.metadata["state"] == "submitted"
    assert Path(tmp_test_directory, "submission.json").is_file()


@pytest.mark.parametrize(
    ("jobs", "message"),
    [
        (
            [
                JobSpec("same", 0, function=_square, item=1),
                JobSpec("same", 1, function=_square, item=2),
            ],
            "job IDs",
        ),
        (
            [
                JobSpec("first", 0, function=_square, item=1),
                JobSpec("second", 0, function=_square, item=2),
            ],
            "indices",
        ),
    ],
)
def test_execute_jobs_rejects_duplicate_identifiers(jobs, message):
    """Execution plans must provide unique job IDs and input indices."""
    with pytest.raises(ValueError, match=message):
        execute_jobs(jobs)


def test_execute_jobs_rejects_duplicate_output_paths(tmp_test_directory):
    """Two jobs cannot declare the same expected output."""
    output = Path(tmp_test_directory) / "output.txt"
    jobs = [
        JobSpec("first", 0, function=_square, item=1, output_paths=(output,)),
        JobSpec("second", 1, function=_square, item=2, output_paths=(output,)),
    ]

    with pytest.raises(ValueError, match="duplicate expected output"):
        execute_jobs(jobs)


def test_wait_for_submission_marks_failure_when_expected_output_is_missing(tmp_test_directory):
    """Missing outputs fail a remote submission and preserve its manifest state."""
    backend = _RemoteBackend([types.SimpleNamespace(index=0, value=1)])
    submission = SubmissionHandle(
        backend="remote",
        work_dir=Path(tmp_test_directory),
        job_ids=("job-000000",),
        metadata={"expected_outputs": {"job-000000": [str(Path(tmp_test_directory) / "missing")]}},
    )

    with pytest.raises(FileNotFoundError, match="missing"):
        wait_for_submission(submission, backend=backend)

    assert submission.metadata["state"] == "failed"


def test_wait_for_submission_cancels_interrupted_configured_submission(tmp_test_directory):
    """Interrupt handling records the state and invokes the configured cancellation policy."""
    backend = _RemoteBackend()
    backend.wait = _raise_keyboard_interrupt
    submission = SubmissionHandle(
        backend="remote", work_dir=Path(tmp_test_directory), job_ids=("job-000000",)
    )

    with pytest.raises(KeyboardInterrupt):
        wait_for_submission(
            submission,
            backend=backend,
            options=ExecutionOptions(cancel_on_interrupt=True),
        )

    assert submission.metadata["state"] == "interrupted"
    assert backend.cancelled == [submission]


def test_options_from_args_reads_yaml(tmp_test_directory):
    """Backend options can be loaded from a YAML file."""
    config_file = Path(tmp_test_directory) / "backend.yml"
    config_file.write_text("request_cpus: 2\nrequest_memory: 1GB\n", encoding="utf-8")

    options = options_from_args(
        {"backend": "htcondor", "backend_config": str(config_file)},
        work_dir=tmp_test_directory,
    )

    assert options.backend == "htcondor"
    assert options.backend_config["request_cpus"] == 2
    assert options.work_dir == Path(tmp_test_directory)


@pytest.mark.parametrize("config_value", [None, {"request_cpus": 2}])
def test_options_from_args_accepts_empty_and_inline_configuration(config_value):
    """Options preserve empty and inline backend configuration values."""
    options = options_from_args({"backend_config": config_value})

    assert options.backend == "local"
    assert options.backend_config == (config_value or {})


@pytest.mark.parametrize("content", ["- request_cpus\n", "request_cpus: ["])
def test_options_from_args_rejects_invalid_yaml_configuration(tmp_test_directory, content):
    """Backend configuration files must contain valid YAML mappings."""
    config_file = Path(tmp_test_directory) / "backend.yml"
    config_file.write_text(content, encoding="utf-8")

    with pytest.raises((ValueError, yaml.YAMLError)):
        options_from_args({"backend_config": config_file})


def test_submit_jobs_keeps_local_execution_blocking():
    """Submit-only requests remain blocking for the local backend."""
    jobs = [JobSpec("job-000000", 0, function=_square, item=1)]

    submission = submit_jobs(jobs, ExecutionOptions(backend="local"))

    assert submission.metadata["direct_results"][0].value == 1


def test_submission_handle_round_trip(tmp_test_directory):
    """Submission manifests can be restored for a later wait operation."""
    handle = SubmissionHandle(
        backend="htcondor",
        work_dir=Path(tmp_test_directory),
        job_ids=("job-000000",),
        scheduler_id=17,
        process_ids={"job-000000": 0},
        metadata={"expected_outputs": [str(Path(tmp_test_directory) / "output.hdf5")]},
    )

    restored = SubmissionHandle.from_dict(handle.as_dict())

    assert restored == handle


def test_load_submission_reads_manifest(tmp_test_directory):
    """A saved manifest can be loaded for a later wait operation."""
    manifest = Path(tmp_test_directory) / "submission.json"
    manifest.write_text(
        '{"backend": "htcondor", "work_dir": "run", "job_ids": []}\n',
        encoding="utf-8",
    )

    restored = load_submission(manifest)

    assert restored.backend == "htcondor"
    assert restored.work_dir == Path("run")


def test_worker_writes_result(tmp_test_directory):
    """The remote worker writes a successful serialized result."""
    run_directory = Path(tmp_test_directory)
    input_directory = run_directory / "inputs"
    input_directory.mkdir()
    result_directory = run_directory / "results"
    result_directory.mkdir()
    job = JobSpec("job-000000", 0, function=_square, item=4)

    write_job_payload(job, input_directory / "job-000000.pkl")

    assert run_worker(run_directory, "job-000000") == 0
    with (result_directory / "job-000000.pkl").open("rb") as handle:
        payload = pickle.load(handle)
    assert payload == {"ok": True, "value": 16}


def test_worker_writes_info_log(tmp_test_directory):
    """The remote worker writes INFO messages to its dedicated log file."""
    run_directory = Path(tmp_test_directory)
    input_directory = run_directory / "inputs"
    input_directory.mkdir()
    result_directory = run_directory / "results"
    result_directory.mkdir()
    log_file = run_directory / "logs" / "job-000000.log"
    job = JobSpec("job-000000", 0, function=_log_square, item=4)

    write_job_payload(job, input_directory / "job-000000.pkl")

    assert run_worker(run_directory, "job-000000", log_file) == 0
    assert "processing value 4" in log_file.read_text(encoding="utf-8")


def test_worker_rejects_path_traversal_job_id(tmp_test_directory):
    """Workers only load job-owned payload files from the private input directory."""
    with pytest.raises(ValueError, match="Invalid job ID"):
        run_worker(tmp_test_directory, "../payload")


def test_htcondor_backend_reports_missing_dependency(monkeypatch):
    """Selecting HTCondor without the optional package gives an actionable error."""
    monkeypatch.setitem(__import__("sys").modules, "htcondor2", None)
    with pytest.raises(BackendConfigurationError, match=r"gammasimtools\[htcondor\]"):
        HTCondorBackend()._load_htcondor()


def test_htcondor_submit_creates_one_process_per_job(monkeypatch, tmp_test_directory):
    """Submission contains one item-data entry for every job."""
    captured = {}

    class _Submit(dict):
        pass

    class _Result:
        def cluster(self):
            return 17

        def first_proc(self):
            return 0

    class _Schedd:
        def submit(self, description, itemdata):
            captured["description"] = description
            captured["itemdata"] = list(itemdata)
            return _Result()

    fake_module = types.SimpleNamespace(Submit=_Submit, Schedd=lambda: _Schedd())
    monkeypatch.setitem(sys.modules, "htcondor2", fake_module)

    jobs = [
        JobSpec("job-000000", 0, function=_square, item=1),
        JobSpec("job-000001", 1, function=_square, item=2),
    ]
    options = ExecutionOptions(
        backend="htcondor",
        work_dir=Path(tmp_test_directory),
        backend_config={"request_cpus": 1, "priority": 42},
    )
    handle = HTCondorBackend().submit(jobs, options)

    assert handle.scheduler_id == 17
    assert captured["itemdata"] == [{"job_id": "job-000000"}, {"job_id": "job-000001"}]
    assert captured["description"]["request_cpus"] == "1"
    assert captured["description"]["priority"] == "42"
    arguments = captured["description"]["arguments"]
    assert "--job-id $(job_id)" in arguments
    assert "--log-file" in arguments
    assert arguments.endswith("logs/$(job_id).log")
    assert "'$(job_id)'" not in arguments
    assert handle.metadata["job_log_dir"] == str(handle.work_dir / "logs")


def test_htcondor_rejects_unknown_configuration():
    """Scheduler configuration is explicit rather than silently ignored."""
    with pytest.raises(BackendConfigurationError, match="Unknown HTCondor configuration"):
        HTCondorBackend._validate_config({"not_a_submit_option": True})


def test_htcondor_rejects_unknown_per_job_resource():
    """Per-job resources use the same explicit scheduler vocabulary."""
    job = JobSpec("job-000000", 0, function=_square, item=1, resources={"gpu": 1})
    with pytest.raises(BackendConfigurationError, match="Unknown resource key"):
        HTCondorBackend._validate_job_resources([job])


@pytest.mark.parametrize("priority", ["high", 1.5, True])
def test_htcondor_rejects_invalid_priority(priority):
    """Scheduler priority must be an integer."""
    with pytest.raises(BackendConfigurationError, match="priority must be an integer"):
        HTCondorBackend._validate_config({"priority": priority})


def test_htcondor_event_log_uses_integer_poll_deadline(tmp_test_directory):
    """The version-2 bindings receive an integer event-log deadline."""
    captured = {}

    class _Event(dict):
        cluster = 17
        proc = 0
        type = "JOB_TERMINATED"

    class _EventLog:
        def events(self, stop_after):
            captured["stop_after"] = stop_after
            return iter([_Event(ReturnValue=0)])

    backend = HTCondorBackend()
    backend._htcondor = types.SimpleNamespace(JobEventLog=lambda _path: _EventLog())
    submission = SubmissionHandle(
        backend="htcondor",
        work_dir=Path(tmp_test_directory),
        job_ids=("job-000000",),
        scheduler_id=17,
        process_ids={"job-000000": 0},
        metadata={"poll_interval": 60.0, "timeout": 1},
    )

    assert backend._wait_for_processes(submission) == []
    assert captured["stop_after"] == 60


def test_htcondor_preserves_artifacts_when_expected_output_is_missing(
    monkeypatch, tmp_test_directory
):
    """Output validation happens before successful-artifact cleanup."""
    work_dir = Path(tmp_test_directory)
    input_dir = work_dir / "inputs"
    result_dir = work_dir / "results"
    input_dir.mkdir()
    result_dir.mkdir()
    (input_dir / "job-000000.pkl").touch()
    with (result_dir / "job-000000.pkl").open("wb") as handle:
        pickle.dump({"ok": True, "value": 1}, handle)
    submission = SubmissionHandle(
        backend="htcondor",
        work_dir=work_dir,
        job_ids=("job-000000",),
        scheduler_id=17,
        process_ids={"job-000000": 0},
        metadata={
            "indices": {"job-000000": 0},
            "expected_outputs": {"job-000000": [str(work_dir / "missing.hdf5")]},
        },
    )
    backend = HTCondorBackend()
    backend._htcondor = object()
    monkeypatch.setattr(backend, "_wait_for_processes", lambda _submission: [])

    with pytest.raises(BackendExecutionError, match="job job-000000"):
        backend.wait(submission)

    assert input_dir.exists()
    assert result_dir.exists()
