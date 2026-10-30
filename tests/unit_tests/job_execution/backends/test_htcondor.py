"""Tests for HTCondor backend validation and event handling."""

import pickle
import sys
import types
from pathlib import Path

import pytest

from simtools.job_execution.backends.base import (
    BackendConfigurationError,
    BackendExecutionError,
    BackendSubmissionError,
)
from simtools.job_execution.backends.htcondor import HTCondorBackend
from simtools.job_execution.job import ExecutionOptions, JobSpec, SubmissionHandle


def _raise_submission_error(*_args, **_kwargs):
    raise RuntimeError("denied")


def _raise_cancellation_error(*_args, **_kwargs):
    raise RuntimeError("unavailable")


def _raise_offline_error(*_args, **_kwargs):
    raise RuntimeError("offline")


def _raise_payload_error(*_args, **_kwargs):
    raise TypeError("bad payload")


def _raise_event_log_error(*_args, **_kwargs):
    raise RuntimeError("missing log")


def _environment_entries(environment):
    """Parse the semicolon-separated HTCondor environment representation."""
    return {
        key: value
        for item in environment.split(";")
        for key, separator, value in (item.partition("="),)
        if separator
    }


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"request_memory": ""}, "request_memory"),
        ({"request_memory": 4}, "request_memory"),
        ({"python_executable": ""}, "python_executable"),
        ({"container_target_dir": "workdir"}, "absolute path"),
        ({"container_target_dir": ""}, "container_target_dir"),
        ({"unknown": True}, "Unknown HTCondor configuration"),
        ({"request_cpus": "invalid"}, "request_cpus"),
        ({"request_cpus": 0}, "request_cpus"),
        ({"request_cpus": 1.5}, "request_cpus"),
        ({"request_cpus": True}, "request_cpus"),
        ({"priority": "invalid"}, "priority"),
        ({"priority": 1.5}, "priority"),
        ({"priority": "01"}, "priority"),
        ({"extra_submit_attributes": []}, "extra_submit_attributes"),
        ({"poll_interval": 0}, "poll_interval"),
        ({"poll_interval": "invalid"}, "poll_interval"),
        ({"timeout": 0}, "timeout"),
        ({"timeout": "invalid"}, "timeout"),
        ({"cancel_on_interrupt": "yes"}, "cancel_on_interrupt"),
    ],
)
def test_htcondor_rejects_invalid_execution_configuration(config, message):
    with pytest.raises(BackendConfigurationError, match=message):
        HTCondorBackend._validate_config(config)


def test_htcondor_container_submit_values(tmp_test_directory):
    job = JobSpec("job-000000", 0, command=("echo", "ok"))
    submit_values, _, _ = HTCondorBackend()._build_submit_values(
        {
            "container_image": "/shared/simtools.sif",
            "python_executable": "/opt/conda/bin/python",
            "container_target_dir": "/scratch/simtools",
        },
        [job],
        Path(tmp_test_directory),
        Path(tmp_test_directory) / "scheduler.log",
    )

    assert submit_values["executable"] == "/usr/bin/env"
    assert submit_values["arguments"].startswith(
        "/opt/conda/bin/python -P -m simtools.job_execution.worker"
    )
    assert submit_values["universe"] == "container"
    assert submit_values["container_target_dir"] == "/scratch/simtools"
    bind_paths = _environment_entries(submit_values["environment"])["APPTAINER_BINDPATH"].split(",")
    assert Path(tmp_test_directory).resolve().parent.as_posix() in bind_paths
    assert Path.cwd().resolve().as_posix() not in bind_paths


def test_htcondor_container_does_not_use_submitter_pythonpath(tmp_test_directory):
    job = JobSpec("job-000000", 0, command=("echo", "ok"))
    submit_values, _, _ = HTCondorBackend()._build_submit_values(
        {"container_image": "/shared/simtools.sif"},
        [job],
        Path(tmp_test_directory),
        Path(tmp_test_directory) / "scheduler.log",
    )
    assert "PYTHONPATH" not in _environment_entries(submit_values["environment"])
    assert submit_values["arguments"].startswith("python -P -m simtools.job_execution.worker")
    assert submit_values["container_target_dir"] == "/simtools-run"


def test_htcondor_container_working_directory_is_not_bound_twice(tmp_test_directory):
    root = Path(tmp_test_directory)
    working_directory = root / "submit"
    other_bind = root.parent / f"{root.name}-inputs"
    environment_file = root / "environment"
    environment_file.write_text(
        f"APPTAINER_BINDPATH={working_directory},{other_bind},{working_directory}\n",
        encoding="utf-8",
    )
    job = JobSpec("job-000000", 0, command=("echo", "ok"), mount_paths=(working_directory,))

    submit_values, _, _ = HTCondorBackend()._build_submit_values(
        {"container_image": "/shared/simtools.sif", "environment_file": environment_file},
        [job],
        root / "work",
        root / "scheduler.log",
        working_directory,
    )

    assert submit_values["initialdir"] == str(working_directory)
    bind_paths = _environment_entries(submit_values["environment"])["APPTAINER_BINDPATH"].split(",")
    assert str(working_directory) not in bind_paths
    assert str(other_bind) in bind_paths
    assert len(bind_paths) == len(set(bind_paths))


def test_htcondor_non_container_uses_submission_python(tmp_test_directory):
    job = JobSpec("job-000000", 0, command=("echo", "ok"))
    submit_values, _, _ = HTCondorBackend()._build_submit_values(
        {},
        [job],
        Path(tmp_test_directory),
        Path(tmp_test_directory) / "scheduler.log",
    )

    assert submit_values["executable"] == sys.executable
    assert submit_values["arguments"].startswith("-m simtools.job_execution.worker")
    assert "universe" not in submit_values
    entries = _environment_entries(submit_values["environment"])
    source_path = Path(__file__).resolve().parents[4] / "src"
    assert entries["PYTHONPATH"].split(":")[0] == source_path.as_posix()


def test_htcondor_rewrites_controller_python_in_container_commands():
    job = JobSpec(
        "job-000000",
        0,
        command=(sys.executable, "-m", "simtools.applications.simulate_prod"),
    )
    prepared = HTCondorBackend._prepare_jobs(
        [job],
        ExecutionOptions(),
        {"container_image": "/shared/simtools.sif", "python_executable": "python3"},
    )
    assert prepared[0].command[0] == "python3"


def test_htcondor_rejects_job_ids_that_escape_the_run_directory(tmp_test_directory):
    work_dir = Path(tmp_test_directory)
    (work_dir / "inputs").mkdir()
    job = JobSpec("../../target", 0, command=("echo", "ok"))

    with pytest.raises(BackendSubmissionError, match="Invalid job ID"):
        HTCondorBackend._serialize_jobs([job], work_dir)
    assert not (work_dir.parent / "target.pkl").exists()


def test_htcondor_build_config_uses_the_single_backend_configuration_source():
    options = ExecutionOptions(
        backend_config={
            "request_cpus": 0,
            "timeout": 0,
            "cancel_on_interrupt": False,
            "keep_successful_artifacts": False,
            "extra_submit_attributes": {},
        }
    )
    config = HTCondorBackend._build_config(options)

    assert config["request_cpus"] == 0
    assert config["timeout"] == 0
    assert config["cancel_on_interrupt"] is False
    assert config["keep_successful_artifacts"] is False
    assert config["extra_submit_attributes"] == {}
    with pytest.raises(BackendConfigurationError, match="request_cpus"):
        HTCondorBackend._validate_config(config)


@pytest.mark.parametrize("attribute", ["output", "environment", "universe", "request_cpus"])
def test_htcondor_rejects_extra_attributes_overriding_generated_values(
    tmp_test_directory, attribute
):
    job = JobSpec("job-000000", 0, command=("echo", "ok"))
    with pytest.raises(BackendConfigurationError, match="Protected submit attribute"):
        HTCondorBackend()._build_submit_values(
            {
                "container_image": "/shared/simtools.sif",
                "extra_submit_attributes": {attribute: "x"},
            },
            [job],
            Path(tmp_test_directory),
            Path(tmp_test_directory) / "scheduler.log",
        )


def test_htcondor_reads_environment_files(tmp_test_directory):
    environment = Path(tmp_test_directory) / "environment"
    environment.write_text(
        "# comment\nexport FOO='bar baz'\nBAZ=qux # inline comment\n", encoding="utf-8"
    )
    assert HTCondorBackend._read_environment_file(environment) == "FOO=bar baz;BAZ=qux"
    assert HTCondorBackend._read_environment_file(None) is None

    table_path = "/shared/corsika-interaction-tables"
    environment.write_text(
        f"SIMTOOLS_CORSIKA_INTERACTION_TABLE_PATH={table_path}\n", encoding="utf-8"
    )
    assert HTCondorBackend._read_environment_file(environment) == (
        f"SIMTOOLS_CORSIKA_INTERACTION_TABLE_PATH={table_path};APPTAINER_BINDPATH={table_path}"
    )

    environment.write_text("INVALID\n", encoding="utf-8")
    with pytest.raises(BackendConfigurationError, match="expected KEY=VALUE"):
        HTCondorBackend._read_environment_file(environment)


def test_htcondor_environment_line_parsing():
    assert HTCondorBackend._parse_environment_line("VALUE='one # two'", "environment") == (
        "VALUE",
        "one # two",
    )
    assert HTCondorBackend._parse_environment_line('VALUE="one\\" # two"', "environment") == (
        "VALUE",
        'one\\" # two',
    )
    with pytest.raises(BackendConfigurationError, match="empty environment key"):
        HTCondorBackend._parse_environment_line("=value", "environment")


@pytest.mark.parametrize(
    ("initial", "new_path", "expected"),
    [
        ({}, "/one", "/one"),
        ({"APPTAINER_BINDPATH": "/one"}, "/one", "/one"),
        ({"APPTAINER_BINDPATH": "/one"}, "/two", "/one,/two"),
        ({}, "", None),
        # nested same-destination binds are not duplicated
        ({"APPTAINER_BINDPATH": "/shared"}, "/shared/tables", "/shared"),
        (
            {"APPTAINER_BINDPATH": "/shared:/container/shared"},
            "/shared/tables",
            "/shared:/container/shared,/shared/tables",
        ),
        # a broader same-destination bind replaces narrower entries
        ({"APPTAINER_BINDPATH": "/shared/tables"}, "/shared", "/shared"),
        (
            {"APPTAINER_BINDPATH": "/host/tables:/container/shared/tables:ro"},
            "/other:/container/shared",
            "/other:/container/shared",
        ),
    ],
)
def test_htcondor_add_apptainer_bind_path(initial, new_path, expected):
    entries = dict(initial)
    HTCondorBackend._add_apptainer_bind_path(entries, new_path)
    assert entries == ({"APPTAINER_BINDPATH": expected} if expected else initial)


@pytest.mark.parametrize("key", ["container_image", "environment_file"])
def test_htcondor_rejects_missing_shared_paths(key):
    with pytest.raises(BackendConfigurationError, match=key):
        HTCondorBackend._validate_config({key: "/missing/path"})


def test_htcondor_loads_and_caches_scheduler_bindings(monkeypatch):
    schedd = object()
    module = types.SimpleNamespace(Schedd=lambda: schedd)
    monkeypatch.setitem(sys.modules, "htcondor2", module)
    backend = HTCondorBackend()

    assert backend._load_htcondor() is module
    assert backend._schedd is schedd
    assert backend._load_htcondor() is module


@pytest.mark.parametrize(
    "module",
    [None, types.SimpleNamespace(Schedd=_raise_offline_error)],
)
def test_htcondor_load_reports_binding_errors(monkeypatch, module):
    monkeypatch.setitem(sys.modules, "htcondor2", module)
    with pytest.raises(BackendConfigurationError):
        HTCondorBackend()._load_htcondor()


def test_htcondor_submit_empty_job_list_returns_handle(tmp_test_directory):
    handle = HTCondorBackend().submit([], ExecutionOptions(work_dir=tmp_test_directory))
    assert handle.backend == "htcondor"
    assert handle.job_ids == ()


def test_htcondor_submit_success_builds_handle(monkeypatch, tmp_test_directory):
    class _Result:
        def cluster(self):
            return 17

        first_proc = 4

    module = types.SimpleNamespace(Submit=lambda values: values)
    backend = HTCondorBackend()
    backend._htcondor = module
    backend._schedd = types.SimpleNamespace(submit=lambda *_args, **_kwargs: _Result())
    work_dir = Path(tmp_test_directory)
    monkeypatch.setattr(backend, "_load_htcondor", lambda: module)
    monkeypatch.setattr(backend, "_create_work_dir", lambda _options: work_dir)
    monkeypatch.setattr(backend, "_serialize_jobs", lambda *_args: None)
    monkeypatch.setattr(backend, "_resolve_event_log", lambda _config, _work: work_dir / "events")

    jobs = [JobSpec("job-000000", 3, command=("echo", "ok"))]
    handle = backend.submit(jobs, ExecutionOptions(work_dir=work_dir))

    assert handle.scheduler_id == 17
    assert handle.process_ids == {"job-000000": 4}
    assert handle.metadata["indices"] == {"job-000000": 3}


def test_htcondor_resource_overrides_build_itemdata(tmp_test_directory):
    job = JobSpec(
        "job-000000",
        0,
        command=("echo", "ok"),
        resources={"request_cpus": 2, "request_memory": "4GB", "priority": 3},
    )
    backend = HTCondorBackend()
    values, defaults, keys = backend._build_submit_values(
        {}, [job], Path(tmp_test_directory), Path(tmp_test_directory) / "events"
    )

    assert values["request_cpus"] == "$(request_cpus)"
    assert set(keys) == {"request_cpus", "request_memory", "priority"}
    assert backend._build_itemdata([job], defaults, keys) == [
        {"job_id": "job-000000", "request_cpus": "2", "request_memory": "4GB", "priority": "3"}
    ]


def test_htcondor_normalizes_job_and_configured_paths(tmp_test_directory):
    image = Path(tmp_test_directory) / "image.sif"
    environment = Path(tmp_test_directory) / "environment"
    image.touch()
    environment.touch()
    job = JobSpec(
        "job-000000", 0, command=("echo", "ok"), resources={"container_image": str(image)}
    )
    options = ExecutionOptions(
        backend_config={"container_image": str(image), "environment_file": str(environment)}
    )

    prepared = HTCondorBackend._prepare_jobs([job], options, {})
    config = HTCondorBackend._build_config(options)

    assert prepared[0].resources["container_image"] == str(image.resolve())
    assert config["container_image"] == str(image.resolve())
    assert config["environment_file"] == str(environment.resolve())


@pytest.mark.parametrize(
    ("resources", "message"),
    [
        ({"unsupported": 1}, "Unknown resource"),
        ({"container_image": "/missing.sif"}, "container_image does not exist"),
        ({"container_image": None}, "Every job must define"),
    ],
)
def test_htcondor_rejects_invalid_job_resources(resources, message):
    job = JobSpec("job-000000", 0, command=("echo", "ok"), resources=resources)
    with pytest.raises(BackendConfigurationError, match=message):
        HTCondorBackend._validate_job_resources([job], {})


def test_htcondor_creates_work_directory_and_serializes_jobs(tmp_test_directory):
    work_dir = HTCondorBackend._create_work_dir(ExecutionOptions(work_dir=tmp_test_directory))
    job = JobSpec("job-000000", 0, command=("echo", "ok"))
    HTCondorBackend._serialize_jobs([job], work_dir)

    assert all(
        (work_dir / name).is_dir() for name in ("inputs", "results", "stdout", "stderr", "logs")
    )
    assert (work_dir / "inputs" / "job-000000.pkl").is_file()


def test_htcondor_resolves_relative_event_log(tmp_test_directory):
    work_dir = Path(tmp_test_directory)
    event_log = HTCondorBackend._resolve_event_log({"log_path": "events/scheduler.log"}, work_dir)
    assert event_log == work_dir / "events" / "scheduler.log"
    assert event_log.parent.is_dir()


def test_htcondor_serialization_errors_are_wrapped(monkeypatch, tmp_test_directory):
    monkeypatch.setattr(
        "simtools.job_execution.backends.htcondor.write_job_payload", _raise_payload_error
    )
    job = JobSpec("job-000000", 0, command=("echo", "ok"))
    with pytest.raises(BackendSubmissionError, match=r"job-000000.*bad payload"):
        HTCondorBackend._serialize_jobs([job], Path(tmp_test_directory))


def _submission(tmp_test_directory, **metadata):
    return SubmissionHandle(
        "htcondor",
        Path(tmp_test_directory),
        ("job-000000",),
        scheduler_id=17,
        process_ids={"job-000000": 0},
        metadata={"poll_interval": 1, **metadata},
    )


def test_htcondor_wait_handles_cluster_remove_and_event_log_errors(monkeypatch, tmp_test_directory):
    class _EventLog:
        def events(self, stop_after):
            yield types.SimpleNamespace(cluster=17, type="CLUSTER_REMOVE")

    backend = HTCondorBackend()
    backend._htcondor = types.SimpleNamespace(JobEventLog=lambda _path: _EventLog())
    submission = _submission(tmp_test_directory)
    assert backend._wait_for_processes(submission) == ["process 0: CLUSTER_REMOVE"]

    monkeypatch.setattr(backend._htcondor, "JobEventLog", _raise_event_log_error)
    with pytest.raises(BackendExecutionError, match="missing log"):
        backend._wait_for_processes(submission)


def test_htcondor_wait_timeout_and_expected_output_failures(monkeypatch, tmp_test_directory):
    submission = _submission(tmp_test_directory, expected_outputs=["missing.dat"])
    assert HTCondorBackend._missing_output_failures(submission) == [
        "job unknown: missing expected output missing.dat"
    ]

    backend = HTCondorBackend()
    backend._htcondor = types.SimpleNamespace(JobEventLog=lambda _path: object())
    submission.metadata.update({"timeout": 0})
    cancelled = []
    monkeypatch.setattr(backend, "cancel", lambda handle: cancelled.append(handle))

    assert backend._wait_for_processes(submission) == ["process 0: timeout"]
    assert cancelled == [submission]


def test_htcondor_wait_cancels_cluster_when_a_job_is_held(monkeypatch, tmp_test_directory):
    class _Event(dict):
        cluster = 17
        proc = 0
        type = "JOB_HELD"

    class _EventLog:
        def events(self, stop_after):
            yield _Event()

    backend = HTCondorBackend()
    backend._htcondor = types.SimpleNamespace(JobEventLog=lambda _path: _EventLog())
    submission = _submission(tmp_test_directory)
    cancelled = []
    monkeypatch.setattr(backend, "cancel", lambda handle: cancelled.append(handle))

    failures = backend._wait_for_processes(submission)

    assert "JOB_HELD" in failures[0]
    assert cancelled == [submission]


def test_htcondor_wait_reports_cancellation_failure(monkeypatch, tmp_test_directory):
    backend = HTCondorBackend()
    backend._htcondor = types.SimpleNamespace(JobEventLog=lambda _path: object())
    submission = _submission(tmp_test_directory, timeout=0)

    def cancel(_handle):
        raise BackendExecutionError("unavailable")

    monkeypatch.setattr(backend, "cancel", cancel)
    assert backend._wait_for_processes(submission) == [
        "process 0: timeout",
        "Cancellation failed: unavailable",
    ]


def test_htcondor_wait_polls_again_after_an_empty_event_batch(monkeypatch, tmp_test_directory):
    class _HeldEvent(dict):
        cluster = 17
        proc = 0
        type = "JOB_HELD"

    backend = HTCondorBackend()
    backend._htcondor = types.SimpleNamespace(JobEventLog=lambda _path: object())
    events = iter((None, _HeldEvent()))
    submission = _submission(tmp_test_directory)
    cancelled = []
    monkeypatch.setattr(backend, "_next_event", lambda *_args: next(events))
    monkeypatch.setattr(backend, "cancel", lambda handle: cancelled.append(handle))

    assert backend._wait_for_processes(submission)[0].startswith("process 0: JOB_HELD")
    assert cancelled == [submission]


def test_htcondor_wait_reports_scheduler_failures_and_preserves_artifacts(
    monkeypatch, tmp_test_directory
):
    backend = HTCondorBackend()
    submission = _submission(tmp_test_directory, keep_successful_artifacts=True)
    monkeypatch.setattr(backend, "_load_htcondor", lambda: object())
    monkeypatch.setattr(backend, "_wait_for_processes", lambda _submission: ["process 0: failed"])

    with pytest.raises(BackendExecutionError, match="process 0: failed"):
        backend.wait(submission)

    (submission.work_dir / "results").mkdir()
    HTCondorBackend._cleanup_successful_artifacts(submission)
    assert (submission.work_dir / "results").is_dir()


def test_htcondor_wait_returns_empty_for_empty_submission():
    assert HTCondorBackend().wait(SubmissionHandle("htcondor", Path("run"), ())) == []


def test_htcondor_next_event_returns_none_when_log_is_empty():
    class _EventLog:
        def events(self, stop_after):
            return iter(())

    assert HTCondorBackend._next_event(_EventLog(), 0) is None


def test_htcondor_process_event_ignores_unrelated_events():
    class _Event:
        cluster = 99
        proc = 4
        type = "JOB_SUBMITTED"

    remaining = {0}
    assert HTCondorBackend._process_event(_Event(), 17, remaining, {"JOB_TERMINATED"}) == []
    _Event.cluster = 17
    assert HTCondorBackend._process_event(_Event(), 17, remaining, {"JOB_TERMINATED"}) == []


def test_htcondor_process_event_tracks_success_and_failure():
    class _EventType:
        name = "JOB_TERMINATED"

        def __str__(self):
            return "5"

    class _Event(dict):
        cluster = 17
        proc = 0
        type = _EventType()

    remaining = {0, 1}
    assert (
        HTCondorBackend._process_event(_Event(ReturnValue=0), 17, remaining, {"JOB_TERMINATED"})
        == []
    )
    assert remaining == {1}

    _Event.proc = 1
    assert HTCondorBackend._process_event(_Event(ReturnValue=1), 17, remaining, {"JOB_TERMINATED"})
    assert not remaining


def test_htcondor_load_results_reports_invalid_payloads(tmp_test_directory):
    work_dir = Path(tmp_test_directory)
    result_dir = work_dir / "results"
    result_dir.mkdir()
    with (result_dir / "job-000000.pkl").open("wb") as handle:
        pickle.dump({"ok": True, "value": 1}, handle)
    (result_dir / "unreadable.pkl").write_bytes(b"not a pickle")
    with (result_dir / "invalid.pkl").open("wb") as handle:
        pickle.dump(["not", "a mapping"], handle)
    with (result_dir / "failed.pkl").open("wb") as handle:
        pickle.dump({"ok": False, "message": "worker failed"}, handle)
    submission = SubmissionHandle(
        backend="htcondor",
        work_dir=work_dir,
        job_ids=("job-000000", "missing", "unreadable", "invalid", "failed"),
        process_ids={"job-000000": 0, "missing": 0, "unreadable": 1, "invalid": 2, "failed": 3},
        metadata={"indices": {"unreadable": 1, "invalid": 2, "failed": 3}},
    )

    results, failures = HTCondorBackend._load_results(submission)

    assert results == []
    assert len(failures) == 5
    assert "missing job index" in failures[0]
    assert "missing result" in failures[1]
    assert "unreadable result" in failures[2]
    assert "invalid result payload" in failures[3]
    assert "worker failed" in failures[4]


@pytest.mark.parametrize("method", ["wait", "collect"])
def test_htcondor_wait_cleans_transient_artifacts_after_success(
    monkeypatch, tmp_test_directory, method
):
    work_dir = Path(tmp_test_directory)
    for directory in ("inputs", "results", "stdout", "stderr"):
        (work_dir / directory).mkdir()
    with (work_dir / "results" / "job-000000.pkl").open("wb") as handle:
        pickle.dump({"ok": True, "value": 4}, handle)
    submission = SubmissionHandle(
        backend="htcondor",
        work_dir=work_dir,
        job_ids=("job-000000",),
        process_ids={"job-000000": 0},
        metadata={"indices": {"job-000000": 0}},
    )
    backend = HTCondorBackend()
    monkeypatch.setattr(backend, "_load_htcondor", lambda: object())
    monkeypatch.setattr(backend, "_wait_for_processes", lambda _submission: [])

    assert [result.value for result in getattr(backend, method)(submission)] == [4]
    assert not (work_dir / "results").exists()


def test_htcondor_submission_failure_is_wrapped(monkeypatch, tmp_test_directory):
    backend = HTCondorBackend()
    backend._htcondor = types.SimpleNamespace(Submit=dict)
    backend._schedd = types.SimpleNamespace(submit=_raise_submission_error)
    monkeypatch.setattr(backend, "_load_htcondor", lambda: backend._htcondor)
    monkeypatch.setattr(backend, "_create_work_dir", lambda _options: Path(tmp_test_directory))
    monkeypatch.setattr(backend, "_serialize_jobs", lambda *_args: None)
    job = JobSpec("job-000000", 0, command=("echo", "ok"))

    with pytest.raises(BackendSubmissionError, match="denied"):
        backend.submit([job], ExecutionOptions(work_dir=tmp_test_directory))


def test_htcondor_cancel_wraps_scheduler_errors():
    backend = HTCondorBackend()
    backend._htcondor = types.SimpleNamespace(JobAction=types.SimpleNamespace(Remove="remove"))
    backend._schedd = types.SimpleNamespace(act=_raise_cancellation_error)

    with pytest.raises(BackendExecutionError, match="unavailable"):
        backend.cancel(SubmissionHandle("htcondor", Path("run"), (), scheduler_id=17))


def test_htcondor_cancel_loads_scheduler_for_detached_submission(monkeypatch):
    backend = HTCondorBackend()
    calls = []
    scheduler = types.SimpleNamespace(
        act=lambda action, constraint: calls.append((action, constraint))
    )
    htcondor = types.SimpleNamespace(JobAction=types.SimpleNamespace(Remove="remove"))

    def load_htcondor():
        backend._htcondor = htcondor
        backend._schedd = scheduler
        return htcondor

    monkeypatch.setattr(backend, "_load_htcondor", load_htcondor)
    backend.cancel(SubmissionHandle("htcondor", Path("run"), (), scheduler_id=17))
    assert calls == [("remove", "ClusterId == 17")]


@pytest.mark.parametrize(
    ("queued", "history", "expected"),
    [
        ([{"ProcId": 0}], [], False),
        ([], [{"ProcId": 0, "JobStatus": 4}], True),
        ([], [{"ProcId": 0, "JobStatus": 3}], True),
        ([], [], None),
        ([], [{"ProcId": 0, "JobStatus": 5}], None),
    ],
)
def test_is_finished_requires_terminal_history(
    mocker, queued, history, expected, tmp_test_directory
):
    backend = HTCondorBackend()
    mocker.patch.object(backend, "_load_htcondor")
    backend._schedd = mocker.Mock()
    backend._schedd.query.return_value = queued
    backend._schedd.history.return_value = history
    submission = SubmissionHandle(
        backend="htcondor",
        work_dir=Path(tmp_test_directory),
        job_ids=("job",),
        scheduler_id=1,
        process_ids={"job": 0},
    )
    if expected is None:
        with pytest.raises(BackendExecutionError, match="incomplete HTCondor history"):
            backend.is_finished(submission)
    else:
        assert backend.is_finished(submission) is expected


def test_is_finished_refuses_unknown_scheduler_state(mocker, tmp_test_directory):
    backend = HTCondorBackend()
    submission = SubmissionHandle(
        backend="htcondor", work_dir=Path(tmp_test_directory), job_ids=("job",)
    )
    with pytest.raises(BackendExecutionError, match="missing scheduler identifiers"):
        backend.is_finished(submission)
    submission.scheduler_id = 1
    submission.process_ids = {"job": 0}
    mocker.patch.object(backend, "_load_htcondor")
    backend._schedd = mocker.Mock()
    backend._schedd.query.side_effect = RuntimeError("offline")
    with pytest.raises(BackendExecutionError, match="Cannot confirm finished HTCondor jobs"):
        backend.is_finished(submission)


def test_collect_reports_failed_workers(mocker, tmp_test_directory):
    backend = HTCondorBackend()
    mocker.patch.object(backend, "_load_results", return_value=([], ["worker failed"]))
    submission = SubmissionHandle(
        backend="htcondor", work_dir=Path(tmp_test_directory), job_ids=("job",)
    )
    with pytest.raises(BackendExecutionError, match="worker failed"):
        backend.collect(submission)
