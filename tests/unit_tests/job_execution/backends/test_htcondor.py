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
    """Raise a scheduler submission failure."""
    raise RuntimeError("denied")


def _raise_cancellation_error(*_args, **_kwargs):
    """Raise a scheduler cancellation failure."""
    raise RuntimeError("unavailable")


def _raise_offline_error(*_args, **_kwargs):
    """Raise a scheduler connection failure."""
    raise RuntimeError("offline")


def _raise_payload_error(*_args, **_kwargs):
    """Raise a payload serialization failure."""
    raise TypeError("bad payload")


def _raise_event_log_error(*_args, **_kwargs):
    """Raise an event-log access failure."""
    raise RuntimeError("missing log")


def _environment_entries(environment):
    """Parse the semicolon-separated HTCondor environment representation."""
    return {
        key: value
        for item in environment.split(";")
        for key, separator, value in (item.partition("="),)
        if separator
    }


def test_htcondor_validates_resource_sizes():
    """Memory and disk requests must be non-empty expressions."""
    with pytest.raises(BackendConfigurationError, match="request_memory"):
        HTCondorBackend._validate_config({"request_memory": ""})


def test_htcondor_validates_container_python_command():
    with pytest.raises(BackendConfigurationError, match="python_executable"):
        HTCondorBackend._validate_config({"python_executable": ""})


def test_htcondor_validates_container_target_directory():
    """Container scratch mounts must use an absolute path."""
    with pytest.raises(BackendConfigurationError, match="absolute path"):
        HTCondorBackend._validate_config({"container_target_dir": "workdir"})


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"unknown": True}, "Unknown HTCondor configuration"),
        ({"request_cpus": "invalid"}, "request_cpus"),
        ({"request_cpus": 0}, "request_cpus"),
        ({"request_cpus": 1.5}, "request_cpus"),
        ({"request_cpus": True}, "request_cpus"),
        ({"priority": "invalid"}, "priority"),
        ({"priority": 1.5}, "priority"),
        ({"priority": "01"}, "priority"),
        ({"request_memory": 4}, "request_memory"),
        ({"extra_submit_attributes": []}, "extra_submit_attributes"),
        ({"container_target_dir": ""}, "container_target_dir"),
        ({"poll_interval": 0}, "poll_interval"),
        ({"poll_interval": "invalid"}, "poll_interval"),
        ({"timeout": 0}, "timeout"),
        ({"timeout": "invalid"}, "timeout"),
        ({"cancel_on_interrupt": "yes"}, "cancel_on_interrupt"),
    ],
)
def test_htcondor_rejects_invalid_execution_configuration(config, message):
    """Scheduler settings are type-checked before job submission."""
    with pytest.raises(BackendConfigurationError, match=message):
        HTCondorBackend._validate_config(config)


def test_htcondor_uses_container_python_command(tmp_test_directory):
    job = JobSpec("job-000000", 0, command=("echo", "ok"))
    backend = HTCondorBackend()
    submit_values, _, _ = backend._build_submit_values(
        {
            "container_image": "/shared/simtools.sif",
            "python_executable": "/opt/conda/bin/python",
        },
        [job],
        Path(tmp_test_directory),
        Path(tmp_test_directory) / "scheduler.log",
    )

    assert submit_values["executable"] == "/usr/bin/env"
    assert submit_values["arguments"].startswith(
        "/opt/conda/bin/python -m simtools.job_execution.worker"
    )
    assert submit_values["universe"] == "container"
    assert submit_values["container_target_dir"] == "/simtools-run"
    bind_paths = _environment_entries(submit_values["environment"])["APPTAINER_BINDPATH"].split(",")
    assert Path(tmp_test_directory).resolve().parent.as_posix() in bind_paths
    assert Path.cwd().resolve().as_posix() in bind_paths


def test_htcondor_exposes_source_checkout_to_container_python(tmp_test_directory):
    """Container workers can import the source checkout used for submission."""
    job = JobSpec("job-000000", 0, command=("echo", "ok"))
    submit_values, _, _ = HTCondorBackend()._build_submit_values(
        {"container_image": "/shared/simtools.sif"},
        [job],
        Path(tmp_test_directory),
        Path(tmp_test_directory) / "scheduler.log",
    )

    entries = {
        key: value
        for item in submit_values["environment"].split(";")
        for key, separator, value in (item.partition("="),)
        if separator
    }
    source_path = Path(__file__).resolve().parents[4] / "src"
    assert source_path.as_posix() in entries["PYTHONPATH"].split(":")
    assert source_path.parent.as_posix() in entries["APPTAINER_BINDPATH"].split(",")


def test_htcondor_preserves_submission_working_directory_for_containers(tmp_test_directory):
    """Container jobs retain the submitter working directory and bind it."""
    work_dir = Path(tmp_test_directory) / "work"
    working_directory = Path.cwd().resolve()
    job = JobSpec("job-000000", 0, command=("echo", "ok"))

    submit_values, _, _ = HTCondorBackend()._build_submit_values(
        {"container_image": "/shared/simtools.sif"},
        [job],
        work_dir,
        work_dir / "scheduler.log",
        working_directory,
    )

    assert submit_values["initialdir"] == str(working_directory)
    bind_paths = _environment_entries(submit_values["environment"])["APPTAINER_BINDPATH"]
    assert str(working_directory) in bind_paths.split(",")


def test_htcondor_avoids_nested_container_bind_paths(tmp_test_directory):
    """Container jobs do not duplicate a bind already covered by the work directory."""
    work_dir = Path(tmp_test_directory) / "work"
    output_dir = Path(tmp_test_directory) / "output" / "job-000000"
    job = JobSpec("job-000000", 0, command=("echo", "ok"), mount_paths=(output_dir,))

    submit_values, _, _ = HTCondorBackend()._build_submit_values(
        {"container_image": "/shared/simtools.sif"},
        [job],
        work_dir,
        work_dir / "scheduler.log",
    )

    bind_paths = _environment_entries(submit_values["environment"])["APPTAINER_BINDPATH"].split(",")
    assert work_dir.resolve().parent.as_posix() in bind_paths
    assert Path.cwd().resolve().as_posix() in bind_paths


def test_htcondor_rewrites_controller_python_in_container_commands():
    """Nested commands use the configured container Python executable."""
    job = JobSpec(
        "job-000000",
        0,
        command=(sys.executable, "-m", "simtools.applications.simulate_prod"),
    )
    options = ExecutionOptions()

    prepared = HTCondorBackend._prepare_jobs(
        [job], options, {"container_image": "/shared/simtools.sif", "python_executable": "python3"}
    )

    assert prepared[0].command[0] == "python3"


def test_htcondor_uses_configured_container_target_directory(tmp_test_directory):
    job = JobSpec("job-000000", 0, command=("echo", "ok"))
    submit_values, _, _ = HTCondorBackend()._build_submit_values(
        {
            "container_image": "/shared/simtools.sif",
            "container_target_dir": "/scratch/simtools",
        },
        [job],
        Path(tmp_test_directory),
        Path(tmp_test_directory) / "scheduler.log",
    )

    assert submit_values["container_target_dir"] == "/scratch/simtools"


def test_htcondor_uses_submission_python_without_container(tmp_test_directory):
    """Non-container workers can import the source checkout used for submission."""
    job = JobSpec("job-000000", 0, command=("echo", "ok"))
    submit_values, _, _ = HTCondorBackend()._build_submit_values(
        {},
        [job],
        Path(tmp_test_directory),
        Path(tmp_test_directory) / "scheduler.log",
    )

    assert submit_values["executable"] == sys.executable
    assert submit_values["arguments"].startswith("-m simtools.job_execution.worker")
    assert sys.executable not in submit_values["arguments"]
    assert "universe" not in submit_values
    entries = _environment_entries(submit_values["environment"])
    source_path = Path(__file__).resolve().parents[4] / "src"
    assert entries["PYTHONPATH"].split(":")[0] == source_path.as_posix()


def test_htcondor_rejects_job_ids_that_escape_the_run_directory(tmp_test_directory):
    """Job IDs must be safe before payload paths are constructed."""
    work_dir = Path(tmp_test_directory)
    (work_dir / "inputs").mkdir()
    job = JobSpec("../../target", 0, command=("echo", "ok"))

    with pytest.raises(BackendSubmissionError, match="Invalid job ID"):
        HTCondorBackend._serialize_jobs([job], work_dir)

    assert not (work_dir.parent / "target.pkl").exists()


def test_htcondor_build_config_uses_the_single_backend_configuration_source():
    """HTCondor options come exclusively from the backend configuration mapping."""
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
    """Custom submit attributes cannot replace backend-generated settings."""
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


def test_htcondor_reads_dotenv_entries(tmp_test_directory):
    """Simple environment files are converted to scheduler syntax."""
    environment = Path(tmp_test_directory) / "environment"
    environment.write_text(
        "# comment\nexport FOO='bar baz'\nBAZ=qux # inline comment\n", encoding="utf-8"
    )

    result = HTCondorBackend._read_environment_file(environment)

    assert result == "FOO=bar baz;BAZ=qux"


def test_htcondor_binds_corsika_interaction_tables(tmp_test_directory):
    """CORSIKA interaction tables are bound into container jobs automatically."""
    environment = Path(tmp_test_directory) / "environment"
    table_path = "/shared/corsika-interaction-tables"
    environment.write_text(
        f"SIMTOOLS_CORSIKA_INTERACTION_TABLE_PATH={table_path}\n", encoding="utf-8"
    )

    result = HTCondorBackend._read_environment_file(environment)

    assert result == (
        f"SIMTOOLS_CORSIKA_INTERACTION_TABLE_PATH={table_path};APPTAINER_BINDPATH={table_path}"
    )


def test_htcondor_rejects_invalid_environment_entries(tmp_test_directory):
    """Environment files must contain KEY=VALUE entries."""
    environment = Path(tmp_test_directory) / "environment"
    environment.write_text("INVALID\n", encoding="utf-8")

    with pytest.raises(BackendConfigurationError, match="expected KEY=VALUE"):
        HTCondorBackend._read_environment_file(environment)


def test_htcondor_environment_parsing_preserves_quoted_comment_and_bind_paths():
    """Quoted comments remain values and duplicate container binds are removed."""
    entries = {}
    assert HTCondorBackend._parse_environment_line("VALUE='one # two'", "environment") == (
        "VALUE",
        "one # two",
    )

    HTCondorBackend._add_apptainer_bind_path(entries, "/one")
    HTCondorBackend._add_apptainer_bind_path(entries, "/one")
    HTCondorBackend._add_apptainer_bind_path(entries, "/two")

    assert entries == {"APPTAINER_BINDPATH": "/one,/two"}
    assert HTCondorBackend._parse_environment_line('VALUE="one\\" # two"', "environment") == (
        "VALUE",
        'one\\" # two',
    )


def test_htcondor_avoids_only_nested_same_destination_bind_paths():
    """Bind de-duplication compares container destinations, not host sources."""
    entries = {"APPTAINER_BINDPATH": "/shared"}

    HTCondorBackend._add_apptainer_bind_path(entries, "/shared/tables")

    assert entries == {"APPTAINER_BINDPATH": "/shared"}

    entries = {"APPTAINER_BINDPATH": "/shared:/container/shared"}
    HTCondorBackend._add_apptainer_bind_path(entries, "/shared/tables")

    assert entries == {"APPTAINER_BINDPATH": "/shared:/container/shared,/shared/tables"}


def test_htcondor_replaces_nested_container_destination_bind_paths():
    """A broader same-destination bind replaces narrower configured entries."""
    entries = {"APPTAINER_BINDPATH": "/shared/tables"}

    HTCondorBackend._add_apptainer_bind_path(entries, "/shared")

    assert entries == {"APPTAINER_BINDPATH": "/shared"}


def test_htcondor_environment_parsing_rejects_empty_key_and_ignores_empty_bind():
    """Environment keys are required and empty bind paths are harmless."""
    with pytest.raises(BackendConfigurationError, match="empty environment key"):
        HTCondorBackend._parse_environment_line("=value", "environment")

    entries = {}
    HTCondorBackend._add_apptainer_bind_path(entries, "")
    assert entries == {}
    assert HTCondorBackend._read_environment_file(None) is None


@pytest.mark.parametrize("key", ["container_image", "environment_file"])
def test_htcondor_rejects_missing_shared_paths(key):
    """Configured shared files must exist before submission."""
    with pytest.raises(BackendConfigurationError, match=key):
        HTCondorBackend._validate_config({key: "/missing/path"})


def test_htcondor_loads_and_caches_scheduler_bindings(monkeypatch):
    """The HTCondor bindings and schedd are initialized once."""
    schedd = object()
    module = types.SimpleNamespace(Schedd=lambda: schedd)
    monkeypatch.setitem(sys.modules, "htcondor2", module)
    backend = HTCondorBackend()

    assert backend._load_htcondor() is module
    assert backend._schedd is schedd
    assert backend._load_htcondor() is module


@pytest.mark.parametrize(
    "module_factory",
    [
        lambda: None,
        lambda: types.SimpleNamespace(Schedd=_raise_offline_error),
    ],
)
def test_htcondor_load_reports_binding_errors(monkeypatch, module_factory):
    """Missing bindings and unavailable schedds become configuration errors."""
    module = module_factory()
    if module is None:
        monkeypatch.setitem(sys.modules, "htcondor2", None)
    else:
        monkeypatch.setitem(sys.modules, "htcondor2", module)

    with pytest.raises(BackendConfigurationError):
        HTCondorBackend()._load_htcondor()


def test_htcondor_submit_empty_job_list_returns_handle(tmp_test_directory):
    """Empty submissions do not require an HTCondor installation."""
    handle = HTCondorBackend().submit([], ExecutionOptions(work_dir=tmp_test_directory))

    assert handle.backend == "htcondor"
    assert handle.job_ids == ()


def test_htcondor_submit_success_builds_handle(monkeypatch, tmp_test_directory):
    """A successful scheduler submission returns cluster and process metadata."""

    class _Result:
        def cluster(self):
            return 17

        first_proc = 4

    module = types.SimpleNamespace(Submit=lambda values: values)
    schedd = types.SimpleNamespace(submit=lambda *_args, **_kwargs: _Result())
    backend = HTCondorBackend()
    backend._htcondor = module
    backend._schedd = schedd
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
    """Per-job resource overrides are emitted as item data."""
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
    """Per-job and backend file paths are normalized before use."""
    image = Path(tmp_test_directory) / "image.sif"
    environment = Path(tmp_test_directory) / "environment"
    image.touch()
    environment.touch()
    job = JobSpec(
        "job-000000",
        0,
        command=("echo", "ok"),
        resources={"container_image": str(image)},
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
    ("resources", "config", "message"),
    [
        ({"unsupported": 1}, {}, "Unknown resource"),
        ({"container_image": "/missing.sif"}, {}, "container_image does not exist"),
        ({"container_image": None}, {}, "Every job must define"),
    ],
)
def test_htcondor_rejects_invalid_job_resources(resources, config, message):
    """Invalid per-job resource overrides fail before scheduler submission."""
    job = JobSpec("job-000000", 0, command=("echo", "ok"), resources=resources)

    with pytest.raises(BackendConfigurationError, match=message):
        HTCondorBackend._validate_job_resources([job], config)


def test_htcondor_creates_work_directory_and_serializes_jobs(tmp_test_directory):
    """Work directories contain isolated scheduler subdirectories and payloads."""
    options = ExecutionOptions(work_dir=tmp_test_directory)
    work_dir = HTCondorBackend._create_work_dir(options)
    job = JobSpec("job-000000", 0, command=("echo", "ok"))

    HTCondorBackend._serialize_jobs([job], work_dir)

    assert all(
        (work_dir / name).is_dir() for name in ("inputs", "results", "stdout", "stderr", "logs")
    )
    assert (work_dir / "inputs" / "job-000000.pkl").is_file()


def test_htcondor_resolves_relative_event_log(tmp_test_directory):
    """Relative event-log paths are placed below the shared work directory."""
    work_dir = Path(tmp_test_directory)

    event_log = HTCondorBackend._resolve_event_log({"log_path": "events/scheduler.log"}, work_dir)

    assert event_log == work_dir / "events" / "scheduler.log"
    assert event_log.parent.is_dir()


def test_htcondor_serialization_errors_are_wrapped(monkeypatch, tmp_test_directory):
    """Payload serialization failures identify the affected job."""
    monkeypatch.setattr(
        "simtools.job_execution.backends.htcondor.write_job_payload",
        _raise_payload_error,
    )
    job = JobSpec("job-000000", 0, command=("echo", "ok"))

    with pytest.raises(BackendSubmissionError, match=r"job-000000.*bad payload"):
        HTCondorBackend._serialize_jobs([job], Path(tmp_test_directory))


def test_htcondor_wait_handles_cluster_remove_and_event_log_errors(monkeypatch, tmp_test_directory):
    """Cluster removal and event-log failures are reported as execution errors."""

    class _EventLog:
        def events(self, stop_after):
            yield types.SimpleNamespace(cluster=17, type="CLUSTER_REMOVE")

    backend = HTCondorBackend()
    backend._htcondor = types.SimpleNamespace(JobEventLog=lambda _path: _EventLog())
    submission = SubmissionHandle(
        "htcondor",
        Path(tmp_test_directory),
        ("job-000000",),
        scheduler_id=17,
        process_ids={"job-000000": 0},
        metadata={"poll_interval": 1},
    )
    assert backend._wait_for_processes(submission) == ["process 0: CLUSTER_REMOVE"]

    monkeypatch.setattr(
        backend._htcondor,
        "JobEventLog",
        _raise_event_log_error,
    )
    with pytest.raises(BackendExecutionError, match="missing log"):
        backend._wait_for_processes(submission)


def test_htcondor_wait_timeout_and_expected_output_failures(monkeypatch, tmp_test_directory):
    """Timeouts cancel active jobs and missing declared outputs remain actionable failures."""
    submission = SubmissionHandle(
        "htcondor",
        Path(tmp_test_directory),
        ("job-000000",),
        scheduler_id=17,
        process_ids={"job-000000": 0},
        metadata={"expected_outputs": ["missing.dat"]},
    )
    assert HTCondorBackend._missing_output_failures(submission) == [
        "job unknown: missing expected output missing.dat"
    ]

    backend = HTCondorBackend()
    backend._htcondor = types.SimpleNamespace(JobEventLog=lambda _path: object())
    submission.metadata.update({"poll_interval": 1, "timeout": 0})
    cancelled = []
    monkeypatch.setattr(backend, "cancel", lambda handle: cancelled.append(handle))

    assert backend._wait_for_processes(submission) == ["process 0: timeout"]
    assert cancelled == [submission]


def test_htcondor_wait_cancels_cluster_when_a_job_is_held(monkeypatch, tmp_test_directory):
    """Held jobs stop the remaining cluster instead of leaving scheduler work behind."""

    class _Event(dict):
        cluster = 17
        proc = 0
        type = "JOB_HELD"

    class _EventLog:
        def events(self, stop_after):
            yield _Event()

    backend = HTCondorBackend()
    backend._htcondor = types.SimpleNamespace(JobEventLog=lambda _path: _EventLog())
    submission = SubmissionHandle(
        "htcondor",
        Path(tmp_test_directory),
        ("job-000000",),
        scheduler_id=17,
        process_ids={"job-000000": 0},
        metadata={"poll_interval": 1},
    )
    cancelled = []
    monkeypatch.setattr(backend, "cancel", lambda handle: cancelled.append(handle))

    failures = backend._wait_for_processes(submission)

    assert "JOB_HELD" in failures[0]
    assert cancelled == [submission]


def test_htcondor_wait_reports_scheduler_failures_and_preserves_artifacts(
    monkeypatch, tmp_test_directory
):
    """Scheduler failures are raised and successful-artifact retention is honored."""
    backend = HTCondorBackend()
    submission = SubmissionHandle(
        "htcondor",
        Path(tmp_test_directory),
        ("job-000000",),
        scheduler_id=17,
        process_ids={"job-000000": 0},
        metadata={"keep_successful_artifacts": True},
    )
    monkeypatch.setattr(backend, "_load_htcondor", lambda: object())
    monkeypatch.setattr(backend, "_wait_for_processes", lambda _submission: ["process 0: failed"])

    with pytest.raises(BackendExecutionError, match="process 0: failed"):
        backend.wait(submission)

    (submission.work_dir / "results").mkdir()
    HTCondorBackend._cleanup_successful_artifacts(submission)
    assert (submission.work_dir / "results").is_dir()


def test_htcondor_wait_returns_empty_for_empty_submission():
    """Waiting on an empty submission is a no-op."""
    assert HTCondorBackend().wait(SubmissionHandle("htcondor", Path("run"), ())) == []


def test_htcondor_next_event_returns_none_when_log_is_empty():
    """An empty event stream is treated as a polling interval."""

    class _EventLog:
        def events(self, stop_after):
            return iter(())

    assert HTCondorBackend._next_event(_EventLog(), 0) is None


def test_htcondor_process_event_ignores_unrelated_events():
    """Events from other clusters and non-terminal processes are ignored."""

    class _Event:
        cluster = 99
        proc = 4
        type = "JOB_SUBMITTED"

    remaining = {0}
    assert HTCondorBackend._process_event(_Event(), 17, remaining, {"JOB_TERMINATED"}) == []
    _Event.cluster = 17
    assert HTCondorBackend._process_event(_Event(), 17, remaining, {"JOB_TERMINATED"}) == []


def test_htcondor_process_event_tracks_success_and_failure():
    """Terminal scheduler events update remaining processes and failures."""

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


def test_htcondor_load_results_reports_missing_job_index(tmp_test_directory):
    """Results without manifest index metadata are rejected."""
    work_dir = Path(tmp_test_directory)
    (work_dir / "results").mkdir()
    with (work_dir / "results" / "job-000000.pkl").open("wb") as handle:
        pickle.dump({"ok": True, "value": 1}, handle)
    submission = SubmissionHandle(
        backend="htcondor",
        work_dir=work_dir,
        job_ids=("job-000000",),
        process_ids={"job-000000": 0},
    )

    results, failures = HTCondorBackend._load_results(submission)

    assert results == []
    assert "missing job index" in failures[0]


def test_htcondor_load_results_reports_each_invalid_payload(tmp_test_directory):
    """Missing, unreadable, invalid, and failed worker payloads stay diagnostic."""
    work_dir = Path(tmp_test_directory)
    result_dir = work_dir / "results"
    result_dir.mkdir()
    (result_dir / "unreadable.pkl").write_bytes(b"not a pickle")
    with (result_dir / "invalid.pkl").open("wb") as handle:
        pickle.dump(["not", "a mapping"], handle)
    with (result_dir / "failed.pkl").open("wb") as handle:
        pickle.dump({"ok": False, "message": "worker failed"}, handle)
    submission = SubmissionHandle(
        backend="htcondor",
        work_dir=work_dir,
        job_ids=("missing", "unreadable", "invalid", "failed"),
        process_ids={"missing": 0, "unreadable": 1, "invalid": 2, "failed": 3},
        metadata={"indices": {"unreadable": 1, "invalid": 2, "failed": 3}},
    )

    results, failures = HTCondorBackend._load_results(submission)

    assert results == []
    assert len(failures) == 4
    assert "missing result" in failures[0]
    assert "unreadable result" in failures[1]
    assert "invalid result payload" in failures[2]
    assert "worker failed" in failures[3]


def test_htcondor_wait_cleans_transient_artifacts_after_success(monkeypatch, tmp_test_directory):
    """Successful scheduler runs remove payload and stream artifacts by default."""
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

    assert [result.value for result in backend.wait(submission)] == [4]
    assert not (work_dir / "results").exists()


def test_htcondor_submission_failure_is_wrapped(monkeypatch, tmp_test_directory):
    """Scheduler submission errors retain an actionable backend exception."""
    backend = HTCondorBackend()
    fake_schedd = types.SimpleNamespace(submit=_raise_submission_error)
    backend._htcondor = types.SimpleNamespace(Submit=dict)
    backend._schedd = fake_schedd
    monkeypatch.setattr(backend, "_load_htcondor", lambda: backend._htcondor)
    monkeypatch.setattr(backend, "_create_work_dir", lambda _options: Path(tmp_test_directory))
    monkeypatch.setattr(backend, "_serialize_jobs", lambda *_args: None)
    job = JobSpec("job-000000", 0, command=("echo", "ok"))

    with pytest.raises(BackendSubmissionError, match="denied"):
        backend.submit([job], ExecutionOptions(work_dir=tmp_test_directory))


def test_htcondor_cancel_wraps_scheduler_errors():
    """Cancellation failures are reported through the backend execution error."""
    backend = HTCondorBackend()
    backend._htcondor = types.SimpleNamespace(JobAction=types.SimpleNamespace(Remove="remove"))
    backend._schedd = types.SimpleNamespace(act=_raise_cancellation_error)

    with pytest.raises(BackendExecutionError, match="unavailable"):
        backend.cancel(SubmissionHandle("htcondor", Path("run"), (), scheduler_id=17))


def test_htcondor_cancel_loads_scheduler_for_detached_submission(monkeypatch):
    """Cancellation loads HTCondor bindings when using a fresh backend instance."""
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
