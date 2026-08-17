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


def _environment_entries(environment):
    """Parse the semicolon-separated HTCondor environment representation."""
    return {  # noqa: C416 - required by the Sonar maintainability rule
        key: value
        for key, value in (
            item.split("=", maxsplit=1) for item in environment.split(";") if "=" in item
        )
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
        ({"request_cpus": 1.5}, "request_cpus"),
        ({"request_cpus": True}, "request_cpus"),
        ({"poll_interval": 0}, "poll_interval"),
        ({"timeout": 0}, "timeout"),
        ({"cancel_on_interrupt": "yes"}, "cancel_on_interrupt"),
        ({"extra_submit_attributes": {"output": "other"}}, "Protected submit attribute"),
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

    entries = {  # noqa: C416 - required by the Sonar maintainability rule
        key: value
        for key, value in (
            item.split("=", maxsplit=1)
            for item in submit_values["environment"].split(";")
            if "=" in item
        )
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


def test_htcondor_rejects_job_ids_that_escape_the_run_directory(tmp_test_directory):
    """Job IDs must be safe before payload paths are constructed."""
    work_dir = Path(tmp_test_directory)
    (work_dir / "inputs").mkdir()
    job = JobSpec("../../target", 0, command=("echo", "ok"))

    with pytest.raises(BackendSubmissionError, match="Invalid job ID"):
        HTCondorBackend._serialize_jobs([job], work_dir)

    assert not (work_dir.parent / "target.pkl").exists()


def test_htcondor_build_config_preserves_explicit_falsey_options():
    """Zero, false, and empty mapping options remain available for validation."""
    options = ExecutionOptions(
        request_cpus=0,
        timeout=0,
        cancel_on_interrupt=False,
        keep_successful_artifacts=False,
        extra_submit_attributes={},
    )

    config = HTCondorBackend._build_config(options)

    assert config["request_cpus"] == 0
    assert config["timeout"] == 0
    assert config["cancel_on_interrupt"] is False
    assert config["keep_successful_artifacts"] is False
    assert config["extra_submit_attributes"] == {}
    with pytest.raises(BackendConfigurationError, match="request_cpus"):
        HTCondorBackend._validate_config(config)


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
