"""Tests for HTCondor backend validation and event handling."""

import sys
from pathlib import Path

import pytest

from simtools.job_execution.backends.base import BackendConfigurationError
from simtools.job_execution.backends.htcondor import HTCondorBackend
from simtools.job_execution.job import ExecutionOptions, JobSpec, SubmissionHandle


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
    assert (
        submit_values["environment"]
        == f"APPTAINER_BINDPATH={Path(tmp_test_directory).resolve().parent}"
    )


def test_htcondor_binds_declared_job_mount_paths(tmp_test_directory):
    """Container jobs bind the directories explicitly declared by their specifications."""
    work_dir = Path(tmp_test_directory) / "work"
    output_dir = Path(tmp_test_directory) / "output" / "job-000000"
    job = JobSpec("job-000000", 0, command=("echo", "ok"), mount_paths=(output_dir,))

    submit_values, _, _ = HTCondorBackend()._build_submit_values(
        {"container_image": "/shared/simtools.sif"},
        [job],
        work_dir,
        work_dir / "scheduler.log",
    )

    assert submit_values["environment"] == (
        f"APPTAINER_BINDPATH={work_dir.resolve().parent},{output_dir.resolve()}"
    )


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
    assert submit_values["arguments"].startswith(f"{sys.executable} -m")
    assert "universe" not in submit_values


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
        import pickle

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
