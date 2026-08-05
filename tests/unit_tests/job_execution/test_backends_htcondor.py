"""Tests for HTCondor backend validation and event handling."""

from pathlib import Path

import pytest

from simtools.job_execution.backends.base import BackendConfigurationError
from simtools.job_execution.backends.htcondor import HTCondorBackend
from simtools.job_execution.job import SubmissionHandle


def test_htcondor_validates_resource_sizes():
    """Memory and disk requests must be non-empty expressions."""
    with pytest.raises(BackendConfigurationError, match="request_memory"):
        HTCondorBackend._validate_config({"request_memory": ""})


def test_htcondor_reads_dotenv_entries(tmp_test_directory):
    """Simple environment files are converted to scheduler syntax."""
    environment = Path(tmp_test_directory) / "environment"
    environment.write_text("# comment\nexport FOO='bar baz'\nBAZ=qux\n", encoding="utf-8")

    result = HTCondorBackend._read_environment_file(environment)

    assert "FOO=bar baz" in result
    assert "BAZ=qux" in result


def test_htcondor_rejects_invalid_environment_entries(tmp_test_directory):
    """Environment files must contain KEY=VALUE entries."""
    environment = Path(tmp_test_directory) / "environment"
    environment.write_text("INVALID\n", encoding="utf-8")

    with pytest.raises(BackendConfigurationError, match="expected KEY=VALUE"):
        HTCondorBackend._read_environment_file(environment)


def test_htcondor_process_event_tracks_success_and_failure():
    """Terminal scheduler events update remaining processes and failures."""

    class _Event(dict):
        cluster = 17
        proc = 0
        type = "JOB_TERMINATED"

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
