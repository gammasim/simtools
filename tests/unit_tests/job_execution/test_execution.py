"""Tests for the generic execution facade."""

import sys
import types
from pathlib import Path

import pytest

from simtools.job_execution import (
    ExecutionOptions,
    JobSpec,
    SubmissionHandle,
    map_ordered,
    options_from_args,
)
from simtools.job_execution.backends.base import BackendConfigurationError
from simtools.job_execution.backends.htcondor import HTCondorBackend
from simtools.job_execution.worker import run as run_worker


def _square(value):
    """Return the square of one value."""
    return value * value


def test_map_ordered_returns_input_order():
    """Local execution returns values in input order."""
    assert map_ordered(_square, [3, 1, 2], max_workers=2) == [9, 1, 4]


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


def test_worker_writes_result(tmp_test_directory):
    """The remote worker writes a successful serialized result."""
    run_directory = Path(tmp_test_directory)
    input_directory = run_directory / "inputs"
    input_directory.mkdir()
    result_directory = run_directory / "results"
    result_directory.mkdir()
    job = JobSpec("job-000000", 0, function=_square, item=4)

    import pickle

    with (input_directory / "job-000000.pkl").open("wb") as handle:
        pickle.dump(job, handle)

    assert run_worker(run_directory, "job-000000") == 0
    with (result_directory / "job-000000.pkl").open("rb") as handle:
        payload = pickle.load(handle)
    assert payload == {"ok": True, "value": 16}


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
        def submit(self, description, count, itemdata):
            captured["description"] = description
            captured["count"] = count
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
        backend_config={"request_cpus": 1},
    )
    handle = HTCondorBackend().submit(jobs, options)

    assert handle.scheduler_id == 17
    assert captured["count"] == 2
    assert captured["itemdata"] == [{"job_id": "job-000000"}, {"job_id": "job-000001"}]
    assert captured["description"]["request_cpus"] == "1"


def test_htcondor_rejects_unknown_configuration():
    """Scheduler configuration is explicit rather than silently ignored."""
    with pytest.raises(BackendConfigurationError, match="Unknown HTCondor configuration"):
        HTCondorBackend._validate_config({"not_a_submit_option": True})


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
