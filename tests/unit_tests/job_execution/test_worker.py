"""Tests for serialized remote worker execution."""

import json
import shlex
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from astropy.tests.helper import assert_quantity_allclose

from simtools.job_execution import worker
from simtools.job_execution.job import CommandOptions, JobSpec
from simtools.job_execution.worker import execute_command, execute_job_spec, run, write_job_payload
from simtools.ray_tracing import mirror_panel_psf


def _square(value):
    """Return the square of one value."""
    return value * value


def test_execute_job_spec_runs_commands(mocker):
    """Command jobs invoke their executable and return its status."""
    completed = mocker.patch("simtools.job_execution.worker.subprocess.run")
    completed.return_value.returncode = 0
    job = JobSpec("job-000000", 0, command=("example", "--flag"))

    return_code, runtime, _, _ = execute_command(job)

    assert return_code == 0
    assert runtime >= 0
    call_args = completed.call_args
    assert call_args.args == (("example", "--flag"),)
    assert call_args.kwargs["check"] is True
    assert call_args.kwargs["stdout"] is None
    assert call_args.kwargs["stderr"] is None


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

    return_code, runtime, _, _ = execute_command(job)

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


def test_execute_command_script_with_spaces_and_configuration(tmp_test_directory):
    script = Path(tmp_test_directory) / "script with spaces.sh"
    script.write_text('#!/bin/sh\nprintf "%s" "$1"\n', encoding="utf-8")
    job = JobSpec(
        "job",
        0,
        command=str(script),
        options=CommandOptions(configuration={"enabled": True}, capture_output=True),
    )
    code, _, output, _ = execute_command(job)
    assert code == 0
    assert output == "--enabled"


def test_corsika_settings_use_job_environment(monkeypatch):
    monkeypatch.setenv("SIMTOOLS_CORSIKA_HE_INTERACTION", "epos")
    job = JobSpec(
        "job",
        0,
        command=("simulate",),
        options=CommandOptions(
            env={
                "SIMTOOLS_SIMULATION_SOFTWARE": "corsika",
                "SIMTOOLS_CORSIKA_HE_INTERACTION": "qgs3",
            }
        ),
    )
    assert worker._corsika_job_settings(job)[:2] == ("corsika", "qgs3")


def test_execute_job_spec_initializes_serialized_runtime(mocker):
    """Function jobs receive a rebuilt run context in the worker interpreter."""
    function = mocker.Mock(return_value=4)
    args = {"output_path": "output", "model_path": "models", "sim_telarray_path": "simtel"}
    job = JobSpec(
        "job-000000",
        0,
        function=function,
        item=2,
        context_spec={"args": args, "model_source": None, "model_path": "models"},
    )

    assert execute_job_spec(job) == 4

    function.assert_called_once_with(2)


def test_execute_job_spec_restores_serialized_model_source(mocker):
    """Workers reopen the immutable model source recorded at submission time."""
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
        context_spec={"args": {}, "model_source": source_config, "model_path": None},
    )

    assert execute_job_spec(job) == 2

    restore_reader.assert_called_once_with(source_config)


def test_worker_writes_failure_result_for_invalid_payload(tmp_test_directory):
    """Malformed payloads produce a durable failure record."""
    run_directory = Path(tmp_test_directory)
    (run_directory / "inputs").mkdir()
    (run_directory / "results").mkdir()
    with (run_directory / "inputs" / "job-000000.json").open("w", encoding="utf-8") as handle:
        json.dump({"payload_version": -1}, handle)

    assert run(run_directory, "job-000000") == 1
    with (run_directory / "results" / "job-000000.json").open(encoding="utf-8") as handle:
        result = json.load(handle)
    assert result["ok"] is False
    assert result["exception"] == "ValueError"


def test_write_job_payload_is_readable_by_worker(tmp_test_directory):
    """Payload serialization preserves the complete JobSpec."""
    path = Path(tmp_test_directory) / "job.json"
    job = JobSpec("job-000000", 0, command=("echo", "ok"))

    write_job_payload(job, path)

    assert worker.load_job_payload(path) == job
    with path.open(encoding="utf-8") as handle:
        assert json.load(handle)["payload_version"] == 2


def test_worker_rejects_unregistered_function_from_payload(tmp_test_directory):
    path = Path(tmp_test_directory) / "job.json"
    write_job_payload(JobSpec("job", 0, command=("echo", "ok")), path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["job_spec"]["value"]["function"] = "os:system"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Unregistered worker type"):
        worker.load_job_payload(path)


def test_worker_rejects_pickle_payload(tmp_test_directory):
    path = Path(tmp_test_directory) / "job.json"
    path.write_bytes(b"cos\nsystem\n(S'echo unsafe'\ntR.")
    with pytest.raises(ValueError, match="Expecting value"):
        worker.load_job_payload(path)


def test_worker_round_trips_supported_values(tmp_test_directory):
    path = Path(tmp_test_directory) / "job.json"
    item = {"path": Path("data/file"), "array": np.array([1, 2]), "quantity": 3 * u.m}
    write_job_payload(JobSpec("job", 0, command=("echo", "ok"), item=item), path)
    restored = worker.load_job_payload(path).item
    assert restored["path"] == item["path"]
    np.testing.assert_array_equal(restored["array"], item["array"])
    assert_quantity_allclose(restored["quantity"], item["quantity"])


def test_worker_round_trips_registered_function_and_record(tmp_test_directory):
    path = Path(tmp_test_directory) / "job.json"
    item = mirror_panel_psf.MirrorOptimizationJob("label", {"indices": (1, 2)}, 0, 1.5)
    job = JobSpec("job", 0, function=mirror_panel_psf._optimize_single_mirror_worker, item=item)
    write_job_payload(job, path)
    assert worker.load_job_payload(path) == job


@pytest.mark.parametrize("value", [object(), np.array([object()], dtype=object)])
def test_worker_rejects_unsupported_values(tmp_test_directory, value):
    path = Path(tmp_test_directory) / "job.json"
    with pytest.raises(TypeError, match=r"Unsupported worker data|Object arrays"):
        write_job_payload(JobSpec("job", 0, command=("echo",), item=value), path)


def test_worker_rejects_initializer(tmp_test_directory):
    path = Path(tmp_test_directory) / "job.json"
    with pytest.raises(ValueError, match="initializers are not supported"):
        write_job_payload(JobSpec("job", 0, command=("echo",), initializer=_square), path)


def test_worker_stages_tables_for_corsika_command_and_passes_local_path(
    tmp_test_directory, mocker, capfd, monkeypatch
):
    """CORSIKA worker commands use a checksummed worker-local interaction-table path."""
    run_directory = Path(tmp_test_directory)
    source = run_directory / "source-tables"
    scratch = run_directory / "scratch"
    source.mkdir()
    scratch.mkdir()
    (run_directory / "inputs").mkdir()
    monkeypatch.setenv("SIMTOOLS_CORSIKA_INTERACTION_TABLE_PATH", str(source))
    from tests.unit_tests.runtime.test_dependencies import _write_interaction_table_manifest

    _write_interaction_table_manifest(source)
    job = JobSpec(
        "job-000000",
        0,
        command=(
            sys.executable,
            "-c",
            "import os; print(os.environ['SIMTOOLS_CORSIKA_INTERACTION_TABLE_PATH'])",
            "--simulation_software",
            "corsika_sim_telarray",
            "--corsika_he_interaction",
            "epos",
            "--corsika_le_interaction",
            "urqmd",
        ),
    )
    worker.write_job_payload(job, run_directory / "inputs" / "job-000000.json")

    assert worker.run(run_directory, "job-000000", scratch_directory=scratch) == 0

    result = capfd.readouterr()
    assert str(scratch) in result.out
    assert not list(scratch.iterdir())
    assert (run_directory / "results" / "job-000000.json").is_file()


def test_worker_does_not_stage_tables_for_non_corsika_job(tmp_test_directory, mocker):
    """Non-CORSIKA worker jobs do not copy interaction tables."""
    run_directory = Path(tmp_test_directory)
    source = run_directory / "source-tables"
    source.mkdir()
    (run_directory / "inputs").mkdir()
    mocker.patch.dict("os.environ", {"SIMTOOLS_CORSIKA_INTERACTION_TABLE_PATH": str(source)})
    job = JobSpec("job-000000", 0, command=(sys.executable, "-c", "pass"))
    worker.write_job_payload(job, run_directory / "inputs" / "job-000000.json")

    assert worker.run(run_directory, "job-000000") == 0


def test_worker_stages_selected_tables_for_non_epos_model(tmp_test_directory, monkeypatch):
    """The manifest selects the requested model, excluding other models."""
    from tests.unit_tests.runtime.test_dependencies import _write_interaction_table_manifest

    root = Path(tmp_test_directory)
    source = root / "source-tables"
    source.mkdir()
    _write_interaction_table_manifest(source)
    monkeypatch.setenv("SIMTOOLS_CORSIKA_INTERACTION_TABLE_PATH", str(source))
    job = JobSpec(
        "job-000000",
        0,
        command=(
            "simulate",
            "--simulation_software",
            "corsika",
            "--corsika_he_interaction",
            "qgs3",
            "--corsika_le_interaction",
            "urqmd",
        ),
    )
    with worker._stage_tables_for_job(job, root) as staged:
        destination, _ = staged
        assert (destination / "qgsdat-III").is_file()
        assert not (destination / "epos.iniev").exists()


def test_table_source_prefers_configuration_then_command_over_environment(monkeypatch):
    monkeypatch.setenv("SIMTOOLS_CORSIKA_INTERACTION_TABLE_PATH", "environment")
    job = JobSpec(
        "job",
        0,
        command=("simulate", "--corsika_interaction_table_path=command"),
        options=CommandOptions(env={"SIMTOOLS_CORSIKA_INTERACTION_TABLE_PATH": "job-env"}),
    )
    assert worker._corsika_table_source(job) == "command"
    job = JobSpec(
        "job",
        0,
        command=job.command,
        options=CommandOptions(configuration={"corsika_interaction_table_path": "configuration"}),
    )
    assert worker._corsika_table_source(job) == "configuration"


def test_shell_command_receives_staged_path_after_explicit_argument(tmp_test_directory):
    script = Path(tmp_test_directory) / "show args.py"
    script.write_text("import sys\nprint(sys.argv[-1])\n", encoding="utf-8")
    shared = Path(tmp_test_directory) / "shared tables"
    staged = Path(tmp_test_directory) / "staged tables"
    job = JobSpec(
        "job",
        0,
        command=f"{shlex.quote(sys.executable)} {shlex.quote(str(script))} "
        f"--corsika_interaction_table_path {shlex.quote(str(shared))}",
        options=CommandOptions(capture_output=True),
    )
    assert worker._corsika_table_source(job) == str(shared)
    local_job = worker._job_with_table_path(job, staged)
    return_code, _, output, _ = execute_command(local_job)
    assert return_code == 0
    assert output.strip() == str(staged)


def test_command_job_without_options_inherits_output(capfd):
    job = JobSpec("job-000002", 2, command=(sys.executable, "-c", "print('child output')"))

    assert execute_job_spec(job) == 0
    assert "child output" in capfd.readouterr().out
