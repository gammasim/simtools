"""Tests for PSF execution worker limits."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from simtools.job_execution.backends.htcondor import HTCondorBackend
from simtools.job_execution.job import JobResult
from simtools.ray_tracing import psf_evaluation as evaluation

BACKEND = "backend"
BACKEND_ERROR = "Backend failed"
FINAL_RESULT = "final result"
CONFIG = "config"
EXECUTE_JOBS = "execute_jobs"


@pytest.mark.parametrize("cleanup", [False, True])
@pytest.mark.parametrize("failure", [None, "simulation", BACKEND])
def test_dispatch_cleanup_removes_all_evaluation_files(
    tmp_test_directory, mocker, failure, cleanup
):
    output = Path(tmp_test_directory)
    final = output / "final.ecsv"
    final.write_text(FINAL_RESULT)
    mocker.patch.object(evaluation, CONFIG, SimpleNamespace(args={}, model_reader=None))
    paths = []
    args = {"cleanup": cleanup}

    def execute(jobs, _options):
        for job in jobs:
            path = job.item.output_dir
            paths.append(path)
            nested = path / "model"
            nested.mkdir(parents=True)
            (nested / "config.dat").write_text("intermediate")
        if failure == BACKEND:
            raise RuntimeError(BACKEND_ERROR)
        return [
            JobResult(job.job_id, job.index, None if failure and job.index else 1.0) for job in jobs
        ]

    mocker.patch.object(evaluation, EXECUTE_JOBS, side_effect=execute)
    if failure == BACKEND:
        with pytest.raises(RuntimeError, match=BACKEND_ERROR):
            evaluation.dispatch_evaluations([{}, {}], args, {}, None, output, 2, execute)
    else:
        result = evaluation.dispatch_evaluations([{}, {}], args, {}, None, output, 2, execute)
        assert result == [1.0, None if failure else 1.0]
    assert paths[0].parents[1].name == "parallel_evaluations"
    assert paths[0].exists() == (not cleanup)
    assert paths[1].exists() == (not cleanup)
    assert paths[0].parents[1].exists() == (not cleanup)
    assert final.read_text() == FINAL_RESULT


@pytest.mark.parametrize("source_key", [None, "path", "repository"])
def test_dispatch_mounts_external_model_and_output_roots(tmp_test_directory, mocker, source_key):
    root = Path(tmp_test_directory).resolve()
    output = root / "external-output"
    model = root / "external-model"
    model.mkdir()
    source = {source_key: str(model)} if source_key else None
    reader = SimpleNamespace(source_config=source) if source else None
    mocker.patch.object(evaluation, CONFIG, SimpleNamespace(args={}, model_reader=reader))
    execute = mocker.patch.object(evaluation, EXECUTE_JOBS, return_value=[])
    evaluation.dispatch_evaluations([{}], {}, {}, None, output, 1, str)
    jobs = execute.call_args.args[0]
    assert output.is_dir()
    assert jobs[0].item.output_dir.is_relative_to(output)
    expected = (output, model) if source else (output,)
    assert jobs[0].mount_paths == expected
    working = root / "working"
    binds = HTCondorBackend._build_container_bind_paths(True, None, working / "runs", jobs)
    assert output in binds
    assert (model in binds) == bool(source)
