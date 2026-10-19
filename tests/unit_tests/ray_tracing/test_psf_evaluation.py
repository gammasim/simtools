"""Tests for PSF execution worker limits."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from simtools.job_execution.job import JobResult
from simtools.ray_tracing import psf_evaluation as evaluation

BACKEND = "backend"
BACKEND_ERROR = "Backend failed"
FINAL_RESULT = "final result"


@pytest.mark.parametrize(("cpus", "expected"), [(None, 1), (1, 1), (3, 1), (8, 4)])
def test_half_available_cpus(mocker, cpus, expected):
    mocker.patch.object(evaluation.os, "process_cpu_count", return_value=cpus)
    assert evaluation._psf_worker_count({}) == expected
    assert evaluation._psf_worker_count({evaluation.MAX_WORKERS: None}) == expected
    assert evaluation._psf_worker_count({evaluation.MAX_WORKERS: 7}) == 7


@pytest.mark.parametrize("workers", [0, -1])
def test_invalid_worker_limit(workers):
    with pytest.raises(ValueError, match="max_workers must be a positive integer"):
        evaluation._psf_worker_count({evaluation.MAX_WORKERS: workers})


@pytest.mark.parametrize("cleanup", [False, True])
@pytest.mark.parametrize("failure", [None, "simulation", BACKEND])
def test_dispatch_cleanup_removes_all_evaluation_files(
    tmp_test_directory, mocker, failure, cleanup
):
    output = Path(tmp_test_directory)
    final = output / "final.ecsv"
    final.write_text(FINAL_RESULT)
    mocker.patch.object(evaluation, "config", SimpleNamespace(args={}, model_reader=None))
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

    mocker.patch.object(evaluation, "execute_jobs", side_effect=execute)
    if failure == BACKEND:
        with pytest.raises(RuntimeError, match=BACKEND_ERROR):
            evaluation._dispatch_evaluations([{}, {}], args, {}, None, output, 2, execute)
    else:
        result = evaluation._dispatch_evaluations([{}, {}], args, {}, None, output, 2, execute)
        assert result == [1.0, None if failure else 1.0]
    assert paths[0].parents[1].name == "parallel_evaluations"
    assert paths[0].exists() == (not cleanup)
    assert paths[1].exists() == (not cleanup)
    assert paths[0].parents[1].exists() == (not cleanup)
    assert final.read_text() == FINAL_RESULT
