"""Verify concurrent PSF evaluations, isolation, caching, and error handling."""

import os
import pickle
import time
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from simtools.job_execution.backends.base import BackendConfigurationError
from simtools.job_execution.job import JobResult
from simtools.ray_tracing import psf_evaluation as evaluation
from simtools.ray_tracing import psf_parameter_optimisation as psf

BATCH_METHOD = "_run_parameter_batch"
DISPATCH_METHOD = "_dispatch_parameter_evaluations"
INITIALIZE_MODELS = "initialize_simulation_models"
SIMULATE = "run_psf_simulation"
EXECUTE = "execute_jobs"
INITIALIZE_RUNTIME = "simtools.job_execution.worker._initialize_runtime"
ERROR_MESSAGE = "Invalid PSF input"
SIGMA = "sigma"
OTHER_PARAMETER = "other"
BACKEND = "backend"
SOURCE_CONFIG = {"kind": "test"}


@pytest.fixture
def optimizer(mocker, tmp_test_directory):
    args = {"site": "North", "telescope": "LSTN-01", "model_version": "6.0.0"}
    mocker.patch.object(evaluation, "config", SimpleNamespace(args=args, model_reader=None))
    return psf.PSFParameterOptimizer(
        object(), object(), args, {"measured": [1.0]}, np.array([1.0]), Path(tmp_test_directory)
    )


def test_cached_batch_deduplicates_and_preserves_order(optimizer, mocker):
    first = {SIGMA: [1.0]}
    second = {SIGMA: [2.0]}
    values = [(1.0, 2.0, None, None), (3.0, 4.0, None, None)]
    dispatch = mocker.patch.object(optimizer, DISPATCH_METHOD, return_value=values)
    assert optimizer._run_parameter_batch([first, second, first], True) == [*values, values[0]]
    dispatch.assert_called_once_with([first, second])
    assert optimizer.cache_misses == 2
    assert optimizer._run_parameter_batch([second, first], True) == values[::-1]
    assert optimizer.cache_hits == 2
    assert dispatch.call_count == 1


def test_failed_batch_result_is_not_cached(optimizer, mocker):
    params = {SIGMA: 1.0}
    dispatch = mocker.patch.object(optimizer, DISPATCH_METHOD, return_value=[None])
    assert optimizer._run_parameter_batch([params], True) == [None]
    assert optimizer._run_parameter_batch([params], True) == [None]
    assert dispatch.call_count == 2
    assert optimizer.simulation_cache == {}


def test_uncached_repetitions_remain_independent(optimizer, mocker):
    params = {SIGMA: 1.0}
    values = [(1.0, 2.0, None, None), (3.0, 4.0, None, None)]
    dispatch = mocker.patch.object(optimizer, DISPATCH_METHOD, return_value=values)
    assert optimizer._run_parameter_batch([params, params], False) == values
    dispatch.assert_called_once_with([params, params])
    assert optimizer.simulation_cache == {}
    assert optimizer.cache_misses == 0


def test_empty_batch_does_not_submit(optimizer, mocker):
    dispatch = mocker.patch.object(optimizer, DISPATCH_METHOD)
    assert optimizer._run_parameter_batch([], True) == []
    dispatch.assert_not_called()


@pytest.mark.parametrize(BACKEND, ["local", "htcondor"])
def test_dispatch_serializable_jobs_with_unique_paths(optimizer, mocker, backend):
    optimizer.args_dict.update({BACKEND: backend, "backend_config": {}})
    evaluation.config.model_reader = SimpleNamespace(source_config=SOURCE_CONFIG)
    execute = mocker.patch.object(
        evaluation, EXECUTE, return_value=[JobResult("first", 0, 2.0), JobResult("second", 1, 3.0)]
    )
    assert optimizer._dispatch_parameter_evaluations([{SIGMA: 1}, {SIGMA: 2}]) == [2.0, 3.0]
    jobs, options = execute.call_args.args
    assert options.backend == backend
    assert options.max_workers == min(optimizer.max_workers, len(jobs))
    assert jobs[0].item.output_dir != jobs[1].item.output_dir
    assert all(job.function is psf._run_psf_evaluation for job in jobs)
    assert [job.index for job in jobs] == [0, 1]
    for job in jobs:
        restored = pickle.loads(pickle.dumps(job))
        assert restored.item.parameters == job.item.parameters
        assert restored.runtime_args == optimizer.args_dict
        assert restored.model_source_config == SOURCE_CONFIG
    previous_paths = [job.item.output_dir for job in jobs]
    optimizer._dispatch_parameter_evaluations([{SIGMA: 3}])
    assert execute.call_args.args[0][0].item.output_dir not in previous_paths


def test_backend_failure_is_not_treated_as_simulation_failure(optimizer, mocker):
    mocker.patch.object(evaluation, EXECUTE, side_effect=BackendConfigurationError(ERROR_MESSAGE))
    with pytest.raises(BackendConfigurationError, match=ERROR_MESSAGE):
        optimizer._run_parameter_batch([{SIGMA: 1}], True)


@pytest.mark.parametrize("fails", [False, True])
def test_worker_isolates_models_and_restores_paths(optimizer, mocker, fails):
    handler = psf.IOHandler()
    before_paths, before_model = dict(handler.output_path), handler.model_path
    output_dir = optimizer.output_dir / "evaluation"
    args = optimizer.args_dict | {psf.CLEANUP: True, psf.KS_STATISTIC: True}
    job = evaluation._PSFEvaluationJob(args, {}, {}, np.array([1.0]), output_dir)
    telescope, site = object(), object()
    initialize = mocker.patch.object(psf, INITIALIZE_MODELS, return_value=(telescope, site, None))
    expected = (3.0, 0.2, 0.7, None)

    def simulate(*args, **kwargs):
        assert handler.output_path["default"] == output_dir
        assert args[:2] == (telescope, site)
        assert kwargs["use_ks_statistic"] is True
        if fails:
            raise ValueError(ERROR_MESSAGE)
        return expected

    mocker.patch.object(psf, SIMULATE, side_effect=simulate)
    cleanup = mocker.patch.object(psf.gen, "cleanup_intermediate_files")
    result = psf._run_psf_evaluation(job)
    assert result == (None if fails else expected)
    assert initialize.call_args.kwargs["model_directory"] == output_dir / "model"
    assert handler.output_path == before_paths
    assert handler.model_path == before_model
    cleanup.assert_called_once_with(output_dir, patterns=psf.INTERMEDIATE_PATTERNS)


def test_worker_restores_paths_after_unexpected_failure(optimizer, mocker):
    handler = psf.IOHandler()
    before_paths = dict(handler.output_path)
    mocker.patch.object(psf, INITIALIZE_MODELS, side_effect=OSError(ERROR_MESSAGE))
    job = evaluation._PSFEvaluationJob(optimizer.args_dict, {}, {}, None, optimizer.output_dir)
    with pytest.raises(OSError, match=ERROR_MESSAGE):
        psf._run_psf_evaluation(job)
    assert handler.output_path == before_paths


def test_monte_carlo_aggregates_successes_in_order(optimizer, mocker):
    params = {SIGMA: 1.0}
    mocker.patch.object(optimizer, "get_initial_parameters", return_value=params)
    batch = mocker.patch.object(
        optimizer, BATCH_METHOD, return_value=[(2.0, 4.0, None, None), None, (6.0, 8.0, None, None)]
    )
    result = optimizer.analyze_monte_carlo_error(3)
    batch.assert_called_once_with([params] * 3, use_cache=False)
    assert result[0] == pytest.approx(6.0)
    assert result[1] == pytest.approx(np.sqrt(8.0))
    assert result[2] == [4.0, 8.0]
    assert result[6] == pytest.approx(4.0)
    assert result[8] == [2.0, 6.0]


@pytest.mark.parametrize("values", [[], [1.0]])
def test_sample_deviation_with_insufficient_data(values):
    assert np.isnan(psf._sample_standard_deviation(values))


def test_real_local_workers_overlap(optimizer, monkeypatch):
    """A barrier requires two real processes to run evaluation work simultaneously."""
    barrier = get_context("fork").Barrier(2, timeout=10)

    def simulate(*_args, **_kwargs):
        started = time.monotonic()
        barrier.wait()
        return os.getpid(), started, time.monotonic(), None

    monkeypatch.setattr(psf, INITIALIZE_MODELS, lambda **_kwargs: (None, None, None))
    monkeypatch.setattr(psf, SIMULATE, simulate)
    # The worker runtime and model source are unrelated to process-pool scheduling.
    monkeypatch.setattr(INITIALIZE_RUNTIME, lambda _job: None)
    optimizer.max_workers = 2
    values = optimizer._run_parameter_batch([{}, {}], False)
    assert values[0][0] != values[1][0]
    assert all(value[0] != os.getpid() for value in values)
    assert max(value[1] for value in values) <= min(value[2] for value in values)


def test_gradient_step_does_not_retry_backend_failure(optimizer, mocker):
    calculate = mocker.patch.object(
        optimizer, "calculate_gradient", side_effect=BackendConfigurationError(ERROR_MESSAGE)
    )
    with pytest.raises(BackendConfigurationError, match=ERROR_MESSAGE):
        optimizer.perform_gradient_step_with_retries({SIGMA: 1.0}, 2.0, 0.1)
    calculate.assert_called_once()


@pytest.mark.parametrize("workers", [1, 2])
def test_serial_and_parallel_gradient_match_analytic_result(optimizer, monkeypatch, workers):
    def simulate(_telescope, _site, _args, parameters, *_rest, **_kwargs):
        return 1.0, 3.0 * parameters[SIGMA], None, None

    monkeypatch.setattr(psf, INITIALIZE_MODELS, lambda **_kwargs: (None, None, None))
    monkeypatch.setattr(psf, SIMULATE, simulate)
    monkeypatch.setattr(INITIALIZE_RUNTIME, lambda _job: None)
    optimizer.max_workers = workers
    gradient = optimizer.calculate_gradient({SIGMA: 2.0, OTHER_PARAMETER: 0.0}, 6.0)
    assert gradient[SIGMA] == pytest.approx(3.0)
    assert gradient[OTHER_PARAMETER] == pytest.approx(0.0)
