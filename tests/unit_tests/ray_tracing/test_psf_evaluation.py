"""Tests for PSF execution worker limits."""

import pytest

from simtools.ray_tracing import psf_evaluation as evaluation


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
