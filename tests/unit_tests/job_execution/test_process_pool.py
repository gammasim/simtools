"""Unit tests for simtools.job_execution.process_pool."""

import os
import time
from concurrent.futures import Future
from multiprocessing import Manager

import pytest

from simtools.job_execution import process_pool as pp


def _sleep_then_return(args):
    """Sleep for `args[0]` seconds and then return `args[1]`."""
    delay_s, value = args
    time.sleep(float(delay_s))
    return value


def _identity(x):
    return x


def _raise_on_3(x):
    if int(x) == 3:
        raise ValueError("boom")
    return int(x)


def _init_record_pid(shared_pid_list):
    """Append worker PID to shared list."""
    shared_pid_list.append(os.getpid())


class _FakeExecutor:
    """Synchronous stand-in for ProcessPoolExecutor for deterministic tests."""

    last_kwargs = None

    def __init__(self, **kwargs):
        _FakeExecutor.last_kwargs = dict(kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, func, item):
        fut = Future()
        try:
            fut.set_result(func(item))
        except (ValueError, TypeError, RuntimeError) as exc:
            fut.set_exception(exc)
        return fut


def test_preserves_input_order_real_pool():
    """Results should be returned in the same order as inputs."""
    items = [(0.03, "a"), (0.00, "b"), (0.01, "c")]
    results = pp.process_pool_map_ordered(
        _sleep_then_return,
        items,
        max_workers=3,
        mp_start_method=None,
    )
    assert results == ["a", "b", "c"]


def test_propagates_worker_exception_real_pool():
    """Worker exceptions should be re-raised when collecting results."""
    with pytest.raises(ValueError, match="boom"):
        pp.process_pool_map_ordered(
            _raise_on_3,
            [1, 2, 3, 4],
            max_workers=2,
            mp_start_method=None,
        )


def test_initializer_runs_in_workers_real_pool():
    """Initializer should run in worker processes (at least one PID recorded)."""
    with Manager() as manager:
        pid_list = manager.list()
        results = pp.process_pool_map_ordered(
            _identity,
            list(range(4)),
            max_workers=2,
            mp_start_method=None,
            initializer=_init_record_pid,
            initargs=(pid_list,),
        )
        assert results == list(range(4))
        assert len(set(pid_list)) >= 1


def test_uses_cpu_count_when_max_workers_nonpositive(monkeypatch):
    """If max_workers is None or <=0, os.cpu_count() should be used."""
    monkeypatch.setattr(pp.os, "cpu_count", (7).__int__)
    monkeypatch.setattr(pp, "ProcessPoolExecutor", _FakeExecutor)
    monkeypatch.setattr(pp, "as_completed", list)

    results = pp.process_pool_map_ordered(_identity, [1, 2, 3], max_workers=0)
    assert results == [1, 2, 3]
    assert _FakeExecutor.last_kwargs["max_workers"] == 7


def test_does_not_set_mp_context_when_start_method_none(monkeypatch):
    """When mp_start_method is None, mp_context should not be passed to the executor."""
    monkeypatch.setattr(pp, "ProcessPoolExecutor", _FakeExecutor)
    monkeypatch.setattr(pp, "as_completed", list)
    pp.process_pool_map_ordered(_identity, [1], max_workers=1, mp_start_method=None)
    assert "mp_context" not in _FakeExecutor.last_kwargs


def test_single_worker(monkeypatch):
    """Should work with n_workers=1 (serial execution)."""
    monkeypatch.setattr(pp, "ProcessPoolExecutor", _FakeExecutor)
    monkeypatch.setattr(pp, "as_completed", list)
    results = pp.process_pool_map_ordered(_identity, [5, 6, 7], max_workers=1)
    assert results == [5, 6, 7]


def test_empty_input(monkeypatch):
    """Should handle empty input gracefully."""
    monkeypatch.setattr(pp, "ProcessPoolExecutor", _FakeExecutor)
    monkeypatch.setattr(pp, "as_completed", list)
    results = pp.process_pool_map_ordered(_identity, [], max_workers=2)
    assert results == []
