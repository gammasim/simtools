"""Unit tests for simtools.job_execution.process_pool."""

import os
import time
from concurrent.futures import Future

import pytest


def _process_pools_available():
    """Return whether the host permits multiprocessing semaphore creation."""
    try:
        os.sysconf("SC_SEM_NSEMS_MAX")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _process_pools_available(),
    reason="the host does not permit multiprocessing semaphore creation",
)


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
