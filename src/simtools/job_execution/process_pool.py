"""Generic process pool helpers.

This module centralizes ProcessPoolExecutor usage to ensure a consistent
approach across simtools applications.

Features
- Ordered results (input order preserved)
- Configurable worker count
- Optional per-process initializer/initargs
- Configurable multiprocessing start method (e.g. 'fork', 'spawn')
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")  # type of input items
R = TypeVar("R")  # type of return values


def process_pool_map_ordered(
    func,
    items,
    max_workers=None,
    mp_start_method="fork",
    initializer=None,
    initargs=(),
):
    """Apply *func* to each item in *items* in a process pool.

    Parameters
    ----------
    func:
        Function to apply to each item.
    items:
        Iterable of items.
    max_workers:
        Number of worker processes. If None or <= 0, uses os.cpu_count().
    mp_start_method:
        Multiprocessing start method (e.g. 'fork', 'spawn'). If None, uses the
        default context.
    initializer, initargs:
        Optional per-process initializer and its arguments.

    Returns
    -------
    list
        Results ordered to match the input item order.

    Notes
    -----
    Exceptions raised in worker processes are re-raised when collecting results.
    """
    item_list = list(items)
    n_items = len(item_list)

    if max_workers is None or int(max_workers) <= 0:
        max_workers = os.cpu_count() or 1

    # create a temporary list of Nones to hold results in input order
    results: list[R] = [None] * n_items  # type: ignore[list-item]

    ctx = None
    if mp_start_method:
        ctx = get_context(str(mp_start_method))

    logger.debug(
        "Starting ProcessPoolExecutor: n_items=%d, max_workers=%s, start_method=%s",
        n_items,
        str(max_workers),
        str(mp_start_method),
    )

    executor_kwargs: dict[str, Any] = {
        "max_workers": int(max_workers),
        "initializer": initializer,
        "initargs": tuple(initargs),
    }
    if ctx is not None:
        executor_kwargs["mp_context"] = ctx

    with ProcessPoolExecutor(**executor_kwargs) as executor:
        future_to_index = {
            executor.submit(func, item): index for index, item in enumerate(item_list)
        }
        for fut in as_completed(future_to_index):
            index = future_to_index[fut]
            results[index] = fut.result()

    return results
