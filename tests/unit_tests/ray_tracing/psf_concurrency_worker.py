"""Run fork-dependent PSF test scenarios in a fresh single-threaded interpreter."""

import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from simtools.ray_tracing import psf_parameter_optimisation as psf
from simtools.runtime.context import RunContext


def _optimizer(output_dir):
    args = {"site": "North", "telescope": "LSTN-01", "model_version": "6.0.0"}
    context = RunContext.from_arguments(
        args, model_reader=SimpleNamespace(source_config={"kind": "test"})
    )
    optimizer = psf.PSFParameterOptimizer(
        object(), object(), args, {"measured": [1.0]}, np.array([1.0]), output_dir, context=context
    )
    optimizer.max_workers = 2
    return optimizer


def _overlap(optimizer):
    barrier = get_context("fork").Barrier(2, timeout=10)

    def simulate(*_args, **_kwargs):
        started = time.monotonic()
        barrier.wait()
        return os.getpid(), started, time.monotonic(), None

    with patch.object(psf, "run_psf_simulation", simulate):
        values = optimizer._run_parameter_batch([{}, {}], False)
    return {"parent_pid": os.getpid(), "values": values}


def _gradient(optimizer):
    def simulate(_telescope, _site, _args, parameters, *_rest, **_kwargs):
        return 1.0, 3.0 * parameters["sigma"], None, None

    with patch.object(psf, "run_psf_simulation", simulate):
        return optimizer.calculate_gradient({"sigma": 2.0, "other": 0.0}, 6.0)


def main():
    optimizer = _optimizer(Path(sys.argv[2]))
    with (
        patch.object(psf, "initialize_simulation_models", return_value=(None, None, None)),
        patch("simtools.job_execution.worker._initialize_runtime", return_value=None),
    ):
        result = _overlap(optimizer) if sys.argv[1] == "overlap" else _gradient(optimizer)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
