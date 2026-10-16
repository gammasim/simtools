"""Prepare isolated PSF evaluation jobs for local and remote execution."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np

from simtools.job_execution.execution import execute_jobs, options_from_args
from simtools.job_execution.job import JobSpec
from simtools.settings import config

logger = logging.getLogger(__name__)
MAX_WORKERS = "max_workers"
JOB_PREFIX = "evaluation-"


def _psf_worker_count(args):
    """Use half the available CPUs unless a positive worker count is supplied."""
    workers = args.get(MAX_WORKERS)
    if workers is None:
        return max(1, (os.process_cpu_count() or 1) // 2)
    if workers < 1:
        raise ValueError("max_workers must be a positive integer")
    return workers


@dataclass(frozen=True)
class _PSFEvaluationJob:
    """Serializable inputs for an independent simulation with isolated model files."""

    args: dict
    parameters: dict
    measured_data: dict
    radius: np.ndarray
    output_dir: Path


def _dispatch_evaluations(parameters, args, measured_data, radius, output_dir, max_workers, worker):
    """Execute a batch with independent output paths and portable runtime settings."""
    directory = Path(output_dir) / "evaluations" / uuid4().hex
    runtime_args = dict(config.args)
    source_config = (
        dict(config.model_reader.source_config) if config.model_reader is not None else None
    )
    jobs = [
        JobSpec(
            job_id=f"{JOB_PREFIX}{index:06d}",
            index=index,
            function=worker,
            item=_PSFEvaluationJob(
                args=dict(args),
                parameters=params,
                measured_data=measured_data,
                radius=radius,
                output_dir=directory / f"{JOB_PREFIX}{index:06d}",
            ),
            runtime_args=runtime_args,
            model_source_config=source_config,
        )
        for index, params in enumerate(parameters)
    ]
    options = options_from_args(args, max_workers=min(max_workers, len(jobs)))
    logger.info(
        "Evaluating %d PSF simulations with backend %s (local worker limit %d)",
        len(jobs),
        options.backend,
        options.max_workers,
    )
    return [result.value for result in execute_jobs(jobs, options)]
