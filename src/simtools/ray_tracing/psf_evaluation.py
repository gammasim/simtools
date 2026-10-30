"""Prepare isolated PSF evaluation jobs for local and remote execution."""

import logging
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np

from simtools.context import RunContext
from simtools.job_execution.execution import execute_jobs, options_from_args
from simtools.job_execution.job import JobSpec
from simtools.utils.general import cleanup_intermediate_files

logger = logging.getLogger(__name__)
JOB_PREFIX = "evaluation-"


@dataclass(frozen=True)
class _PSFEvaluationJob:
    """Serializable inputs for an independent simulation with isolated model files."""

    args: dict
    parameters: dict
    measured_data: dict
    radius: np.ndarray
    output_dir: Path


def dispatch_evaluations(
    parameters, args, measured_data, radius, output_dir, max_workers, worker, context=None
):
    """Execute a batch with independent output paths and portable runtime settings."""
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    directory = output_root / "parallel_evaluations" / uuid4().hex
    context = context or RunContext.from_environment()
    source_config = context.model_reader.source_config if context.model_reader is not None else None
    mount_paths = [output_root]
    for key in ("path", "repository"):
        if source_config and source_config.get(key):
            mount_paths.append(Path(source_config[key]).expanduser().resolve())
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
            context_spec=context.to_spec(),
            mount_paths=tuple(mount_paths),
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
    try:
        return [result.value for result in execute_jobs(jobs, options)]
    finally:
        if args.get("cleanup", False):
            cleanup_intermediate_files(
                directory, patterns=("**/*",), directories=(directory, directory.parent)
            )
