"""Pre-submission path checks for science-test production runs.

Checks run on the submit host and in one high-priority worker job using the production
backend configuration. Checked are all absolute context paths and the absolute ``*_PATH``
entries of the job environment file.
"""

import copy
import shlex
from pathlib import Path

from simtools.job_execution.execution import submit_jobs, wait_for_submission
from simtools.job_execution.job import ExecutionOptions, JobSpec
from simtools.runners.simtools_runner import resolve_nested_configuration_files
from simtools.science_tests.acceptance import nonempty

PREFLIGHT_PRIORITY = 1_000_000
_CHECK_PATH_FUNCTION = (
    "check_path() {\n"
    '  if ! test -r "$2" || ! find -L "$2" -type f -size +0c -print -quit | grep -q .; then\n'
    '    echo "PREFLIGHT_FAILED: $1 ($2)" >&2; exit 1\n'
    "  fi\n"
    '  echo "PREFLIGHT_OK: $1 ($2)"\n'
    "}"
)


def context_paths(context):
    """Return the absolute paths configured in a science-test context.

    Parameters
    ----------
    context : dict
        Science-test context.

    Returns
    -------
    dict
        Absolute paths by context key.
    """
    return {
        key: Path(value).expanduser()
        for key, value in context.items()
        if isinstance(value, str) and Path(value).expanduser().is_absolute()
    }


def validate_context_paths(context):
    """Require every absolute context path to be a non-empty file or directory.

    Parameters
    ----------
    context : dict
        Science-test context.
    """
    errors = [
        f"{key} is missing or empty: {path}"
        for key, path in context_paths(context).items()
        if not nonempty(path)
    ]
    if errors:
        raise ValueError("Science-test preflight failed:\n- " + "\n- ".join(errors))


def run_worker_canary(test, context, work_dir):
    """Run one worker job checking the context and environment paths in the container.

    Parameters
    ----------
    test : simtools.science_tests.execution.PreparedTest
        Prepared production test whose first application defines ``backend_config``.
    context : dict
        Science-test context.
    work_dir : pathlib.Path
        Work directory for the canary job.
    """
    applications = copy.deepcopy(test.workflow["applications"][:1])
    resolve_nested_configuration_files(applications, test.replacements)
    config = applications[0]["configuration"].get("backend_config") if applications else None
    if not isinstance(config, dict) or not config.get("container_image"):
        raise ValueError(
            f"Preflight requires a backend_config with container_image: {test.result['id']}"
        )
    config["priority"] = PREFLIGHT_PRIORITY
    paths = {key: str(path) for key, path in context_paths(context).items() if path.is_dir()}
    paths |= _environment_paths(config.get("environment_file"))
    checks = [f"check_path {shlex.quote(key)} {shlex.quote(path)}" for key, path in paths.items()]
    command = ("sh", "-c", "\n".join(["set -eu", _CHECK_PATH_FUNCTION, *checks]))
    options = ExecutionOptions(backend="htcondor", work_dir=Path(work_dir), backend_config=config)
    submission = submit_jobs([JobSpec("science-preflight", 0, command=command)], options)
    wait_for_submission(submission, options=options)


def _environment_paths(environment_file):
    """Return absolute ``*_PATH`` entries of a job environment file."""
    if not environment_file:
        return {}
    paths = {}
    for line in Path(environment_file).read_text(encoding="utf-8").splitlines():
        key, separator, value = line.strip().removeprefix("export ").partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if separator and key.endswith("_PATH") and value.startswith("/") and "$" not in value:
            paths[key] = value
    return paths
