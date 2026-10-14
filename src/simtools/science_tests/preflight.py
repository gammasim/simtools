"""Pre-submission checks for science-test production runs."""

import shlex
from pathlib import Path

import yaml

from simtools.job_execution.execution import submit_jobs, wait_for_submission
from simtools.job_execution.job import ExecutionOptions, JobSpec
from simtools.utils.general import replace_placeholders_recursively

_DIRECTORY_CONTEXT_PATHS = (
    "__SCIENCE_CANDIDATE_PATH__",
    "__SCIENCE_BASELINE_PATH__",
    "__PRODUCTION_CONFIGURATION_PATH__",
    "__CORSIKA_INTERACTION_TABLE_PATH__",
    "simulation_models_path",
    "simulation_models_git_path",
)
_CONTAINER_DIRECTORY_PATHS = (
    "/workdir/simulation_software/corsika7",
    "/workdir/simulation_software/sim_telarray",
)
PREFLIGHT_PRIORITY = 1_000_000


def validate_context_paths(context):
    """Validate configured host paths before submitting any worker jobs."""
    errors = []
    for key in _DIRECTORY_CONTEXT_PATHS:
        value = context.get(key)
        if not value:
            continue
        path = Path(value).expanduser()
        if not path.is_dir():
            errors.append(f"{key} directory does not exist: {path}")
        elif not any(item.is_file() and item.stat().st_size > 0 for item in path.rglob("*")):
            errors.append(f"{key} directory contains no non-empty files: {path}")
    image = context.get("__SCIENCE_CONTAINER_IMAGE_PATH__")
    if image:
        image_path = Path(image).expanduser()
        if not image_path.is_file() or image_path.stat().st_size == 0:
            errors.append(f"__SCIENCE_CONTAINER_IMAGE_PATH__ is not a non-empty file: {image_path}")
    if errors:
        raise ValueError("Science-test preflight failed:\n- " + "\n- ".join(errors))


def _canary_config(template_dir, workflow, context):
    """Resolve the HTCondor configuration used by a production workflow."""
    workflow_path = Path(template_dir) / workflow
    values = {key: value for key, value in context.items() if value is not None}
    values["__CONFIG_DIRECTORY__"] = str(workflow_path.parent.resolve())
    resolved = replace_placeholders_recursively(
        yaml.safe_load(workflow_path.read_text(encoding="utf-8")), values
    )
    application = resolved["applications"][0]["configuration"]
    config = application.get("backend_config")
    if not isinstance(config, dict):
        raise ValueError(f"Preflight requires inline backend_config in {workflow_path}")
    return config


def run_worker_canary(template_dir, workflow, context, work_dir):
    """Run one worker-side filesystem check with the production container setup."""
    config = _canary_config(template_dir, workflow, context)
    config["priority"] = PREFLIGHT_PRIORITY
    image = config.get("container_image")
    if not image:
        raise ValueError(f"Preflight workflow has no container image: {workflow}")
    checks = [
        "check_path() {",
        "  name=$1; path=$2",
        '  if ! test -d "$path" || ! test -r "$path" || ',
        '! find -L "$path" -type f -size +0c -print -quit | grep -q .; then',
        '    echo "PREFLIGHT_FAILED: $name ($path)" >&2',
        "    exit 1",
        "  fi",
        '  echo "PREFLIGHT_OK: $name ($path)"',
        "}",
    ]
    for key in _DIRECTORY_CONTEXT_PATHS:
        value = context.get(key)
        if value:
            checks.append(f"check_path {shlex.quote(key)} {shlex.quote(str(value))}")
    for path in _CONTAINER_DIRECTORY_PATHS:
        checks.append(f"check_path container:{shlex.quote(path)} {shlex.quote(path)}")
    command = ("sh", "-c", "set -eu\n" + "\n".join(checks))
    options = ExecutionOptions(
        backend="htcondor",
        work_dir=Path(work_dir),
        backend_config=config,
    )
    submission = submit_jobs(
        [JobSpec("science-preflight", 0, command=command)],
        options,
    )
    wait_for_submission(submission, options=options)
