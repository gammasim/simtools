"""Check the actual production runtime before submitting jobs to unchanged images."""

import hashlib
import inspect
import json
import logging
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from simtools import __version__, dependency_versions
from simtools.job_execution.execution import execute_jobs
from simtools.model_repository import compatibility as models
from simtools.model_repository.compatibility import (
    _add_file_digest,
    _contract,
    _tree_digest,
    _validate,
)

OUTPUT_OPTION = "--output_path"
IMAGE_KEY = "container_image"
PYTHON_COMMAND = "python"
CATALOG_KEY = "catalog"
SELECTION_KEY = "selection"
FILESYSTEM_SELECTION = "explicit filesystem path"
ENCODING = "utf-8"
OUTPUT_KEY = "output_path"

logger = logging.getLogger(__name__)
RECORD_NAME = "dependency-compatibility.json"
TARGET_KEY = "target_simtools"
IMAGE_DIGEST_KEY = "image_sha256"
INITIALIZATION_KEY = "initialization"
PATH_KEY = "simulation_models_path"
GIT_PATH_KEY = "simulation_models_git_path"
GIT_REVISION_KEY = "simulation_models_git_revision"
PROOF_KEY = "dependency_compatibility"


def _bundled_catalog_path():
    """Locate a runtime's own catalog without consulting its working directory."""
    filename = dependency_versions.DEPENDENCY_VERSIONS_FILENAME
    candidates = [
        Path(dependency_versions.__file__).resolve().parents[2] / filename,
        Path(sys.prefix) / "simtools" / filename,
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise RuntimeError("Target installation has no bundled dependency catalog.")


BUNDLED_CATALOG_CODE = (
    "import os, sys\nfrom pathlib import Path\nfrom simtools import dependency_versions\n"
    + inspect.getsource(_bundled_catalog_path)
    + "\ncatalog_path = _bundled_catalog_path()\n"
    + "os.environ['SIMTOOLS_DEPENDENCY_VERSIONS'] = str(catalog_path)\n"
)
INFO_CODE = (
    BUNDLED_CATALOG_CODE
    + """
import json
from simtools import __version__
print(json.dumps({'target_simtools': __version__, 'catalog': dependency_versions.load_dependency_catalog(catalog_path, validate=False)}))
"""
)
INITIALIZE_CODE = """
import os, sys
from simtools.applications.simulate_prod import APPLICATION
from simtools.simulator import Simulator
sys.argv = ['simulate_prod', *sys.argv[1:]]
context = APPLICATION.start()
Simulator(label=context.args.get('label'))
"""


def _image_digest(image):
    with Path(image).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _target(image=None):
    """Inspect the target catalog independently of submit-host overrides."""
    if not image:
        catalog = dependency_versions.load_dependency_catalog(_bundled_catalog_path())
        return {TARGET_KEY: __version__, CATALOG_KEY: catalog, IMAGE_DIGEST_KEY: None}
    image = str(Path(image).expanduser().resolve(strict=True))
    image_digest = _image_digest(image)
    result = subprocess.run(
        ["apptainer", "exec", "--cleanenv", image, PYTHON_COMMAND, "-P", "-c", INFO_CODE],
        check=True,
        capture_output=True,
        text=True,
        env={
            key: value
            for key, value in os.environ.items()
            if not key.endswith(("PYTHONPATH", "PYTHONHOME", "SIMTOOLS_DEPENDENCY_VERSIONS"))
        },
    )
    info = json.loads(result.stdout)
    if _image_digest(image) != image_digest:
        raise ValueError(
            "Target image changed during inspection; no production jobs were submitted."
        )
    info[IMAGE_DIGEST_KEY] = image_digest
    return info


def _preview(args, image=None):
    """Resolve and report compatibility without submitting or initializing jobs."""
    target = _target(image)
    path = args.get(PATH_KEY) or args.get(GIT_PATH_KEY)
    if not path:
        raise ValueError("Production requires an explicit simulation-model source path.")
    revision = None
    if not args.get(PATH_KEY):
        revision = args.get(GIT_REVISION_KEY) or _contract(target[CATALOG_KEY])[models.DEFAULT_KEY]
    proof = _validate(
        path, target[CATALOG_KEY], revision=revision, development=bool(args.get("test"))
    )
    proof.update(
        {
            TARGET_KEY: target[TARGET_KEY],
            IMAGE_DIGEST_KEY: target[IMAGE_DIGEST_KEY],
            "host_simtools": __version__,
            SELECTION_KEY: FILESYSTEM_SELECTION if args.get(PATH_KEY) else "explicit Git source",
            INITIALIZATION_KEY: "not checked",
        }
    )
    logger.info(
        "Dependency compatibility: target simtools %s (host %s), image SHA256 %s; "
        "models %s at %s; allowed %s (%s); %s; initialization not checked",
        proof[TARGET_KEY],
        __version__,
        proof[IMAGE_DIGEST_KEY],
        proof.get(models.VERSION_KEY, "unknown"),
        proof[models.SOURCE_KEY],
        proof[models.ALLOWED_KEY],
        proof[models.POLICY_KEY],
        proof[models.STATUS_KEY],
    )
    return proof


def _guard_code(proof):
    """Ship a stdlib-only identity check that old images can execute unchanged."""
    path = proof[models.SOURCE_KEY]
    imports = "import hashlib, os, subprocess, sys\nfrom pathlib import Path\n"
    definitions = (
        f"MODELS_DIRECTORY = {models.MODELS_DIRECTORY!r}\n"
        + inspect.getsource(_add_file_digest)
        + "\n"
        + inspect.getsource(_tree_digest)
    )
    check = (
        f"\nif _tree_digest({path!r}) != {proof[models.DIGEST_KEY]!r}:\n"
        "    raise RuntimeError('Model inputs changed since pre-submission validation.')\n"
    )
    return imports + definitions + check


def _representatives(jobs):
    """Initialize each distinct simulation configuration, excluding run bookkeeping."""
    seen = set()
    ignored = {
        OUTPUT_OPTION,
        "--run_number",
        "--run_number_offset",
        "--label",
        "--grid_output_path",
        "--energy_range",
        "--showers_per_run",
        "--core_scatter",
        "--view_cone",
        "--azimuth_angle",
    }
    for job in jobs:
        arguments = []
        skip = False
        for value in job.command[3:]:
            if skip:
                skip = False
            elif value in ignored:
                skip = True
            else:
                arguments.append(value)
        key = tuple(arguments)
        if key not in seen:
            seen.add(key)
            yield job


def _initialization_job(job, index, directory, proof):
    arguments = list(_pin_git_command(job.command, proof)[3:])
    arguments[arguments.index(OUTPUT_OPTION) + 1] = str(directory / f"configuration-{index}")
    code = BUNDLED_CATALOG_CODE + "\n" + INITIALIZE_CODE
    if proof.get(models.DIGEST_KEY) and proof[SELECTION_KEY] == FILESYSTEM_SELECTION:
        code = _guard_code(proof) + "\n" + code
    return replace(
        job,
        job_id=f"compatibility-{index:06d}",
        index=index,
        command=(sys.executable, "-c", code, *arguments),
        output_paths=(),
        mount_paths=(
            *(path for path in job.mount_paths if path not in job.output_paths),
            directory,
        ),
    )


def _prepare_jobs(jobs, args, options):
    """Gate a campaign on target compatibility and initialization; seal its inputs."""
    if not jobs:
        return jobs
    image = options.backend_config.get(IMAGE_KEY)
    if any(job.resources.get(IMAGE_KEY, image) != image for job in jobs):
        raise ValueError("Validate separate production campaigns for different target images.")
    proof = _preview(args, image)
    output_root = Path(args[OUTPUT_KEY])
    directory = output_root.parent / "work" / f"{output_root.name}-dependency-validation"
    directory.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    canaries = [
        _initialization_job(job, index, directory, proof)
        for index, job in enumerate(_representatives(jobs))
    ]
    canary_options = replace(options, work_dir=directory)
    record = Path(args[OUTPUT_KEY]) / RECORD_NAME
    record.write_text(json.dumps(proof, indent=2) + "\n", encoding=ENCODING)
    logger.info(
        "Checking dependency initialization for %d configurations before production submission",
        len(canaries),
    )
    try:
        execute_jobs(canaries, canary_options)
    except Exception as exc:
        proof[INITIALIZATION_KEY] = "failed"
        record.write_text(json.dumps(proof, indent=2) + "\n", encoding=ENCODING)
        raise RuntimeError(
            f"Dependency initialization failed; no production jobs were submitted. See {record}."
        ) from exc
    proof[INITIALIZATION_KEY] = "passed"
    if image and _image_digest(image) != proof[IMAGE_DIGEST_KEY]:
        raise ValueError("Target image changed during compatibility validation.")
    if (
        args.get(PATH_KEY)
        and proof.get(models.DIGEST_KEY)
        and _tree_digest(proof[models.SOURCE_KEY]) != proof[models.DIGEST_KEY]
    ):
        raise ValueError("Model inputs changed during compatibility validation.")
    record.write_text(json.dumps(proof, indent=2) + "\n", encoding=ENCODING)
    logger.info("Dependency initialization passed for %d configurations; %s", len(canaries), record)
    return [_sealed_job(job, proof) for job in jobs]


def _sealed_job(job, proof):
    metadata = {**job.metadata, PROOF_KEY: proof}
    if proof[SELECTION_KEY] != FILESYSTEM_SELECTION or not proof.get(models.DIGEST_KEY):
        return replace(job, command=_pin_git_command(job.command, proof), metadata=metadata)
    code = (
        BUNDLED_CATALOG_CODE
        + "\n"
        + _guard_code(proof)
        + "\nsubprocess.run(sys.argv[1:], check=True)\n"
    )
    # Old images can execute the guard without importing any new simtools modules.
    nested = (PYTHON_COMMAND, *job.command[1:])
    return replace(job, command=(sys.executable, "-c", code, *nested), metadata=metadata)


def _pin_git_command(command, proof):
    """Keep a validated Git source on the observed commit, even if its tag moves."""
    if proof[SELECTION_KEY] == FILESYSTEM_SELECTION or not proof.get(models.COMMIT_KEY):
        return command
    arguments = list(command)
    option = "--simulation_models_git_revision"
    if option in arguments:
        arguments[arguments.index(option) + 1] = proof[models.COMMIT_KEY]
    else:
        arguments.extend([option, proof[models.COMMIT_KEY]])
    return tuple(arguments)
