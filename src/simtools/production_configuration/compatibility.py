"""Validate models against the actual production runtime before submission."""

import inspect
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from simtools import __version__, dependency_versions, version
from simtools.model_repository.compatibility import _contract, _validate

IMAGE_KEY = "container_image"

logger = logging.getLogger(__name__)
CATALOG_KEY = "catalog"
TARGET_KEY = "target_simtools"
PATH_KEY = "simulation_models_path"
GIT_PATH_KEY = "simulation_models_git_path"
GIT_REVISION_KEY = "simulation_models_git_revision"


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


def _preview(args, image=None):
    """Check selected models using the target installation's bundled catalog."""
    if image:
        result = subprocess.run(
            ["apptainer", "exec", "--cleanenv", str(image), "python", "-P", "-c", INFO_CODE],
            check=True,
            capture_output=True,
            text=True,
            env={
                key: value
                for key, value in os.environ.items()
                if not key.endswith(("PYTHONPATH", "PYTHONHOME", "SIMTOOLS_DEPENDENCY_VERSIONS"))
            },
        )
        target = json.loads(result.stdout)
    else:
        target = {
            TARGET_KEY: __version__,
            CATALOG_KEY: dependency_versions.load_dependency_catalog(_bundled_catalog_path()),
        }
    path = args.get(PATH_KEY) or args.get(GIT_PATH_KEY)
    if not path:
        raise ValueError("Production requires an explicit simulation-model source path.")
    revision = None
    if not args.get(PATH_KEY):
        revision = args.get(GIT_REVISION_KEY) or _contract(target[CATALOG_KEY])["default"]
    logger.info("Checking models for target simtools %s (host %s)", target[TARGET_KEY], __version__)
    return {
        **_validate(
            path,
            target[CATALOG_KEY],
            revision,
            version.is_development_build or bool(args.get("test")),
        ),
        TARGET_KEY: target[TARGET_KEY],
    }


def _prepare_jobs(jobs, args, options):
    """Reject incompatible models before submitting a production grid."""
    if jobs:
        image = options.backend_config.get(IMAGE_KEY)
        if any(job.resources.get(IMAGE_KEY, image) != image for job in jobs):
            raise ValueError("Submit separate production grids for different target images.")
        _preview(args, image)
    return jobs
