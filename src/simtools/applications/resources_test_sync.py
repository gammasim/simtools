#!/usr/bin/env python3

r"""Compare, sync, and prune test resources from a versioned simtools-tests bundle.

.. simtools-cli-help::
   :module: simtools.applications.resources_test_sync

Example
-------

Generate a dry-run report of new, changed, and obsolete files in the
``tests/resources`` directory (no files are copied or deleted):

.. code-block:: console

    simtools-resources-test-sync \\
        --test_directory ../simtools-tests \\
        --simtools_version v0.34.0

To sync the test resources, add the ``--sync`` option. To list obsolete
files that should be removed manually, add the ``--delete_missing`` option.
"""

from pathlib import Path

from simtools.application_control import build_application
from simtools.testing import resource_sync

_APPLICATION_ARG_DEFINITIONS = {
    "test_directory": {
        "type": Path,
        "required": True,
        "help": "Path to the simtools-tests bundle.",
    },
    "simtools_version": {
        "required": True,
        "help": "Version of the simtools-tests bundle.",
    },
    "sync": {
        "action": "store_true",
        "help": "Sync test resources from the simtools-tests bundle.",
    },
    "delete_missing": {
        "action": "store_true",
        "help": "List obsolete test resources that should be removed manually.",
    },
    "resources_path": {
        "type": Path,
        "default": Path("tests/resources"),
        "help": "Destination test-resources directory.",
    },
    "exclude_static": {
        "action": "store_true",
        "default": False,
        "help": "Skip static resources.",
    },
    "exclude_generated": {
        "action": "store_true",
        "default": False,
        "help": "Skip generated resources.",
    },
    "exclude_downloaded": {
        "action": "store_true",
        "default": False,
        "help": "Skip downloaded resources.",
    },
}


def main():
    """See CLI description."""
    app_context = build_application(
        application_argument_definitions=_APPLICATION_ARG_DEFINITIONS,
        initialization_kwargs={"db_config": False, "paths": False},
        startup_kwargs={
            "setup_io_handler": False,
            "resolve_sim_software_executables": False,
        },
    )

    resource_sync.sync_test_resources(app_context.args)


if __name__ == "__main__":
    main()
