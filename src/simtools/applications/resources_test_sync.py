#!/usr/bin/env python3

r"""Compare, sync, and prune test resources from a versioned simtools-tests bundle.

.. simtools-cli-help::
   :module: simtools.applications.resources_test_sync

Example
-------

Generate a list of new, changed, unchanged, and obsolete files in the ``tests/resources``
directory (dry run, no files are copied or deleted):

.. code-block:: console

    simtools-resources-test-sync \\
        --test_directory ../simtools-tests \\
        --simtools_version v0.34.0

To sync the test resources, add the ``--sync`` option. To delete obsolete
files, add the ``--delete_missing`` option.
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
        "help": "Delete obsolete test resources.",
    },
    "resources_path": {
        "type": Path,
        "default": Path("tests/resources"),
        "help": "Destination test-resources directory.",
    },
    "exclude_static": {
        "action": "store_false",
        "dest": "include_static",
        "help": "Skip static resources.",
    },
    "exclude_generated": {
        "action": "store_false",
        "dest": "include_generated",
        "help": "Skip generated resources.",
    },
    "exclude_downloaded": {
        "action": "store_false",
        "dest": "include_downloaded",
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
