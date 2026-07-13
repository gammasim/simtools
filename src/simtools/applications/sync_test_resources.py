#!/usr/bin/env python3

"""Compare, sync, and prune test resources from a versioned simtools-tests bundle.

Command line arguments
----------------------
test_directory (path, required)
    Root directory of the ``simtools-tests`` repository.
simtools_version (str, required)
    Version directory to compare, for example ``v0.35.0``.
sync (flag, optional)
    Copy new and changed files into ``tests/resources``.
delete_missing (flag, optional)
    Remove files from ``tests/resources`` that are not present in the selected
    versioned resource set.
include_static (flag, optional)
    Compare and sync ``static`` resources.
include_generated (flag, optional)
    Compare and sync ``generated`` resources.

Example
-------

Generate a list of new, changed, unchanged, and obsolete files in the ``tests/resources``
directory (dry run, no files are copied or deleted):

.. code-block:: console

    simtools-sync-test-resources \
        --test_directory ../simtools-tests \
        --simtools_version v0.35.0
"""

from pathlib import Path

from simtools.application_control import build_application
from simtools.testing import resource_sync


def _add_arguments(parser):
    """Register application-specific command-line arguments."""
    parser.add_argument("--test_directory", "--test-directory", type=Path, required=True)
    parser.add_argument("--simtools_version", "--simtools-version", required=True)
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--delete_missing", "--delete-missing", action="store_true")
    parser.add_argument(
        "--include_static",
        "--include-static",
        action="store_true",
        default=True,
        help="Include static resources (default: enabled).",
    )
    parser.add_argument(
        "--include_generated",
        "--include-generated",
        action="store_true",
        default=True,
        help="Include generated resources (default: enabled).",
    )
    parser.add_argument(
        "--exclude_static",
        "--exclude-static",
        action="store_false",
        dest="include_static",
        help="Skip static resources.",
    )
    parser.add_argument(
        "--exclude_generated",
        "--exclude-generated",
        action="store_false",
        dest="include_generated",
        help="Skip generated resources.",
    )


def main():
    """See CLI description."""
    app_context = build_application(
        initialization_kwargs={"db_config": False, "paths": False},
        startup_kwargs={"setup_io_handler": False, "resolve_sim_software_executables": False},
    )

    resource_sync.sync_test_resources(app_context.args)


if __name__ == "__main__":
    main()
