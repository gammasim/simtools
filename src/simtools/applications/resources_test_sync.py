#!/usr/bin/env python3

r"""Compare, sync, and prune test resources from a versioned simtools-tests bundle."""

from pathlib import Path

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.testing import resource_sync

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "test_directory",
        type=Path,
        required=True,
        help="Path to the simtools-tests bundle.",
    ),
    cli.ArgumentDefinition(
        "simtools_version",
        required=True,
        help="Version of the simtools-tests bundle.",
    ),
    cli.ArgumentDefinition(
        "sync",
        action="store_true",
        help="Sync test resources from the simtools-tests bundle.",
    ),
    cli.ArgumentDefinition(
        "delete_missing",
        action="store_true",
        help="List obsolete test resources that should be removed manually.",
    ),
    cli.ArgumentDefinition(
        "resources_path",
        type=Path,
        default=Path("tests/resources"),
        help="Destination test-resources directory.",
    ),
    cli.ArgumentDefinition(
        "exclude_static", action="store_true", default=False, help="Skip static resources."
    ),
    cli.ArgumentDefinition(
        "exclude_generated", action="store_true", default=False, help="Skip generated resources."
    ),
    cli.ArgumentDefinition(
        "exclude_downloaded", action="store_true", default=False, help="Skip downloaded resources."
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(*_ARGUMENTS,),
    setup_io_handler=False,
    resolve_sim_software_executables=False,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

    resource_sync.sync_test_resources(app_context.args)


if __name__ == "__main__":
    main()
