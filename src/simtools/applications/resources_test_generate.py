#!/usr/bin/env python3

r"""Generate and check versioned integration-test resources for a simtools release."""

from pathlib import Path

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.testing import resource_generation

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "test_directory",
        type=Path,
        required=True,
        help="Path to the simtools-tests repository.",
    ),
    cli.ArgumentDefinition(
        "simtools_version",
        required=True,
        help="Version of simtools to generate resources for.",
    ),
    cli.ArgumentDefinition(
        "download_only",
        action="store_true",
        help="Only download resources, do not generate new ones.",
    ),
    cli.ArgumentDefinition(
        "test_static_files",
        action="store_true",
        help="Test static files in the simtools-tests repository.",
    ),
    cli.ArgumentDefinition(
        "config_file",
        type=Path,
        help="Run only the selected workflow config file from integration_tests/config_files.",
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

    resource_generation.generate_test_resources(app_context.args, run_time=app_context.run_time)


if __name__ == "__main__":
    main()
