#!/usr/bin/env python3

r"""Generate and check versioned integration-test resources for a simtools release."""

from pathlib import Path

from simtools.application_control import build_application
from simtools.testing import resource_generation

_APPLICATION_ARG_DEFINITIONS = {
    "test_directory": {
        "type": Path,
        "required": True,
        "help": "Path to the simtools-tests repository.",
    },
    "simtools_version": {
        "required": True,
        "help": "Version of simtools to generate resources for.",
    },
    "download_only": {
        "action": "store_true",
        "help": "Only download resources, do not generate new ones.",
    },
    "test_static_files": {
        "action": "store_true",
        "help": "Test static files in the simtools-tests repository.",
    },
    "config_file": {
        "type": Path,
        "help": "Run only the selected workflow config file from integration_tests/config_files.",
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

    resource_generation.generate_test_resources(app_context.args, run_time=app_context.run_time)


if __name__ == "__main__":
    main()
