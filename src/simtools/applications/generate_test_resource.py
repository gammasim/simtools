#!/usr/bin/env python3

"""Generate versioned integration-test resources for a simtools release.

The application reads download and workflow configurations from the release-specific
``integration_tests/config_files`` directory in the ``simtools-tests`` repository. External
inputs are downloaded first, followed by execution of all configured resource-generation
workflows. Intermediate output is written below ``tmp_application_output``; retained resources
and logs are collected below ``generated`` and ``log_files``, respectively.

Command line arguments
----------------------
test_directory (path, required)
    Root directory of the ``simtools-tests`` repository.
simtools_version (str, required)
    Version directory to generate, for example ``v0.32.0``.
download_only (flag, optional)
    Download external inputs without executing workflows.
test_static_files (flag, optional)
    Validate static files against their checksums and exit.
runtime_environment_file (path, optional)
    Standalone runtime-environment YAML reused for all workflows.
ignore_runtime_environment (flag, optional)
    Run applications in the current environment.
overwrite_collection_files (flag, optional)
    Allow collected resources to overwrite existing files.

Example
-------

.. code-block:: console

    simtools-generate-test-resources \
        --test_directory ../simtools-tests \
        --simtools_version v0.32.0
"""

import argparse
from pathlib import Path

from simtools.application_control import build_application
from simtools.testing import resource_generation


def _add_arguments(parser):
    """Register application-specific command-line arguments."""
    parser.add_argument("--test_directory", type=Path, required=True)
    parser.add_argument("--simtools_version", required=True)
    parser.add_argument("--download_only", action="store_true")
    parser.add_argument("--test_static_files", action="store_true")
    parser.add_argument("--runtime_environment_file", type=Path)
    parser.add_argument(
        "--ignore_runtime_environment",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Ignore runtime environments configured in application files.",
    )
    parser.add_argument(
        "--overwrite_collection_files",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow collected files to overwrite existing files with identical names.",
    )


def main():
    """See CLI description."""
    app_context = build_application(
        initialization_kwargs={"db_config": False, "paths": False},
        startup_kwargs={"setup_io_handler": False, "resolve_sim_software_executables": False},
    )
    resource_generation.generate_test_resources(
        test_directory=app_context.args["test_directory"],
        simtools_version=app_context.args["simtools_version"],
        download_only=app_context.args["download_only"],
        test_static_files=app_context.args["test_static_files"],
        runtime_environment_file=app_context.args["runtime_environment_file"],
        ignore_runtime_environment=app_context.args["ignore_runtime_environment"],
        overwrite_collection_files=app_context.args["overwrite_collection_files"],
    )


if __name__ == "__main__":
    main()
