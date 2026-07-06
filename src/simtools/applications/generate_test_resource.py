#!/usr/bin/env python3

"""Generate and check versioned integration-test resources for a simtools release.

The application reads download and workflow configurations from the release-specific
``integration_tests/config_files`` directory in the ``simtools-tests`` repository. External
inputs are downloaded first, followed by execution of all configured resource-generation
workflows. Intermediate output is written below ``tmp_application_output``; retained resources
and logs are collected below ``generated`` and ``log_files``, respectively. The application also
checks existing static files against their configured checksums.

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
config_file (path, optional)
    Run only the selected ``*.config.yml`` workflow from the release-specific
    ``integration_tests/config_files`` directory.
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

Runtime environment file example
--------------------------------

.. code-block:: yaml

        runtime_environment:
            container_engine: podman
            image: ghcr.io/gammasim/simtools-prod:20260622-v78010-v2025-11-30-rc-generic
            network: simtools-mongo-network
            environment_file: .env
            options:
                - "--arch amd64"
                - "-v /path/to/simpipe:/workdir/external/simpipe:ro"
"""

from pathlib import Path

from simtools.application_control import build_application
from simtools.testing import resource_generation


def _add_arguments(parser):
    """Register application-specific command-line arguments."""
    parser.add_argument("--test_directory", type=Path, required=True)
    parser.add_argument("--simtools_version", required=True)
    parser.add_argument("--download_only", action="store_true")
    parser.add_argument("--test_static_files", action="store_true")
    parser.add_argument(
        "--config_file",
        type=Path,
        help="Run only the selected workflow config file from integration_tests/config_files.",
    )


def main():
    """See CLI description."""
    app_context = build_application(
        initialization_kwargs={"db_config": False, "paths": False},
        startup_kwargs={"setup_io_handler": False, "resolve_sim_software_executables": False},
    )

    resource_generation.generate_test_resources(app_context.args, run_time=app_context.run_time)


if __name__ == "__main__":
    main()
