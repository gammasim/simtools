#!/usr/bin/env python3

"""Generate versioned integration-test resources for a simtools release.

The application reads downloads and workflow configurations from
``<test_directory>/simtools-tests/<simtools_version>/integration_tests/config_files``.
It downloads external inputs and then runs every ``*.config.yml`` workflow in that directory.
Workflow configurations define which results are retained under ``generated``; intermediate
application output is written below ``tmp_application_output``, and workflow logs are written
below ``log_files``.

A standalone runtime-environment configuration can be prepared once and reused by all workflows.
Use ``--download_only`` to download external inputs without executing the workflows.

Examples
--------
Generate resources for simtools v0.32.0::

    python src/simtools/applications/generate_test_resource.py \
        --test_directory ../simtools-tests --simtools_version v0.32.0
"""

import argparse
import logging
import urllib.error
import urllib.request
from pathlib import Path

import simtools.utils.general as gen
from simtools.io import ascii_handler
from simtools.runners import simtools_runner

logger = logging.getLogger(__name__)


def get_integration_test_directory(test_directory, simtools_version):
    """Return the integration-test directory for a simtools release.

    Parameters
    ----------
    test_directory : str or pathlib.Path
        Root directory of the ``simtools-tests`` repository.
    simtools_version : str
        Version directory to use, for example ``v0.32.0``.

    Returns
    -------
    pathlib.Path
        Path to the release-specific ``integration_tests`` directory.
    """
    return Path(test_directory) / "simtools-tests" / simtools_version / "integration_tests"


def get_resource_generation_directory(test_directory, simtools_version):
    """Return the resource-generation directory for a simtools release.

    Parameters
    ----------
    test_directory : str or pathlib.Path
        Root directory of the ``simtools-tests`` repository.
    simtools_version : str
        Version directory to use, for example ``v0.32.0``.

    Returns
    -------
    pathlib.Path
        Existing release-specific ``config_files`` directory.

    Raises
    ------
    FileNotFoundError
        If the resource-generation directory does not exist.
    """
    config_dir = get_integration_test_directory(test_directory, simtools_version) / ("config_files")
    if not config_dir.is_dir():
        raise FileNotFoundError(f"Resource-generation directory does not exist: {config_dir}")
    return config_dir


def run_configured_applications(
    config_dir,
    log_dir,
    ignore_runtime_environment=True,
    overwrite_collection_files=False,
    run_time=None,
    runtime_environment=None,
    replacements=None,
):
    """Run all resource-generation workflows using one prepared runtime.

    Parameters
    ----------
    config_dir : str or pathlib.Path
        Directory containing the ``*.config.yml`` workflow files.
    log_dir : str or pathlib.Path
        Destination directory for one log file per workflow.
    ignore_runtime_environment : bool, optional
        Run applications in the current environment instead of a configured runtime.
    overwrite_collection_files : bool, optional
        Allow collection output files to overwrite existing files.
    run_time : list[str] or None, optional
        Prepared runtime command reused for every workflow.
    runtime_environment : dict or None, optional
        Runtime-environment configuration recorded in workflow metadata.
    replacements : dict[str, str] or None, optional
        Placeholders replaced recursively in each workflow configuration.
    """
    config_dir = Path(config_dir)
    log_dir = Path(log_dir)
    for config_file in sorted(config_dir.rglob("*.config.yml")):
        logger.info("Executing applications configured in %s", config_file)
        log_name = f"{config_file.name.removesuffix('.config.yml')}.log"
        args_dict = {
            "config_file": str(config_file),
            "log_file": str(log_dir / log_name),
            "steps": None,
            "ignore_runtime_environment": ignore_runtime_environment,
            "overwrite_collection_files": overwrite_collection_files,
        }
        if runtime_environment is not None:
            args_dict["runtime_environment"] = runtime_environment
        simtools_runner.run_applications(
            args_dict,
            run_time=run_time,
            replacements=replacements,
        )


def _validate_download_entry(entry, index):
    """Validate a download entry from YAML configuration."""
    if not isinstance(entry, dict):
        raise ValueError(f"Download entry {index} must be a dictionary.")

    required_keys = ("url", "description", "target_path")
    missing_keys = [key for key in required_keys if not entry.get(key)]
    if missing_keys:
        missing_keys_str = ", ".join(missing_keys)
        raise ValueError(f"Download entry {index} missing required keys: {missing_keys_str}")


def download_files(config_file, target_dir):
    """Download external files listed in a resource-generation configuration.

    GitLab version placeholders are derived from the top-level ``gitlab_versions`` mapping.

    Parameters
    ----------
    config_file : str or pathlib.Path
        YAML file containing ``gitlab_versions`` and ``files`` entries.
    target_dir : str or pathlib.Path
        Base directory for each entry's relative ``target_path``.

    Raises
    ------
    FileNotFoundError
        If the configuration or a remote file does not exist.
    ValueError
        If the configuration structure is invalid or a version placeholder is unresolved.
    """
    if not Path(config_file).is_file():
        raise FileNotFoundError(f"Download configuration file does not exist: {config_file}")

    download_config = ascii_handler.collect_data_from_file(config_file)
    if not isinstance(download_config, dict):
        raise ValueError("Download configuration must be a YAML mapping.")

    file_entries = download_config.get("files", [])
    gitlab_versions = download_config.get("gitlab_versions", {})

    if not isinstance(file_entries, list):
        raise ValueError("Download configuration key 'files' must be a list.")
    if not isinstance(gitlab_versions, dict):
        raise ValueError("Download configuration key 'gitlab_versions' must be a dictionary.")

    replacements = {
        f"__{repository.upper()}_VERSION__": str(version)
        for repository, version in gitlab_versions.items()
    }
    file_entries = gen.replace_placeholders_recursively(file_entries, replacements)

    for index, entry in enumerate(file_entries):
        _validate_download_entry(entry, index)

        url = entry["url"]
        if "_VERSION__" in url:
            raise ValueError(f"No GitLab version configured for download URL: {url}")

        destination = Path(target_dir) / entry["target_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Downloading %s from %s to %s", entry["description"], url, destination)
        try:
            urllib.request.urlretrieve(url, destination)
        except urllib.error.HTTPError as exc:
            exc.close()
            raise FileNotFoundError(
                f"Failed to download '{entry['description']}' from {url}"
            ) from exc


def prepare_runtime_environment(runtime_environment_file):
    """Read and prepare a standalone runtime environment.

    Parameters
    ----------
    runtime_environment_file : str or pathlib.Path
        YAML file containing a top-level ``runtime_environment`` mapping.

    Returns
    -------
    tuple[dict, list[str]]
        Runtime-environment configuration and prepared runtime command.

    Raises
    ------
    ValueError
        If the YAML content is not a mapping or lacks ``runtime_environment``.
    """
    runtime_config = ascii_handler.collect_data_from_file(runtime_environment_file)
    if not isinstance(runtime_config, dict):
        raise ValueError("Runtime configuration must be a YAML mapping.")
    runtime_environment = runtime_config.get("runtime_environment")
    if runtime_environment is None:
        raise ValueError("Runtime configuration must contain a 'runtime_environment' block.")
    return runtime_environment, simtools_runner.read_runtime_environment(runtime_environment)


def parse_args():
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate release-specific simtools integration-test resources."
    )
    parser.add_argument("--test_directory", type=Path, required=True)
    parser.add_argument("--simtools_version", required=True)
    parser.add_argument("--download_only", action="store_true")
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
        help=(
            "Allow files copied by run_application collection blocks to overwrite existing "
            "files with identical names."
        ),
    )
    return parser.parse_args()


def main():
    """Download inputs and generate release-specific integration-test resources."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", force=True)
    args = parse_args()

    config_dir = get_resource_generation_directory(args.test_directory, args.simtools_version)
    integration_test_dir = config_dir.parent
    replacements = {
        "__TEST_DIRECTORY__": str(args.test_directory),
        "__SIMTOOLS_VERSION__": args.simtools_version,
    }
    download_files(
        config_file=config_dir / "download_files.yml",
        target_dir=integration_test_dir,
    )
    if args.download_only:
        return

    runtime_environment = None
    run_time = None
    ignore_runtime_environment = (
        args.ignore_runtime_environment
        if args.ignore_runtime_environment is not None
        else args.runtime_environment_file is None
    )
    if args.runtime_environment_file is not None and not ignore_runtime_environment:
        runtime_environment, run_time = prepare_runtime_environment(args.runtime_environment_file)
    run_configured_applications(
        config_dir=config_dir,
        log_dir=integration_test_dir / "log_files",
        ignore_runtime_environment=ignore_runtime_environment,
        overwrite_collection_files=args.overwrite_collection_files,
        run_time=run_time,
        runtime_environment=runtime_environment,
        replacements=replacements,
    )


if __name__ == "__main__":
    main()
