"""Generate and collect versioned resources used by simtools integration tests."""

import hashlib
import logging
import urllib.error
import urllib.request
from pathlib import Path

import simtools.utils.general as gen
from simtools.io import ascii_handler
from simtools.runners import simtools_runner

logger = logging.getLogger(__name__)
STATIC_MANIFEST = "static_manifest.yml"
_PRESERVED_INTEGRATION_TEST_DIRECTORIES = {
    "config_files",
    "generated",
    "log_files",
    "static",
    "tmp_application_output",
}


def get_integration_test_directory(test_directory, simtools_version):
    """Return the integration-test directory for a simtools release.

    Parameters
    ----------
    test_directory : str or pathlib.Path
        Root directory of the ``simtools-tests`` repository (or a directory containing it).
    simtools_version : str
        Version directory to use, for example ``v0.32.0``.

    Returns
    -------
    pathlib.Path
        Path to the release-specific ``integration_tests`` directory.
    """
    test_directory = Path(test_directory)
    repo_root = (
        test_directory / "simtools-tests"
        if (test_directory / "simtools-tests").is_dir()
        else test_directory
    )
    return repo_root / simtools_version / "integration_tests"

def get_resource_generation_directory(test_directory, simtools_version):
    """Return the configuration directory for resource generation.

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
        If the configuration directory does not exist.
    """
    config_dir = get_integration_test_directory(test_directory, simtools_version) / "config_files"
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
        args_dict = {
            "config_file": str(config_file),
            "log_file": str(log_dir / f"{config_file.name.removesuffix('.config.yml')}.log"),
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
        raise ValueError(f"Download entry {index} missing required keys: {', '.join(missing_keys)}")

    target_path = Path(entry["target_path"])
    if target_path.is_absolute() or ".." in target_path.parts:
        raise ValueError(f"Download entry {index} has invalid target_path: {target_path.as_posix()}")

def download_files(config_file, target_dir):
    """Download external files listed in a resource-generation configuration.

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

    Returns
    -------
    list[pathlib.Path]
        Downloaded file paths.
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

    downloaded_files = []
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
        downloaded_files.append(destination)
    return downloaded_files


def _remove_empty_download_directories(downloaded_files, target_dir):
    """Remove empty nonstandard top-level directories created for downloads."""
    target_dir = Path(target_dir).resolve()
    candidate_directories = set()
    for downloaded_file in downloaded_files or []:
        try:
            relative_path = Path(downloaded_file).resolve().relative_to(target_dir)
        except ValueError:
            continue
        if len(relative_path.parts) > 1:
            candidate_directories.add(target_dir / relative_path.parts[0])

    for directory in sorted(candidate_directories):
        if directory.name in _PRESERVED_INTEGRATION_TEST_DIRECTORIES:
            continue
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
            logger.info("Removed empty download directory %s", directory)


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


def _calculate_sha256(file_path):
    """Return the SHA-256 checksum of a file."""
    checksum = hashlib.sha256()
    with Path(file_path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def _validate_static_manifest_entry(entry, index, static_dir, declared_files):
    """Validate one static-file manifest entry and return discovered errors."""
    if not isinstance(entry, dict) or not entry.get("file_name") or not entry.get("sha256"):
        return [f"Manifest entry {index} requires file_name and sha256 values."]

    relative_path = Path(entry["file_name"])
    file_name = relative_path.as_posix()
    if relative_path.is_absolute() or ".." in relative_path.parts:
        return [f"Invalid manifest path: {file_name}"]
    if file_name in declared_files:
        return [f"Duplicate manifest entry: {file_name}"]

    declared_files.add(file_name)
    static_file = static_dir / relative_path
    if not static_file.is_file():
        return [f"Missing static file: {file_name}"]
    if _calculate_sha256(static_file) != str(entry["sha256"]):
        return [f"Checksum mismatch: {file_name}"]
    return []


def validate_static_files(manifest_file):
    """Validate static integration-test files against their manifest.

    Parameters
    ----------
    manifest_file : str or pathlib.Path
        YAML manifest containing relative ``file_name`` and ``sha256`` values.

    Raises
    ------
    FileNotFoundError
        If the manifest does not exist.
    ValueError
        If entries are invalid, files are missing or unlisted, or checksums differ.
    """
    manifest_file = Path(manifest_file)
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Static-file manifest does not exist: {manifest_file}")

    manifest = ascii_handler.collect_data_from_file(manifest_file)
    entries = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(entries, list):
        raise ValueError("Static-file manifest key 'files' must be a list.")

    static_dir = manifest_file.parent
    declared_files = set()
    errors = []
    for index, entry in enumerate(entries):
        errors.extend(_validate_static_manifest_entry(entry, index, static_dir, declared_files))

    actual_files = {
        path.relative_to(static_dir).as_posix()
        for path in static_dir.rglob("*")
        if path.is_file() and path != manifest_file
    }
    errors.extend(
        f"File not listed in manifest: {name}" for name in sorted(actual_files - declared_files)
    )

    if errors:
        raise ValueError("Static-file validation failed:\n- " + "\n- ".join(errors))
    logger.info("Validated %d static files using %s", len(declared_files), manifest_file)


def generate_test_resources(
    test_directory,
    simtools_version,
    download_only=False,
    test_static_files=False,
    runtime_environment_file=None,
    ignore_runtime_environment=None,
    overwrite_collection_files=False,
):
    """Download inputs and run resource-generation workflows for a release.

    Parameters
    ----------
    test_directory : str or pathlib.Path
        Root directory of the ``simtools-tests`` repository.
    simtools_version : str
        Version directory to generate, for example ``v0.32.0``.
    download_only : bool, optional
        Download external files without running workflows.
    test_static_files : bool, optional
        Validate static files against their manifest without downloading or running workflows.
    runtime_environment_file : str or pathlib.Path or None, optional
        Standalone runtime-environment YAML reused for all workflows.
    ignore_runtime_environment : bool or None, optional
        Run in the current environment. By default, this is true when no standalone runtime is
        provided and false when one is provided.
    overwrite_collection_files : bool, optional
        Allow collected files to overwrite existing files.
    """
    integration_test_dir = get_integration_test_directory(test_directory, simtools_version)
    if test_static_files:
        validate_static_files(integration_test_dir / "static" / STATIC_MANIFEST)
        return

    config_dir = get_resource_generation_directory(test_directory, simtools_version)
    replacements = {
        "__TEST_DIRECTORY__": str(test_directory),
        "__SIMTOOLS_VERSION__": simtools_version,
    }
    downloaded_files = download_files(config_dir / "download_files.yml", integration_test_dir)
    if download_only:
        return

    if ignore_runtime_environment is None:
        ignore_runtime_environment = runtime_environment_file is None

    runtime_environment = None
    run_time = None
    if runtime_environment_file is not None and not ignore_runtime_environment:
        runtime_environment, run_time = prepare_runtime_environment(runtime_environment_file)

    try:
        run_configured_applications(
            config_dir=config_dir,
            log_dir=integration_test_dir / "log_files",
            ignore_runtime_environment=ignore_runtime_environment,
            overwrite_collection_files=overwrite_collection_files,
            run_time=run_time,
            runtime_environment=runtime_environment,
            replacements=replacements,
        )
    finally:
        _remove_empty_download_directories(downloaded_files, integration_test_dir)
