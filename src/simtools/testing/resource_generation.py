"""Generate and collect versioned resources used by simtools integration tests."""

import hashlib
import logging
import urllib.error
import urllib.request
from pathlib import Path

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


def run_configured_applications(args_dict, config_dir, log_dir, run_time, replacements):
    """Run all resource-generation workflows using one prepared runtime.

    Parameters
    ----------
    args_dict : dict
        Dictionary containing command line arguments.
    config_dir : str or pathlib.Path
        Directory containing the ``*.config.yml`` workflow files.
    log_dir : str or pathlib.Path
        Destination directory for one log file per workflow.
    run_time : list or None
        Prepared runtime command. If provided, reuse it instead of preparing
        the runtime environment from the workflow configuration.
    replacements : dict[str, str] or None
        Placeholders replaced recursively in each workflow configuration.
    """
    config_dir = Path(config_dir)
    log_dir = Path(log_dir)
    for workflow_config in _get_selected_config_files(config_dir, args_dict.get("config_file")):
        logger.info("Executing applications configured in %s", workflow_config)
        tmp_args_dict = {
            "config_file": str(workflow_config),
            "log_file": str(log_dir / f"{workflow_config.name.removesuffix('.config.yml')}.log"),
            "steps": None,
            "runtime_environment": args_dict.get("runtime_environment"),
            "ignore_runtime_environment": args_dict.get("ignore_runtime_environment", False),
            "overwrite_collection_files": args_dict.get("overwrite_collection_files", False),
        }
        simtools_runner.run_applications(
            tmp_args_dict, run_time=run_time, replacements=replacements
        )


def _get_selected_config_files(config_dir, config_file=None):
    """Return the workflow config files to execute."""
    config_dir = Path(config_dir)
    available_configs = sorted(config_dir.rglob("*.config.yml"))

    if config_file is None:
        return available_configs

    resolved_map = {path.resolve(): path for path in available_configs}
    requested_path = Path(config_file)
    for candidate in (requested_path, config_dir / requested_path):
        resolved_candidate = candidate.resolve()
        if resolved_candidate in resolved_map:
            return [resolved_map[resolved_candidate]]

    raise FileNotFoundError(
        f"Selected workflow config does not exist in {config_dir}: {requested_path}"
    )


def _construct_download_url(base_url_info, base_key, path):
    """Construct the full download URL from base URL info, base key, and path.

    Parameters
    ----------
    base_url_info : dict
        Dictionary containing 'url' and 'version' keys for the base URL.
    base_key : str
        The key identifying the base URL configuration.
    path : str
        The path component which may contain version placeholders.

    Returns
    -------
    str
        The fully constructed download URL.
    """
    version = base_url_info["version"]
    placeholder = f"__{base_key.upper()}_VERSION__"
    if placeholder in path:
        path = path.replace(placeholder, version)
    if version and not path.startswith(version + "/"):
        path_with_version = version + "/" + path.lstrip("/")
    else:
        path_with_version = path
    return base_url_info["url"].rstrip("/") + "/" + path_with_version.lstrip("/")


def _validate_download_entry(entry, index):
    """Validate a download entry from YAML configuration."""
    if not isinstance(entry, dict):
        raise ValueError(f"Download entry {index} must be a dictionary.")

    required_keys = ("base_url_key", "path", "description", "target_path")
    missing_keys = [key for key in required_keys if not entry.get(key)]
    if missing_keys:
        raise ValueError(f"Download entry {index} missing required keys: {', '.join(missing_keys)}")

    target_path = Path(entry["target_path"])
    if target_path.is_absolute() or ".." in target_path.parts:
        raise ValueError(
            f"Download entry {index} has invalid target_path: {target_path.as_posix()}"
        )


def download_files(config_file, target_dir):
    """Download external files listed in a resource-generation configuration.

    Parameters
    ----------
    config_file : str or pathlib.Path
        YAML file containing ``base_urls`` and ``files`` entries.
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
    if not isinstance(file_entries, list):
        raise ValueError("Download configuration key 'files' must be a list.")

    base_urls = download_config.get("base_urls", {})
    if not isinstance(base_urls, dict):
        raise ValueError("Download configuration key 'base_urls' must be a dictionary.")

    processed_base_urls = {}
    for base_key, base_info in base_urls.items():
        if isinstance(base_info, dict):
            url = base_info.get("url", "")
            version = base_info.get("version", "")
            processed_base_urls[base_key] = {"url": url, "version": version}

    downloaded_files = []
    for index, entry in enumerate(file_entries):
        _validate_download_entry(entry, index)

        base_key = entry["base_url_key"]
        if base_key not in processed_base_urls:
            raise ValueError(f"Base URL key '{base_key}' not found in base_urls configuration")

        base_url_info = processed_base_urls[base_key]
        url = _construct_download_url(base_url_info, base_key, entry["path"])

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


def generate_test_resources(args_dict, run_time=None):
    """Download inputs and run resource-generation workflows for a release.

    Parameters
    ----------
    args_dict : dict
        Dictionary containing command line arguments.
        Optional key ``overwrite_collection_files`` (bool) allows collection
        output files to be overwritten when different source files have the
        same basename. Defaults to False.
    run_time : list or None
        Prepared runtime command. If provided, reuse it instead of preparing
        the runtime environment from the workflow configuration.
    """
    test_directory = Path(args_dict["test_directory"])
    simtools_version = args_dict["simtools_version"]
    integration_test_dir = get_integration_test_directory(
        args_dict["test_directory"], args_dict["simtools_version"]
    )
    if args_dict.get("test_static_files"):
        validate_static_files(integration_test_dir / "static" / STATIC_MANIFEST)
        return

    config_dir = get_resource_generation_directory(test_directory, simtools_version)
    replacements = {
        "__TEST_DIRECTORY__": str(test_directory),
        "__SIMTOOLS_VERSION__": simtools_version,
    }
    downloaded_files = []
    if not args_dict.get("config_file"):
        downloaded_files = download_files(config_dir / "download_files.yml", integration_test_dir)
    if args_dict.get("download_only"):
        return

    try:
        run_configured_applications(
            args_dict=args_dict,
            config_dir=config_dir,
            log_dir=integration_test_dir / "log_files",
            run_time=run_time,
            replacements=replacements,
        )
    finally:
        _remove_empty_download_directories(downloaded_files, integration_test_dir)
