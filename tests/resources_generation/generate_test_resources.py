#!/usr/bin/env python3

"""Generate test resources from fixtures and application configuration files."""

import argparse
import logging
import shutil
import urllib.error
import urllib.request
from pathlib import Path

from simtools.io import ascii_handler
from simtools.runners import simtools_runner

logger = logging.getLogger(__name__)

RESOURCE_GENERATION_DIR = Path(__file__).resolve().parent
MANUAL_FIXTURES_DIR = RESOURCE_GENERATION_DIR / "manual_fixtures"
RESOURCES_DIR = RESOURCE_GENERATION_DIR.parent / "resources"
DOWNLOAD_CONFIG_FILE = RESOURCE_GENERATION_DIR / "download_files.yml"
MANUAL_FIXTURE_CONFIG_FILE = RESOURCE_GENERATION_DIR / "manual_fixture.yml"


def _normalize_config_glob(config_glob):
    """Normalize config glob to be relative to RESOURCE_GENERATION_DIR."""
    prefix = f"tests/{RESOURCE_GENERATION_DIR.name}/"
    if config_glob.startswith(prefix):
        return config_glob[len(prefix) :]
    return config_glob


def _validate_manual_fixture_entry(entry, index):
    """Validate a manual fixture entry from YAML configuration."""
    if not isinstance(entry, dict):
        raise ValueError(f"Manual fixture entry {index} must be a dictionary.")

    required_keys = ("source_path", "description", "target_path")
    missing_keys = [key for key in required_keys if not entry.get(key)]
    if missing_keys:
        missing_keys_str = ", ".join(missing_keys)
        raise ValueError(f"Manual fixture entry {index} missing required keys: {missing_keys_str}")


def copy_manual_fixtures(
    config_file=MANUAL_FIXTURE_CONFIG_FILE,
    source_dir=MANUAL_FIXTURES_DIR,
    target_dir=RESOURCES_DIR,
):
    """Copy manual fixtures listed in YAML configuration to test resources."""
    if not Path(config_file).exists():
        logger.info("Manual fixture configuration file does not exist: %s", config_file)
        return

    if not Path(source_dir).exists():
        logger.info("Manual fixtures directory does not exist: %s", source_dir)
        return

    fixture_config = ascii_handler.collect_data_from_file(config_file)
    file_entries = fixture_config.get("files", []) if isinstance(fixture_config, dict) else []

    if not isinstance(file_entries, list):
        raise ValueError("Manual fixture configuration key 'files' must be a list.")

    for index, entry in enumerate(file_entries):
        _validate_manual_fixture_entry(entry, index)

        source_path = Path(source_dir) / Path(entry["source_path"])
        target_path = Path(target_dir) / Path(entry["target_path"])

        if not source_path.exists():
            raise FileNotFoundError(
                f"Manual fixture '{entry['description']}' does not exist at {source_path}"
            )

        target_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Copying %s from %s to %s", entry["description"], source_path, target_path)

        if source_path.is_dir():
            shutil.copytree(source_path, target_path, dirs_exist_ok=True)
        else:
            shutil.copy2(source_path, target_path)


def run_configured_applications(
    config_dir=RESOURCE_GENERATION_DIR,
    ignore_runtime_environment=True,
    config_glob="*.config.yml",
    overwrite_collection_files=False,
):
    """Run configured applications matching the given glob pattern."""
    normalized_glob = _normalize_config_glob(config_glob)
    for config_file in sorted(config_dir.rglob(normalized_glob)):
        logger.info("Executing applications configured in %s", config_file)
        simtools_runner.run_applications(
            {
                "config_file": str(config_file),
                "steps": None,
                "ignore_runtime_environment": ignore_runtime_environment,
                "overwrite_collection_files": overwrite_collection_files,
            }
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


def download_files(config_file=DOWNLOAD_CONFIG_FILE, target_dir=RESOURCES_DIR):
    """Download files listed in YAML configuration into test resources."""
    if not Path(config_file).exists():
        logger.info("Download configuration file does not exist: %s", config_file)
        return

    download_config = ascii_handler.collect_data_from_file(config_file)
    file_entries = download_config.get("files", []) if isinstance(download_config, dict) else []

    if not isinstance(file_entries, list):
        raise ValueError("Download configuration key 'files' must be a list.")

    for index, entry in enumerate(file_entries):
        _validate_download_entry(entry, index)

        destination = Path(target_dir) / Path(entry["target_path"])
        destination.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Downloading %s from %s to %s", entry["description"], entry["url"], destination)
        try:
            urllib.request.urlretrieve(entry["url"], destination)
        except urllib.error.HTTPError as exc:
            exc.close()
            raise FileNotFoundError(
                f"Failed to download '{entry['description']}' from {entry['url']}"
            ) from exc


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate files in tests/resources from tests/resources_generation."
    )
    parser.add_argument(
        "--ignore_runtime_environment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ignore runtime environments configured in application files.",
    )
    parser.add_argument(
        "--config_glob",
        type=str,
        default="*.config.yml",
        help=(
            "Glob pattern under tests/resources_generation for selecting config files "
            "(e.g. 'model_parameters/*.config.yml' for debugging)."
        ),
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
    """Generate test resources."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", force=True)
    args = parse_args()

    copy_manual_fixtures()
    download_files()
    run_configured_applications(
        ignore_runtime_environment=args.ignore_runtime_environment,
        config_glob=args.config_glob,
        overwrite_collection_files=args.overwrite_collection_files,
    )


if __name__ == "__main__":
    main()
