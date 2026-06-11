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


def copy_manual_fixtures(source_dir=MANUAL_FIXTURES_DIR, target_dir=RESOURCES_DIR):
    """Copy all manual fixtures to the test resources directory."""
    if not source_dir.exists():
        logger.info("Manual fixtures directory does not exist: %s", source_dir)
        return

    logger.info("Copying manual fixtures from %s to %s", source_dir, target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for source_path in source_dir.iterdir():
        target_path = target_dir / source_path.name
        logger.info("Copying %s to %s", source_path, target_path)
        if source_path.is_dir():
            shutil.copytree(source_path, target_path, dirs_exist_ok=True)
        else:
            shutil.copy2(source_path, target_path)


def run_configured_applications(
    config_dir=RESOURCE_GENERATION_DIR, ignore_runtime_environment=True
):
    """Run all applications configured in *.config.yml files."""
    for config_file in sorted(config_dir.rglob("*.config.yml")):
        logger.info("Executing applications configured in %s", config_file)
        simtools_runner.run_applications(
            {
                "config_file": str(config_file),
                "steps": None,
                "ignore_runtime_environment": ignore_runtime_environment,
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


def download_configured_files(config_file=DOWNLOAD_CONFIG_FILE, target_dir=RESOURCES_DIR):
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
    return parser.parse_args()


def main():
    """Generate test resources."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", force=True)
    args = parse_args()

    copy_manual_fixtures()
    download_configured_files()
    run_configured_applications(ignore_runtime_environment=args.ignore_runtime_environment)


if __name__ == "__main__":
    main()
