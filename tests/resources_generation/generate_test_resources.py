#!/usr/bin/env python3

"""Generate test resources from fixtures and application configuration files."""

import argparse
import logging
import shutil
from pathlib import Path

from simtools.runners import simtools_runner

logger = logging.getLogger(__name__)

RESOURCE_GENERATION_DIR = Path(__file__).resolve().parent
MANUAL_FIXTURES_DIR = RESOURCE_GENERATION_DIR / "manual_fixtures"
RESOURCES_DIR = RESOURCE_GENERATION_DIR.parent / "resources"


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
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    copy_manual_fixtures()
    run_configured_applications(ignore_runtime_environment=args.ignore_runtime_environment)


if __name__ == "__main__":
    main()
