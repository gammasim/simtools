#!/usr/bin/env python3
"""
Build native LightEmission extension for simtools.

This script provides a Python interface for building the native C++ extension
that provides direct bindings to sim_telarray LightEmission functionality.
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

# Constants
SETUP_SCRIPT_NAME = "setup_native_extension.py"


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def find_project_root() -> Path:
    """Find the project root directory."""
    current = Path.cwd()
    pyproject_name = "pyproject.toml"

    # Look for setup files
    for parent in [current, *current.parents]:
        if (parent / SETUP_SCRIPT_NAME).exists() or (parent / pyproject_name).exists():
            return parent

    raise RuntimeError(
        f"Cannot find project root ({SETUP_SCRIPT_NAME} or {pyproject_name} not found)"
    )


def build_extension(verbose: bool = False) -> bool:
    """Build the native extension."""
    logger = logging.getLogger(__name__)

    project_root = find_project_root()
    logger.debug(f"Project root: {project_root}")

    # Set build environment variables
    env = os.environ.copy()
    env["SIMTOOLS_BUILD_LE"] = "1"

    # Change to project directory
    os.chdir(project_root)

    # Use setup_native_extension.py for building
    setup_py = project_root / SETUP_SCRIPT_NAME
    if not setup_py.exists():
        logger.error(f"{SETUP_SCRIPT_NAME} not found in project root")
        return False

    cmd = [sys.executable, SETUP_SCRIPT_NAME, "build_ext", "--inplace"]
    if verbose:
        cmd.append("--verbose")

    logger.info("Building extension with: %s", " ".join(cmd))

    try:
        result = subprocess.run(cmd, env=env, capture_output=not verbose, text=True, check=True)
        logger.info("✓ Native extension built successfully")

        if not verbose and result.stdout:
            logger.debug(f"Build output: {result.stdout}")

    except subprocess.CalledProcessError as e:
        logger.error(f"Build failed with exit code {e.returncode}")
        if e.stdout:
            logger.error(f"stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"stderr: {e.stderr}")
        return False

    return True


def main():
    """Provide main entry point."""
    parser = argparse.ArgumentParser(
        description="Build native LightEmission extension for simtools"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("Starting native LightEmission build process")

    # Check for sources (just for informational purposes)
    simtel_path = os.environ.get("SIMTEL_PATH")
    if simtel_path:
        le_header = Path(simtel_path) / "LightEmission" / "IactLightEmission.hh"
        if le_header.exists():
            logger.info("✓ LightEmission sources found")
        else:
            logger.info("Building with placeholder implementation (no LightEmission sources)")
    else:
        logger.info("SIMTEL_PATH not set, building with placeholder implementation")

    if not build_extension(args.verbose):
        logger.error("Build failed")
        sys.exit(1)

    logger.info("Build process completed successfully")
    logger.info("You can now use native LightEmission acceleration in simtools")


if __name__ == "__main__":
    main()
