"""
Custom build backend for simtools with optional native LightEmission extension.

This build backend automatically attempts to build the native C++ extension
when SIMTOOLS_BUILD_LE=1 environment variable is set, but gracefully falls back
to a pure Python installation if the build fails.
"""

import logging
import os
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

# Import the standard setuptools backend
from setuptools import build_meta as _orig

_logger = logging.getLogger(__name__)


def _should_build_native() -> bool:
    """Check if we should attempt to build the native extension."""
    return os.environ.get("SIMTOOLS_BUILD_LE") == "1"


def _try_build_native() -> bool:
    """
    Attempt to build the native extension.

    Returns
    -------
        bool: True if build succeeded, False otherwise
    """
    if not _should_build_native():
        return False

    if find_spec("pybind11") is None:
        _logger.info("pybind11 not available, skipping native extension build")
        return False

    try:
        # Find the setup script
        setup_script = Path("setup_native_extension.py")
        if not setup_script.exists():
            _logger.info("setup_native_extension.py not found, skipping native extension build")
            return False

        # Try to build the extension
        _logger.info("Attempting to build native LightEmission extension...")
        result = subprocess.run(
            [sys.executable, str(setup_script), "build_ext", "--inplace"],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            check=False,
        )

        if result.returncode == 0:
            _logger.info("\u2713 Native extension built successfully")
            return True
        _logger.warning(f"Native extension build failed (exit code {result.returncode})")
        if result.stdout:
            _logger.debug(f"stdout: {result.stdout}")
        if result.stderr:
            _logger.debug(f"stderr: {result.stderr}")
        return False

    except (OSError, subprocess.SubprocessError) as e:
        _logger.warning(f"Exception during native extension build: {e}")
        return False


# Wrap the original build functions to include native extension building
def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """Build wheel with optional native extension."""
    # Try to build native extension first
    _try_build_native()

    # Use the original setuptools backend
    return _orig.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory, config_settings=None):
    """Build source distribution."""
    return _orig.build_sdist(sdist_directory, config_settings)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    """Build editable wheel with optional native extension."""
    # Try to build native extension first
    _try_build_native()

    # Use the original setuptools backend
    return _orig.build_editable(wheel_directory, config_settings, metadata_directory)


def get_requires_for_build_wheel(config_settings=None):
    """Get requirements for building wheel."""
    reqs = _orig.get_requires_for_build_wheel(config_settings)

    # Add pybind11 if we're building native extension
    if _should_build_native():
        if "pybind11" not in reqs:
            reqs.append("pybind11>=2.11")

    return reqs


def get_requires_for_build_sdist(config_settings=None):
    """Get requirements for building source distribution."""
    return _orig.get_requires_for_build_sdist(config_settings)


def get_requires_for_build_editable(config_settings=None):
    """Get requirements for building editable install."""
    reqs = _orig.get_requires_for_build_editable(config_settings)

    # Add pybind11 if we're building native extension
    if _should_build_native():
        if "pybind11" not in reqs:
            reqs.append("pybind11>=2.11")

    return reqs


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    """Prepare metadata for wheel build."""
    return _orig.prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    """Prepare metadata for editable build."""
    return _orig.prepare_metadata_for_build_editable(metadata_directory, config_settings)


# Export the interface expected by PEP 517
__all__ = [
    "build_editable",
    "build_sdist",
    "build_wheel",
    "get_requires_for_build_editable",
    "get_requires_for_build_sdist",
    "get_requires_for_build_wheel",
    "prepare_metadata_for_build_editable",
    "prepare_metadata_for_build_wheel",
]
