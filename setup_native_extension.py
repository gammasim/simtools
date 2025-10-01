"""
Setup script for simtools with optional native LightEmission C++ extensions.

This setup.py handles conditional building of the _le extension when
SIMTOOLS_BUILD_LE environment variable is set and sim_telarray is available.
"""

import logging
import os
from pathlib import Path

from setuptools import setup

_logger = logging.getLogger(__name__)
_LE_HEADER = "IactLightEmission.hh"
_CONTAINER_ROOT = "/workdir/sim_telarray"
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def _header_candidates_for_path(base: Path) -> list[Path]:
    return [
        base / "LightEmission" / _LE_HEADER,
        base / "sim_telarray" / "LightEmission" / _LE_HEADER,
        Path(_CONTAINER_ROOT) / "sim_telarray" / "LightEmission" / _LE_HEADER,
    ]


def _resolve_simtel_base_from_header(header_path: Path) -> Path | None:
    text = str(header_path)
    if "sim_telarray/LightEmission" in text:
        return header_path.parent.parent
    if "/workdir/sim_telarray" in text:
        return Path("/workdir/sim_telarray/sim_telarray")
    return header_path.parents[2] if header_path.exists() else None


def _detect_simtel_path_from_env() -> tuple[Path | None, bool]:
    """Return (simtel_path, has_sources) based on SIMTEL_PATH or container defaults."""
    simtel_env = os.environ.get("SIMTEL_PATH")
    if simtel_env:
        base = Path(simtel_env)
        for header in _header_candidates_for_path(base):
            if header.exists():
                _logger.info(f"\u2713 Found LightEmission sources at {header}")
                resolved = _resolve_simtel_base_from_header(header)
                return resolved or base, True
        _logger.warning(
            "LightEmission header not found in expected locations; building placeholder"
        )
        return base, False

    container_header = Path(_CONTAINER_ROOT) / "sim_telarray" / "LightEmission" / _LE_HEADER
    if container_header.exists():
        _logger.info(f"\u2713 Found LightEmission sources at {container_header}")
        return Path("/workdir/sim_telarray/sim_telarray"), True

    _logger.warning("SIMTEL_PATH not set and container headers not found; placeholder build")
    return None, False


def _extension_sources_and_includes(
    simtel_path: Path | None, has_sources: bool
) -> tuple[list[str], list[str]]:
    """Return sources and include dirs for the extension based on availability of sources."""
    sources = ["src/simtools/_le.cpp"]
    include_dirs: list[str] = []

    if has_sources and simtel_path:
        include_dirs.extend(
            [
                str(simtel_path / "include"),
                str(simtel_path / "LightEmission"),
                str(simtel_path / "src"),
                str(simtel_path),
            ]
        )
        _logger.info(f"Using include directories: {include_dirs}")

    return sources, include_dirs


def build_le_extension():
    """Build the _le extension if LightEmission support is requested and available."""
    # Only build if explicitly requested
    if not os.environ.get("SIMTOOLS_BUILD_LE"):
        return []
    try:
        from pybind11.setup_helpers import (  # pylint: disable=import-outside-toplevel
            Pybind11Extension,  # type: ignore
        )
    except ImportError:
        _logger.info("pybind11 not available; skipping native extension build")
        return []

    # Detect paths and assemble sources
    simtel_path, has_lightemission_sources = _detect_simtel_path_from_env()
    sources, include_dirs = _extension_sources_and_includes(simtel_path, has_lightemission_sources)

    # Always define USE_LIGHTEMISSION for conditional compilation
    ext_define_macros = [
        ("USE_LIGHTEMISSION", "1" if has_lightemission_sources else "0"),
        ("CTA_PROD3", "1"),
        ("IACT_NO_GRID", "1"),
    ]

    # Create the extension
    ext = Pybind11Extension(
        "simtools.light_emission._le",  # Place in light_emission subpackage
        sources=sources,
        include_dirs=include_dirs,
        define_macros=ext_define_macros,
        extra_compile_args=["-std=c++17", "-Wall"],
        libraries=["m"],  # Remove "z" (zlib) dependency for minimal implementation
        optional=True,
    )

    _logger.info(f"Building native LightEmission extension with sources: {sources}")
    return [ext]


if __name__ == "__main__":
    # Get extensions (if any)
    ext_modules = build_le_extension()

    # Setup arguments
    setup_args = {}
    if ext_modules:
        # Import here to avoid top-level dependency on pybind11
        try:
            from pybind11.setup_helpers import build_ext  # type: ignore
        except ImportError:
            build_ext = None  # type: ignore
        if build_ext is not None:
            setup_args.update({"ext_modules": ext_modules, "cmdclass": {"build_ext": build_ext}})

    # Call setup with conditional extensions
    setup(**setup_args)
