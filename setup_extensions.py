"""
Custom setup.py to handle conditional C++ extension building for LightEmission bindings.

This is used when SIMTOOLS_BUILD_LE environment variable is set, indicating that
the user has sim_telarray LightEmission headers and libraries available.
"""

import os
from pathlib import Path

from setuptools import setup


def build_extension():
    """Build the _le extension if LightEmission support is requested."""
    # Only build if explicitly requested
    if not os.environ.get("SIMTOOLS_BUILD_LE"):
        return []
    try:
        from pybind11.setup_helpers import (  # pylint: disable=import-outside-toplevel
            Pybind11Extension,  # type: ignore
        )
    except ImportError:
        return []

    # Look for sim_telarray path in environment
    simtel_path = os.environ.get("SIMTEL_PATH")
    if not simtel_path:
        print("Error: SIMTEL_PATH environment variable must be set for LightEmission build")
        return []

    simtel_path = Path(simtel_path)
    le_path = simtel_path / "LightEmission"

    if not le_path.exists():
        print(f"Error: LightEmission directory not found at {le_path}")
        return []

    # Required source files (based on ff-1m.cc compilation)
    sources = [
        "src/simtools/_le.cpp",
        str(le_path / "IactLightEmission.cc"),
        str(simtel_path / "src" / "iact.c"),
        str(simtel_path / "src" / "io_simtel.c"),
        str(simtel_path / "src" / "eventio.c"),
        str(simtel_path / "src" / "fileopen.c"),
        str(simtel_path / "src" / "warning.c"),
        str(simtel_path / "src" / "straux.c"),
        str(simtel_path / "src" / "atmo.c"),
        str(simtel_path / "src" / "sampling.c"),
        str(simtel_path / "src" / "rndm2.c"),
        str(simtel_path / "src" / "sim_absorb.c"),
    ]

    # Check that required files exist
    missing = [s for s in sources[1:] if not Path(s).exists()]
    if missing:
        print(f"Warning: Some sim_telarray sources not found: {missing[:3]}...")

    # Define extension
    ext = Pybind11Extension(
        "simtools.light_emission._le",
        sources=sources,
        include_dirs=[
            str(simtel_path / "include"),
            str(le_path),
            str(simtel_path / "src"),
        ],
        define_macros=[("USE_LIGHTEMISSION", "1"), ("CTA_PROD3", "1"), ("IACT_NO_GRID", "1")],
        libraries=["m", "z"],
        cxx_std=17,
        optional=True,  # Don't fail the whole build if this fails
    )

    return [ext]


if __name__ == "__main__":
    # This is only used when building with extensions
    # Import here to avoid top-level dependency on pybind11 when not building
    try:
        from pybind11.setup_helpers import build_ext  # type: ignore
    except ImportError:
        setup()
    else:
        setup(
            ext_modules=build_extension(),
            cmdclass={"build_ext": build_ext},
        )
