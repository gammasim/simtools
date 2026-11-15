"""Locate the executables used by the obdeect simulation chain."""

import importlib.util
import shutil
import sys
from pathlib import Path

_OBDEECT_PROGRAMS = {
    "obdeect-propagation-run": "propagation/bin/obdeect-propagation-run",
    "obdeect-stream-from-corsika": "stream/bin/obdeect-stream-from-corsika",
    "obdeect-stream-to-simtel": "stream/bin/obdeect-stream-to-simtel",
    "obdeect-sources-point-source": "sources/bin/obdeect-sources-point-source",
    "obdeect-sources-flasher": "sources/bin/obdeect-sources-flasher",
}


def _package_directories():
    """Return directories containing an installed obdeect package."""
    try:
        spec = importlib.util.find_spec("obdeect")
    except ImportError, ValueError:
        return []
    if spec is None:
        return []
    return [Path(path) for path in spec.submodule_search_locations or []]


def _install_error(name):
    return ValueError(f"Program {name!r} not found; install the package 'obdeect' to run it.")


def program_command(software, name):
    """Return command tokens for an executable used by a simulation chain."""
    if software != "obdeect":
        raise ValueError("Unknown software; supported software: ['obdeect']")
    if name == "obdeect":
        command = "obdeect"
        candidate = Path(sys.executable).parent / command
        if candidate.is_file():
            return [str(candidate)]
        found = shutil.which(command)
        if found:
            return [found]
        if _package_directories():
            code = "import sys; from obdeect.stream.cli import main; sys.exit(main(sys.argv[1:]))"
            return [sys.executable, "-c", code]
        raise _install_error(name)

    if name not in _OBDEECT_PROGRAMS:
        raise ValueError(
            f"Unknown program {name!r} of software 'obdeect'; "
            f"known: {sorted([*_OBDEECT_PROGRAMS, 'obdeect'])}"
        )
    for directory in _package_directories():
        program = directory / _OBDEECT_PROGRAMS[name]
        if program.is_file():
            return [str(program)]
    raise _install_error(name)
