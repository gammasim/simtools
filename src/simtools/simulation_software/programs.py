"""Locate the executables of the simulation software (``programs`` in the software catalog)."""

import importlib.util
import shutil
import sys
from pathlib import Path

from simtools.runtime import software_catalog


def _package_directories(package):
    """Return directories containing an installed Python package."""
    try:
        spec = importlib.util.find_spec(package)
    except ImportError, ValueError:
        return []
    if spec is None:
        return []
    return [Path(path) for path in spec.submodule_search_locations or []]


def _install_error(name, package):
    return ValueError(f"Program {name!r} not found; install the package {package!r} to run it.")


def _command(name, package, module):
    """Return the tokens of the main command of a package (its console script, else its module)."""
    candidate = Path(sys.executable).parent / name
    if candidate.is_file():
        return [str(candidate)]
    found = shutil.which(name)
    if found:
        return [found]
    if module and _package_directories(package):
        code = f"import sys; from {module} import main; sys.exit(main(sys.argv[1:]))"
        return [sys.executable, "-c", code]
    raise _install_error(name, package)


def program_command(software, name):
    """
    Return the command tokens of a program of a simulation software.

    Parameters
    ----------
    software : str
        Simulation software (catalog name) with a Python package.
    name : str
        The software's ``command`` or one of its ``programs``.

    Raises
    ------
    ValueError
        If the software has no package, the program is unknown or not installed.
    """
    entry = software_catalog.software(software)
    package = entry.get("package")
    if package is None:
        raise ValueError(f"Software {software!r} has no Python package with programs.")
    if name == entry.get("command"):
        return _command(name, package, entry.get("command_module"))
    programs = entry.get("programs", {})
    if name not in programs:
        raise ValueError(
            f"Unknown program {name!r} of software {software!r}; "
            f"known: {sorted([*programs, entry.get('command')])}"
        )
    for directory in _package_directories(package):
        program = directory / programs[name]
        if program.is_file():
            return [str(program)]
    raise _install_error(name, package)
