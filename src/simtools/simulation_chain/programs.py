"""Location of the programs run by simulation chains.

The catalog ``resources/simulation_programs.yml`` lists, for each software, the Python package
that ships its programs and each program as a file relative to the package directory or as a
console command. :func:`program_command` returns the command tokens of a program. The packages
are located without importing them.

A program that is a console command is resolved in this order: an executable next to the running
Python interpreter, an executable on ``PATH``, and a call of the entry point of the command
(``module:function``) in a Python process of the running interpreter.
"""

import importlib.util
import shutil
import sys
from functools import cache
from importlib.resources import files
from pathlib import Path

import yaml

PROGRAM_CATALOG = "resources/simulation_programs.yml"


@cache
def program_catalog():
    """Return the catalog of the software and its programs."""
    text = files("simtools").joinpath(PROGRAM_CATALOG).read_text(encoding="utf-8")
    return yaml.safe_load(text)["software"]


def _program_entry(software, name):
    """Return the catalog entries of a software and of one of its programs."""
    catalog = program_catalog()
    if software not in catalog:
        raise ValueError(f"Unknown software {software!r}; known: {sorted(catalog)}")
    programs = catalog[software].get("programs", {})
    if name not in programs:
        raise ValueError(
            f"Unknown program {name!r} of software {software!r}; known: {sorted(programs)}"
        )
    return catalog[software], programs[name]


def _package_directories(package):
    """Return the directories of an installed package (empty if it is not installed)."""
    try:
        spec = importlib.util.find_spec(package)
    except ImportError, ValueError:
        return []
    return [] if spec is None else [Path(path) for path in spec.submodule_search_locations or []]


def _install_hint(software, name, package):
    """Return the error text of a program that is not available."""
    return (
        f"Program {name!r} of software {software!r} not found; install the package "
        f"{package!r} to run it."
    )


def _file_program(software, name, package, relative_path):
    """Return the command tokens of a program that is a file of a package."""
    for directory in _package_directories(package):
        program = directory / relative_path
        if program.is_file():
            return [str(program)]
    raise ValueError(_install_hint(software, name, package))


def _entry_point_command(entry_point):
    """Return the command tokens that call an entry point (``module:function``) in Python."""
    module, function = entry_point.split(":")
    code = f"import sys; from {module} import {function}; sys.exit({function}(sys.argv[1:]))"
    return [sys.executable, "-c", code]


def _console_command(software, name, package, entry):
    """Return the command tokens of a console command."""
    command = entry["command"]
    candidate = Path(sys.executable).parent / command
    if candidate.is_file():
        return [str(candidate)]
    found = shutil.which(command)
    if found:
        return [found]
    if entry.get("entry_point") and _package_directories(package):
        return _entry_point_command(entry["entry_point"])
    raise ValueError(_install_hint(software, name, package))


def program_command(software, name):
    """
    Return the command tokens of a program of a software.

    Parameters
    ----------
    software : str
        Software name (key of the program catalog), e.g. ``obdeect``.
    name : str
        Program name (key of the programs of the software).

    Returns
    -------
    list of str
        Command tokens; arguments are appended to them.

    Raises
    ------
    ValueError
        If the software or the program is not in the catalog, or the program is not installed.
    """
    entry, program = _program_entry(software, name)
    package = entry["package"]
    if "file" in program:
        return _file_program(software, name, package, program["file"])
    return _console_command(software, name, package, program)
