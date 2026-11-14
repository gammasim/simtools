"""Tests for the location of the programs of simulation chains."""

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from simtools.simulation_chain import programs

CATALOG = {
    "tool": {
        "package": "toolpkg",
        "programs": {
            "tool-file": {"file": "bin/tool-file"},
            "tool": {"command": "tool", "entry_point": "toolpkg.cli:main"},
            "tool-plain": {"command": "tool-plain"},
        },
    }
}


@pytest.fixture
def catalog(monkeypatch):
    monkeypatch.setattr(programs, "program_catalog", lambda: CATALOG)


@pytest.fixture
def package_directory(tmp_test_directory, monkeypatch):
    """Directory of the installed package ``toolpkg`` (found by ``find_spec``)."""
    directory = Path(tmp_test_directory) / "toolpkg"
    (directory / "bin").mkdir(parents=True)
    spec = SimpleNamespace(submodule_search_locations=[str(directory)])
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: spec if name == "toolpkg" else None
    )
    return directory


@pytest.fixture
def interpreter_directory(tmp_test_directory, monkeypatch):
    """Directory of a fake Python interpreter."""
    directory = Path(tmp_test_directory) / "interpreter"
    directory.mkdir()
    monkeypatch.setattr(sys, "executable", str(directory / "python"))
    return directory


def test_catalog_lists_obdeect_programs():
    catalog = programs.program_catalog()

    assert catalog["obdeect"]["package"] == "obdeect"
    assert {"obdeect", "obdeect-propagation-run", "obdeect-sources-point-source"} <= set(
        catalog["obdeect"]["programs"]
    )


def test_file_program_in_package(catalog, package_directory):
    program = package_directory / "bin" / "tool-file"
    program.write_text("#!/bin/sh\n", encoding="utf-8")

    assert programs.program_command("tool", "tool-file") == [str(program)]


def test_file_program_missing_in_installed_package(catalog, package_directory):
    with pytest.raises(ValueError, match=r"Program 'tool-file' .* install the package 'toolpkg'"):
        programs.program_command("tool", "tool-file")


def test_missing_package(catalog, monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    with pytest.raises(ValueError, match=r"install the package 'toolpkg'"):
        programs.program_command("tool", "tool-file")


def test_package_without_search_locations(catalog, monkeypatch):
    spec = SimpleNamespace(submodule_search_locations=None)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: spec)

    with pytest.raises(ValueError, match="install the package"):
        programs.program_command("tool", "tool-file")


def test_package_lookup_error(catalog, monkeypatch):
    def failing(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(importlib.util, "find_spec", failing)

    with pytest.raises(ValueError, match="install the package"):
        programs.program_command("tool", "tool-file")


def test_unknown_software(catalog):
    with pytest.raises(ValueError, match=r"Unknown software 'other'; known: \['tool'\]"):
        programs.program_command("other", "tool")


def test_unknown_program(catalog):
    with pytest.raises(ValueError, match=r"Unknown program 'other' of software 'tool'"):
        programs.program_command("tool", "other")


def test_command_next_to_interpreter_is_first(catalog, interpreter_directory, monkeypatch):
    executable = interpreter_directory / "tool"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/tool")

    assert programs.program_command("tool", "tool") == [str(executable)]


def test_command_on_path(catalog, interpreter_directory, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")

    assert programs.program_command("tool", "tool") == ["/usr/bin/tool"]


def test_command_entry_point_fallback(
    catalog, interpreter_directory, package_directory, monkeypatch
):
    monkeypatch.setattr(shutil, "which", lambda name: None)

    command = programs.program_command("tool", "tool")

    assert command[:2] == [str(interpreter_directory / "python"), "-c"]
    assert "from toolpkg.cli import main" in command[2]
    assert "sys.exit(main(sys.argv[1:]))" in command[2]


def test_command_missing_package_has_no_fallback(catalog, interpreter_directory, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    with pytest.raises(ValueError, match="install the package 'toolpkg'"):
        programs.program_command("tool", "tool")


def test_command_without_entry_point_is_missing(
    catalog, interpreter_directory, package_directory, monkeypatch
):
    monkeypatch.setattr(shutil, "which", lambda name: None)

    with pytest.raises(ValueError, match="Program 'tool-plain'"):
        programs.program_command("tool", "tool-plain")


def test_entry_point_command_runs_a_function(tmp_test_directory):
    package = Path(tmp_test_directory) / "entrypkg"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "cli.py").write_text(
        "def main(argv):\n    print(' '.join(argv))\n    return 0\n", encoding="utf-8"
    )
    command = programs._entry_point_command("entrypkg.cli:main")  # pylint: disable=W0212

    result = subprocess.run(
        [*command, "a", "b"],
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONPATH": str(tmp_test_directory)},
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "a b"


def test_installed_obdeect_programs_are_located():
    pytest.importorskip("obdeect")

    assert Path(programs.program_command("obdeect", "obdeect-propagation-run")[0]).is_file()
    assert programs.program_command("obdeect", "obdeect")
