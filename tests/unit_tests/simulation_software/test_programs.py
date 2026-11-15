"""Tests for locating executables used by simulation chains."""

import importlib.util
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from simtools.simulation_software import programs


@pytest.fixture
def package_directory(tmp_test_directory, monkeypatch):
    """An installed obdeect package directory."""
    directory = Path(tmp_test_directory) / "obdeect"
    (directory / "propagation" / "bin").mkdir(parents=True)
    spec = SimpleNamespace(submodule_search_locations=[str(directory)])
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: spec)
    return directory


@pytest.fixture
def interpreter_directory(tmp_test_directory, monkeypatch):
    directory = Path(tmp_test_directory) / "interpreter"
    directory.mkdir()
    monkeypatch.setattr(sys, "executable", str(directory / "python"))
    return directory


def test_package_program(package_directory):
    program = package_directory / "propagation" / "bin" / "obdeect-propagation-run"
    program.write_text("#!/bin/sh\n", encoding="utf-8")

    assert programs.program_command("obdeect", "obdeect-propagation-run") == [str(program)]


def test_package_program_missing(package_directory):
    with pytest.raises(ValueError, match="install the package 'obdeect'"):
        programs.program_command("obdeect", "obdeect-propagation-run")


def test_missing_package(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    with pytest.raises(ValueError, match="install the package 'obdeect'"):
        programs.program_command("obdeect", "obdeect-propagation-run")


def test_package_lookup_error(monkeypatch):
    def failing(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(importlib.util, "find_spec", failing)

    with pytest.raises(ValueError, match="install the package 'obdeect'"):
        programs.program_command("obdeect", "obdeect-propagation-run")


def test_unknown_software():
    with pytest.raises(ValueError, match="Unknown software"):
        programs.program_command("other", "obdeect")


def test_unknown_program():
    with pytest.raises(ValueError, match="Unknown program 'other'"):
        programs.program_command("obdeect", "other")


def test_console_command_next_to_interpreter(interpreter_directory, monkeypatch):
    executable = interpreter_directory / "obdeect"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/obdeect")

    assert programs.program_command("obdeect", "obdeect") == [str(executable)]


def test_console_command_on_path(interpreter_directory, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")

    assert programs.program_command("obdeect", "obdeect") == ["/usr/bin/obdeect"]


def test_console_entry_point_fallback(package_directory, interpreter_directory, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)

    command = programs.program_command("obdeect", "obdeect")

    assert command[:2] == [str(interpreter_directory / "python"), "-c"]
    assert "from obdeect.stream.cli import main" in command[2]


def test_console_command_missing(package_directory, interpreter_directory, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    with pytest.raises(ValueError, match="install the package 'obdeect'"):
        programs.program_command("obdeect", "obdeect")


def test_installed_obdeect_programs_are_located():
    pytest.importorskip("obdeect")

    assert Path(programs.program_command("obdeect", "obdeect-propagation-run")[0]).is_file()
    assert programs.program_command("obdeect", "obdeect")
