"""Tests for the dependency-versions application entry points."""

import sys

from simtools.applications import dependency_versions


def test_application_parser_accepts_environment_format():
    """Test the regular application parser exposes the environment format."""
    parser = dependency_versions.APPLICATION.build_parser()
    format_action = next(action for action in parser._actions if action.dest == "format")

    assert "env" in format_action.choices


def test_standalone_parser_accepts_environment_format(monkeypatch, mocker, capsys):
    """Test the standalone parser accepts and exports the environment format."""
    monkeypatch.setattr(sys, "argv", ["simtools-dependency-versions", "--format", "env"])
    mocker.patch.object(dependency_versions, "_export_dependency_configuration", return_value="ok")

    dependency_versions._main_standalone()

    assert capsys.readouterr().out == "ok"


def test_standalone_parser_updates_source_revisions(monkeypatch, mocker, tmp_test_directory):
    """Test the standalone parser delegates catalog source-revision updates."""
    updates_path = tmp_test_directory / "source-updates.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["simtools-dependency-versions", "--update-source-revisions", str(updates_path)],
    )
    update = mocker.patch.object(dependency_versions, "_update_source_revisions")

    dependency_versions._main_standalone()

    update.assert_called_once_with(updates_path)
