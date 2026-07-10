"""Unit tests for the simtools CLI-help Sphinx extension."""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs" / "source" / "_ext"))

import simtools_cli_help
from simtools_cli_help import load_application_parser, render_native_cli_docs


def test_render_native_cli_docs_hides_selected_groups():
    """Test native rendering can suppress named argparse groups."""
    parser = argparse.ArgumentParser(prog="simtools-test")
    parser.add_argument("--global-option", help="global option")
    repeated_group = parser.add_argument_group("paths")
    repeated_group.add_argument("--output_path", help="output path")
    app_group = parser.add_argument_group("application")
    app_group.add_argument("--figure_name", help="figure name")

    inspection = simtools_cli_help.CliInspection(
        module=SimpleNamespace(__name__="simtools.applications.fake_app"),
        parser=parser,
    )
    rendered_nodes = render_native_cli_docs(inspection, {"paths"})
    rendered_text = rendered_nodes[0].astext()

    assert "--figure_name" in rendered_text
    assert "--output_path" not in rendered_text


def test_load_application_parser_captures_main_build_application(mocker):
    """Test parser loading reuses the module's build_application call shape."""
    fake_module = SimpleNamespace(
        __name__="simtools.applications.fake_app",
        __file__="/tmp/fake_app.py",
        __doc__="Fake application.",
        _add_arguments=lambda parser: parser.add_argument("--figure_name", help="figure name"),
    )

    def _main():
        fake_module.build_application(initialization_kwargs={"output": True}, usage="simtools-fake")

    fake_module.main = _main
    fake_module.build_application = object()

    parser = argparse.ArgumentParser(prog="simtools-fake")
    parser.add_argument("--figure_name", help="figure name")
    parser.add_argument("--output_file", help="output file")

    mocker.patch.object(simtools_cli_help.importlib, "import_module", return_value=fake_module)
    build_parser = mocker.patch.object(
        simtools_cli_help, "build_application_parser", return_value=parser
    )

    inspection = load_application_parser("fake_app", prog="simtools-fake")

    assert inspection.parser is parser
    assert inspection.parser.prog == "simtools-fake"
    build_parser.assert_called_once_with(
        application_path="/tmp/fake_app.py",
        description="Fake application.",
        add_arguments_function=fake_module._add_arguments,
        initialization_kwargs={"output": True},
        usage="simtools-fake",
        epilog=None,
    )


def test_load_application_parser_requires_build_application():
    """Test parser loading fails clearly when a module cannot be intercepted."""
    fake_module = SimpleNamespace(__name__="simtools.applications.fake_app", main=lambda: None)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(simtools_cli_help.importlib, "import_module", lambda *_: fake_module)
        with pytest.raises(ValueError, match="has no build_application import"):
            load_application_parser("fake_app")


def test_render_native_cli_docs_uses_scoped_doc_groups():
    """Test application-specific doc groups are read from parser metadata."""
    parser = argparse.ArgumentParser(prog="simtools-test")
    app_group = parser.add_argument_group("simulation model")
    action = app_group.add_argument("--site", help="site")
    action.simtools_doc = "site"
    action.simtools_doc_group = "simulation model"
    action.simtools_doc_groups = {"production_generate_grid": "Production context"}
    action.simtools_doc_hidden = False

    inspection = simtools_cli_help.CliInspection(
        module=SimpleNamespace(__name__="simtools.applications.production_generate_grid"),
        parser=parser,
    )
    rendered_text = render_native_cli_docs(inspection, set())[0].astext()

    assert "Production context" in rendered_text
    assert "simulation model" not in rendered_text
