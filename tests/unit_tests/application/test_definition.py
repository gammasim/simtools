"""Tests for explicit application definitions."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from simtools.application.definition import ApplicationDefinition
from simtools.configuration.arguments import ArgumentDefinition


def test_argument_definition_validates_name_and_overrides():
    """Argument definitions validate names and create independent overridden copies."""
    definition = ArgumentDefinition("input_file", type=str, required=False)
    required = definition(required=True)

    assert definition.kwargs["required"] is False
    assert required.kwargs["required"] is True
    with pytest.raises(ValueError, match="Invalid argument name"):
        ArgumentDefinition("--input_file")


def test_argument_definition_override_preserves_declaration_metadata():
    """Application-local overrides retain non-argparse declaration metadata."""
    definition = ArgumentDefinition(
        "input_file", group="input", preserve_by_version=True, required=False
    )

    required = definition(required=True)

    assert required.group == "input"
    assert required.preserve_by_version is True


def test_application_definition_rejects_duplicate_arguments():
    """Duplicate option names fail when the application is declared."""
    with pytest.raises(ValueError, match=r"Duplicate command-line argument.*input"):
        ApplicationDefinition(
            module_name="simtools.applications.test",
            description="Test application.",
            arguments=(ArgumentDefinition("input"), ArgumentDefinition("input")),
        )


def test_build_parser_registers_groups_and_exclusive_arguments():
    """The parser factory preserves display and mutually exclusive groups."""
    application = ApplicationDefinition(
        module_name="simtools.applications.test",
        description="Test application.",
        arguments=(
            ArgumentDefinition("value", group="input", type=int),
            ArgumentDefinition(
                "file",
                exclusive_group="source",
                exclusive_group_required=True,
            ),
            ArgumentDefinition(
                "directory",
                exclusive_group="source",
                exclusive_group_required=True,
            ),
        ),
    )

    parser = application.build_parser()
    assert parser.parse_args(["--file", "events.simtel.zst", "--value", "2"]).value == 2
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_build_parser_uses_declared_usage():
    """Application-specific usage text is retained by the parser."""
    application = ApplicationDefinition(
        module_name="simtools.applications.test",
        description="Test application.",
        usage="simtools-test --value VALUE",
    )

    assert application.build_parser().usage == "simtools-test --value VALUE"


def test_start_delegates_to_common_startup(mocker):
    """Starting an application delegates with its parser and startup options."""
    startup = mocker.patch(
        "simtools.application.definition._initialize_runtime", return_value="context"
    )
    application = ApplicationDefinition(
        module_name="simtools.applications.test",
        description="Test application.",
        setup_io_handler=False,
    )
    mocker.patch.object(ApplicationDefinition, "_parse", return_value=({"value": 3}, {"db": 4}))

    assert application.start() == "context"
    assert startup.call_args.args == ({"value": 3}, {"db": 4})
    assert startup.call_args.kwargs == {
        "setup_io_handler": False,
        "resolve_sim_software_executables": True,
    }


def test_for_module_uses_loaded_module_documentation():
    """The module factory resolves metadata without caller-frame inspection."""
    application = ApplicationDefinition.for_module(
        __name__,
    )

    assert application.module_name == __name__
    assert application.description == __doc__


def test_for_module_uses_file_name_when_application_runs_as_script(monkeypatch, tmp_test_directory):
    """Direct execution has the same label as an installed entry point."""
    module = SimpleNamespace(
        __doc__="Test application.", __file__=str(tmp_test_directory / "example_app.py")
    )
    monkeypatch.setitem(sys.modules, "__main__", module)

    application = ApplicationDefinition.for_module("__main__")

    assert application.module_name == "simtools.applications.example_app"
    assert application.label == "example_app"


def test_post_parse_hook_receives_configuration_sources(mocker):
    """The post-parse hook receives parsed data, source tracking, and the parser."""
    initialize = mocker.patch(
        "simtools.application.definition.configurator.Configurator.configure",
        return_value=({"value": 3}, {}),
    )
    hook = Mock()
    application = ApplicationDefinition(
        module_name="simtools.applications.test",
        description="Test application.",
        arguments=(ArgumentDefinition("value", type=int),),
        post_parse=hook,
    )

    args, database = application._parse()

    assert args == {"value": 3}
    assert database == {}
    initialize.assert_called_once()
    hook.assert_called_once()
    assert hook.call_args.args[0] is args
