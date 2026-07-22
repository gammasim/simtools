"""Tests for explicit application definitions."""

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


def test_application_definition_rejects_duplicate_arguments():
    """Duplicate option names fail when the application is declared."""
    with pytest.raises(ValueError, match=r"Duplicate command-line argument.*input"):
        ApplicationDefinition(
            module_name="simtools.applications.test",
            description="Test application.",
            arguments=(ArgumentDefinition("input"), ArgumentDefinition("input")),
            include_standard_arguments=False,
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
        include_standard_arguments=False,
    )

    parser = application.build_parser()
    assert parser.parse_args(["--file", "events.simtel.zst", "--value", "2"]).value == 2
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_start_delegates_to_common_startup(mocker):
    """Starting an application delegates with its parser and startup options."""
    startup = mocker.patch(
        "simtools.application.definition.startup_application", return_value="context"
    )
    application = ApplicationDefinition(
        module_name="simtools.applications.test",
        description="Test application.",
        include_standard_arguments=False,
        setup_io_handler=False,
    )

    assert application.start() == "context"
    parse_function = startup.call_args.args[0]
    assert parse_function == application._parse
    assert startup.call_args.kwargs == {
        "setup_io_handler": False,
        "logger_name": None,
        "resolve_sim_software_executables": True,
    }


def test_for_module_uses_loaded_module_documentation():
    """The module factory resolves metadata without caller-frame inspection."""
    application = ApplicationDefinition.for_module(
        __name__,
        include_standard_arguments=False,
    )

    assert application.module_name == __name__
    assert application.description == __doc__


def test_post_parse_hook_receives_configuration_sources(mocker):
    """The post-parse hook receives parsed data, source tracking, and the parser."""
    initialize = mocker.patch(
        "simtools.application.definition.configurator.Configurator.initialize_preconfigured",
        return_value=({"value": 3}, {}),
    )
    hook = Mock()
    application = ApplicationDefinition(
        module_name="simtools.applications.test",
        description="Test application.",
        arguments=(ArgumentDefinition("value", type=int),),
        include_standard_arguments=False,
        post_parse=hook,
    )

    args, database = application._parse()

    assert args == {"value": 3}
    assert database == {}
    initialize.assert_called_once()
    hook.assert_called_once()
    assert hook.call_args.args[0] is args
