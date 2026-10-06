"""Tests for explicit application definitions."""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from simtools.application.definition import ApplicationDefinition
from simtools.configuration.arguments import ArgumentDefinition


def test_argument_definition_validates_name_and_overrides():
    definition = ArgumentDefinition("input_file", type=str, required=False)
    required = definition(required=True)

    assert definition.kwargs["required"] is False
    assert required.kwargs["required"] is True
    with pytest.raises(ValueError, match="Invalid argument name"):
        ArgumentDefinition("--input_file")


def test_application_definition_rejects_duplicate_arguments():
    with pytest.raises(ValueError, match=r"Duplicate command-line argument.*input"):
        ApplicationDefinition(
            module_name="simtools.applications.test",
            description="Test application.",
            arguments=(ArgumentDefinition("input"), ArgumentDefinition("input")),
        )


def test_build_parser_registers_groups_and_exclusive_arguments():
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


def test_application_definition_can_exclude_standard_arguments():
    application = ApplicationDefinition(
        module_name="simtools.applications.test",
        description="Test application.",
        excluded_standard_arguments=("test", "ignore_existing_parameter_version"),
    )

    argument_names = {argument.name for argument in application.all_arguments}

    assert "test" not in argument_names
    assert "ignore_existing_parameter_version" not in argument_names
    assert "config" in argument_names


def test_start_delegates_to_common_startup(mocker):
    startup = mocker.patch(
        "simtools.application.control._initialize_runtime", return_value="context"
    )
    application = ApplicationDefinition(
        module_name="simtools.applications.test",
        description="Test application.",
        setup_io_handler=False,
    )
    mocker.patch.object(ApplicationDefinition, "_parse", return_value={"value": 3})

    assert application.start() == "context"
    assert startup.call_args.args == ({"value": 3},)
    assert startup.call_args.kwargs == {
        "setup_io_handler": False,
        "resolve_sim_software_executables": True,
        "validate_simulation_dependencies": False,
        "initialize_model_reader": False,
    }


def test_model_repository_definitions_initialize_the_reader_by_default():
    """Model-backed applications initialize the configured reader by default."""
    application = ApplicationDefinition(
        module_name="simtools.applications.test",
        description="Test application.",
        model_repository=True,
    )

    assert application.initialize_model_reader is True


def test_start_can_skip_model_reader_initialization(mocker):
    """Write-only applications can start without an existing model repository."""
    startup = mocker.patch(
        "simtools.application.control._initialize_runtime", return_value="context"
    )
    application = ApplicationDefinition(
        module_name="simtools.applications.test",
        description="Test application.",
        initialize_model_reader=False,
    )
    mocker.patch.object(ApplicationDefinition, "_parse", return_value={})

    assert application.start() == "context"
    assert startup.call_args.kwargs["initialize_model_reader"] is False


@pytest.mark.parametrize("arguments", [[], ["--help"], ["-h"]])
def test_help_does_not_initialize_configuration(arguments, monkeypatch, mocker):
    """Help only requires parser construction, not configuration or runtime setup."""
    configure = mocker.patch("simtools.application.definition.configurator.Configurator.configure")
    application = ApplicationDefinition(
        module_name="simtools.applications.test",
        description="Test application.",
        arguments=(ArgumentDefinition("required_value", required=True),),
    )
    monkeypatch.setattr(sys, "argv", ["application", *arguments])

    with pytest.raises(SystemExit) as exc:
        application._parse()

    assert exc.value.code == 0
    configure.assert_not_called()


def test_array_position_writer_does_not_require_model_reader():
    """The array-position writer can create a repository from an empty output directory."""
    application = importlib.import_module(
        "simtools.applications.maintain_simulation_model_write_array_element_positions"
    ).APPLICATION

    assert application.initialize_model_reader is False
    argument_names = {argument.name for argument in application.all_arguments}
    assert "output_path" in argument_names
    assert "simulation_models_path" not in argument_names


def test_array_position_writer_uses_output_path(mocker, tmp_test_directory):
    """Write generated parameters to the requested output directory."""
    module = importlib.import_module(
        "simtools.applications.maintain_simulation_model_write_array_element_positions"
    )
    arguments = {
        "array_element_positions_file": "positions.ecsv",
        "coordinate_system": "ground",
        "output_path": Path(tmp_test_directory) / "model-output",
        "parameter_version": "0.1.0",
    }
    mocker.patch.object(
        module.ApplicationDefinition, "start", return_value=SimpleNamespace(args=arguments)
    )
    write = mocker.patch.object(module, "write_array_elements_from_file_to_repository")

    module.main()

    write.assert_called_once_with(
        coordinate_system="ground",
        input_file="positions.ecsv",
        repository_path=arguments["output_path"],
        parameter_version="0.1.0",
    )


def test_for_module_uses_file_name_when_application_runs_as_script(monkeypatch, tmp_test_directory):
    module = SimpleNamespace(
        __doc__="Test application.", __file__=str(tmp_test_directory / "example_app.py")
    )
    monkeypatch.setitem(sys.modules, "__main__", module)

    application = ApplicationDefinition.for_module("__main__")

    assert application.module_name == "simtools.applications.example_app"
    assert application.label == "example_app"


def test_post_parse_hook_receives_configuration_sources(mocker, monkeypatch):
    initialize = mocker.patch(
        "simtools.application.definition.configurator.Configurator.configure",
        return_value={"value": 3},
    )
    hook = Mock()
    application = ApplicationDefinition(
        module_name="simtools.applications.test",
        description="Test application.",
        arguments=(ArgumentDefinition("value", type=int),),
        post_parse=hook,
    )
    monkeypatch.setattr(sys, "argv", ["application", "--value", "3"])

    args = application._parse()

    assert args["value"] == 3
    assert args["_metadata_configuration_sources"] == {
        "constructor": [],
        "cli": [],
        "defaults": [],
        "environment": [],
        "yaml": [],
    }
    initialize.assert_called_once()
    hook.assert_called_once()
    assert hook.call_args.args[0] is args
