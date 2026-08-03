"""Tests for the inspect_file application."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from simtools.applications import inspect_file
from simtools.configuration.commandline_parser import CommandLineParser


def test_add_arguments_parses_required_input_and_default_limit():
    parser = CommandLineParser()
    parser.add_argument_definitions(inspect_file._ARGUMENTS)

    args = parser.parse_args(["--input_file", "test.hdf5"])

    assert args.input_file == Path("test.hdf5")
    assert args.max_entries == 50
    assert args.show_entry is None


def test_add_arguments_accepts_zero_as_unlimited_limit():
    parser = CommandLineParser()
    parser.add_argument_definitions(inspect_file._ARGUMENTS)

    args = parser.parse_args(["--input_file", "test.hdf5", "--max_entries", "0"])

    assert args.max_entries == 0


def test_add_arguments_accepts_hyphenated_show_entry_alias():
    parser = CommandLineParser()
    parser.add_argument_definitions(inspect_file._ARGUMENTS)

    args = parser.parse_args(["--input_file", "test.hdf5", "--show_entry", "SIMULATION_METADATA"])

    assert args.show_entry == "SIMULATION_METADATA"


def test_main_builds_application_and_prints_report():
    app_context = SimpleNamespace(
        args={"input_file": "test.hdf5", "max_entries": 25, "show_entry": None}
    )

    with (
        patch(
            "simtools.application.definition.ApplicationDefinition.start",
            return_value=app_context,
        ) as mock_build,
        patch(
            "simtools.applications.inspect_file.inspect_file",
            return_value=["report output", "specialized output"],
        ) as mock_inspect,
        patch("builtins.print") as mock_print,
    ):
        inspect_file.main()

    mock_build.assert_called_once_with()
    assert inspect_file.APPLICATION.database is False
    assert inspect_file.APPLICATION.setup_io_handler is False
    mock_inspect.assert_called_once_with(
        "test.hdf5", max_entries=25, format_report=True, entry_name=None
    )
    mock_print.assert_called_once_with("report output\nspecialized output")


def test_main_prints_selected_hdf5_entry():
    app_context = SimpleNamespace(
        args={
            "input_file": "test.hdf5",
            "max_entries": 25,
            "show_entry": "SIMULATION_METADATA",
        }
    )

    with (
        patch(
            "simtools.application.definition.ApplicationDefinition.start",
            return_value=app_context,
        ),
        patch(
            "simtools.applications.inspect_file.inspect_file",
            return_value=["entry output"],
        ) as mock_inspect,
        patch("builtins.print") as mock_print,
    ):
        inspect_file.main()

    mock_inspect.assert_called_once_with(
        "test.hdf5", max_entries=25, format_report=True, entry_name="SIMULATION_METADATA"
    )
    mock_print.assert_called_once_with("entry output")
