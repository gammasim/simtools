"""Tests for the inspect_file application."""

from types import SimpleNamespace
from unittest.mock import patch

from simtools.applications import inspect_file
from simtools.configuration.commandline_parser import CommandLineParser


def test_add_arguments_parses_required_input_and_default_limit():
    parser = CommandLineParser()
    inspect_file._add_arguments(parser)

    args = parser.parse_args(["--input_file", "test.hdf5"])

    assert args.input_file == "test.hdf5"
    assert args.max_entries == 50


def test_main_builds_application_and_prints_report():
    app_context = SimpleNamespace(args={"input_file": "test.hdf5", "max_entries": 25})

    with (
        patch(
            "simtools.applications.inspect_file.build_application",
            return_value=app_context,
        ) as mock_build,
        patch("simtools.applications.inspect_file.inspect_file") as mock_inspect,
        patch(
            "simtools.applications.inspect_file.format_inspection_report",
            return_value="report output",
        ) as mock_format,
        patch(
            "simtools.applications.inspect_file.inspect_trigger_histogram_file",
            return_value={"inspector": "trigger_histogram"},
        ) as mock_specialized_inspect,
        patch(
            "simtools.applications.inspect_file.format_trigger_histogram_inspection",
            return_value="specialized output",
        ) as mock_specialized_format,
        patch("builtins.print") as mock_print,
    ):
        inspect_file.main()

    mock_build.assert_called_once_with(
        initialization_kwargs={"db_config": False},
        startup_kwargs={"setup_io_handler": False},
    )
    mock_inspect.assert_called_once_with("test.hdf5", max_entries=25)
    mock_format.assert_called_once()
    mock_specialized_inspect.assert_called_once_with("test.hdf5")
    mock_specialized_format.assert_called_once_with({"inspector": "trigger_histogram"})
    mock_print.assert_called_once_with("report output\nspecialized output")
