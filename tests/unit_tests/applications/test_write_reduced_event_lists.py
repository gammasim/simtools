"""Tests for the write_reduced_event_lists application."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from simtools.applications import write_reduced_event_lists
from simtools.configuration.commandline_parser import CommandLineParser


def test_input_file_list_arguments():
    """Accept a file list and batch size."""
    parser = CommandLineParser()
    parser.add_argument_definitions(write_reduced_event_lists._ARGUMENTS)

    args = parser.parse_args(
        [
            "--input_file_list",
            "simtel_files.txt",
            "--files_per_reduced_event_file",
            "10",
        ]
    )

    assert args.input_file_list == "simtel_files.txt"
    assert args.files_per_reduced_event_file == 10
    assert args.input_files is None


def test_max_workers_option():
    """Read the maximum number of workers."""
    parser = CommandLineParser()
    parser.add_argument_definitions(write_reduced_event_lists._ARGUMENTS)

    args = parser.parse_args(["--input_files", "input.simtel.zst", "--max_workers", "3"])

    assert args.max_workers == 3


def test_input_arguments_are_mutually_exclusive():
    """Require exactly one form of input argument."""
    parser = CommandLineParser()
    parser.add_argument_definitions(write_reduced_event_lists._ARGUMENTS)

    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["--input_files", "input.simtel.zst", "--input_file_list", "inputs.txt"])


def test_main_passes_application_arguments_to_metadata_builder():
    """Pass the generated activity ID into reduced-event metadata."""
    args = {
        "input_files": ["input.simtel.zst"],
        "input_file_list": None,
        "files_per_reduced_event_file": 1,
        "max_workers": 1,
    }
    app_context = SimpleNamespace(
        args=args,
        io_handler=SimpleNamespace(get_output_directory=lambda: "output"),
    )

    with (
        patch(
            "simtools.application.definition.ApplicationDefinition.start",
            return_value=app_context,
        ),
        patch.object(
            write_reduced_event_lists.Simulator, "write_reduced_event_lists"
        ) as mock_write,
    ):
        write_reduced_event_lists.main()

    assert mock_write.call_args.kwargs["metadata_args"] is args
