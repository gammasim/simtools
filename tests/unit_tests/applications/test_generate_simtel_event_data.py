"""Tests for the generate_simtel_event_data application."""

import pytest

from simtools.applications import generate_simtel_event_data
from simtools.configuration.commandline_parser import CommandLineParser


def test_max_files_default_and_explicit_value():
    """Process all files by default and preserve an explicit limit."""
    parser = CommandLineParser()
    parser.add_argument_definitions(generate_simtel_event_data._ARGUMENTS)

    assert parser.parse_args(["--simtel_file", "*.simtel.zst"]).max_files is None
    assert parser.parse_args(["--simtel_file", "*.simtel.zst", "--max_files", "12"]).max_files == 12


def test_main_fails_when_input_glob_matches_no_files(mocker):
    context = mocker.Mock()
    context.args = {"simtel_file": "missing/*.simtel.zst"}
    application = mocker.Mock()
    application.start.return_value = context
    mocker.patch.object(generate_simtel_event_data, "APPLICATION", application)

    with pytest.raises(FileNotFoundError, match="No matching input files"):
        generate_simtel_event_data.main()
