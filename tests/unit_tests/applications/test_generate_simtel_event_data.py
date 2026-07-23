"""Tests for the generate_simtel_event_data application."""

from simtools.applications import generate_simtel_event_data
from simtools.configuration.commandline_parser import CommandLineParser


def test_max_files_default_and_explicit_value():
    """Process all files by default and preserve an explicit limit."""
    parser = CommandLineParser()
    parser.add_argument_definitions(generate_simtel_event_data._ARGUMENTS)

    assert parser.parse_args(["--input", "*.simtel.zst"]).max_files is None
    assert parser.parse_args(["--input", "*.simtel.zst", "--max_files", "12"]).max_files == 12
