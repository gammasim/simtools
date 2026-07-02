"""Tests for the generate_simtel_event_data application."""

import argparse

from simtools.applications import generate_simtel_event_data


def test_max_files_default_and_explicit_value():
    """Process all files by default and preserve an explicit limit."""
    parser = argparse.ArgumentParser()
    generate_simtel_event_data._add_arguments(parser)

    assert parser.parse_args(["--input", "*.simtel.zst"]).max_files is None
    assert parser.parse_args(["--input", "*.simtel.zst", "--max_files", "12"]).max_files == 12
