"""Tests for the write_reduced_event_lists application."""

import argparse

import pytest

from simtools.applications import write_reduced_event_lists


def test_input_file_list_arguments():
    """Accept a file list and the suggested plural batch-size alias."""
    parser = argparse.ArgumentParser()
    write_reduced_event_lists._add_arguments(parser)

    args = parser.parse_args(
        [
            "--input_file_list",
            "simtel_files.txt",
            "--files_per_reduced_events_file",
            "10",
        ]
    )

    assert args.input_file_list == "simtel_files.txt"
    assert args.files_per_reduced_event_file == 10
    assert args.input_files is None


def test_input_arguments_are_mutually_exclusive():
    """Require exactly one form of input argument."""
    parser = argparse.ArgumentParser()
    write_reduced_event_lists._add_arguments(parser)

    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["--input_files", "input.simtel.zst", "--input_file_list", "inputs.txt"])
