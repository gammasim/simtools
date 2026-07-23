#!/usr/bin/env python3

from simtools.applications import resources_test_sync
from simtools.configuration.commandline_parser import CommandLineParser


def test_add_arguments_exclude_flags_override_defaults():
    parser = CommandLineParser()
    parser.add_argument_definitions(resources_test_sync._ARGUMENTS)

    args = parser.parse_args(
        [
            "--test_directory",
            "../simtools-tests",
            "--simtools_version",
            "v0.34.0",
            "--exclude_static",
            "--exclude_downloaded",
        ]
    )

    assert args.exclude_static is True
    assert args.exclude_generated is False
    assert args.exclude_downloaded is True
