#!/usr/bin/env python3

import argparse

from simtools.applications import resources_test_sync


def test_add_arguments_exclude_flags_override_defaults():
    parser = argparse.ArgumentParser()
    for parameter, definition in resources_test_sync._APPLICATION_ARG_DEFINITIONS.items():
        parser.add_argument(f"--{parameter}", **definition)

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
