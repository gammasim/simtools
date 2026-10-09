"""Tests for shared command-line argument definitions."""

from simtools.configuration import arguments
from simtools.configuration.commandline_parser import CommandLineParser


def test_select_accepts_multiple_values_after_one_option():
    parser = CommandLineParser()
    parser.add_argument_definitions((arguments.SELECT,))

    args = parser.parse_args(
        [
            "--select",
            "configuration.primary=gamma",
            "configuration.site=North",
        ]
    )

    assert args.select == ["configuration.primary=gamma", "configuration.site=North"]
