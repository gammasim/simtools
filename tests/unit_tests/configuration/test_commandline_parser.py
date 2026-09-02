"""Tests for explicit command-line argument registration."""

import pytest

from simtools.configuration.arguments import (
    FIGURE_DPI,
    ArgumentDefinition,
)
from simtools.configuration.commandline_parser import CommandLineParser


def test_add_argument_definitions_rejects_conflicting_exclusive_group_state():
    parser = CommandLineParser()
    with pytest.raises(ValueError, match="Conflicting required state"):
        parser.add_argument_definitions(
            (
                ArgumentDefinition(
                    "first", exclusive_group="source", exclusive_group_required=True
                ),
                ArgumentDefinition(
                    "second", exclusive_group="source", exclusive_group_required=False
                ),
            )
        )


def test_figure_dpi_defaults_and_accepts_cli_value():
    parser = CommandLineParser()
    parser.add_argument_definitions((FIGURE_DPI,))

    assert parser.parse_args([]).figure_dpi == 300
    assert parser.parse_args(["--figure_dpi", "123"]).figure_dpi == 123
