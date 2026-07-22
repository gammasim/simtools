"""Tests for explicit command-line argument registration."""

import astropy.units as u
import pytest
from astropy.tests.helper import assert_quantity_allclose

from simtools.configuration.arguments import (
    ENERGY_RANGE,
    PRIMARY,
    SITE,
    ArgumentDefinition,
    layout_selection_arguments,
)
from simtools.configuration.commandline_parser import CommandLineParser


def test_add_argument_definitions_registers_shared_arguments():
    """Shared argument templates retain their argparse conversion behavior."""
    parser = CommandLineParser()
    parser.add_argument_definitions(
        (
            SITE,
            PRIMARY,
            ENERGY_RANGE,
        )
    )

    args = parser.parse_args(
        ["--site", "North", "--primary", "gamma", "--energy_range", "30 GeV", "2 TeV"]
    )

    assert args.site == "North"
    assert args.primary == "gamma"
    assert_quantity_allclose(args.energy_range[0], 30 * u.GeV)
    assert_quantity_allclose(args.energy_range[1], 2 * u.TeV)


def test_add_argument_definitions_registers_mutually_exclusive_bundle():
    """Layout bundles preserve their required mutually exclusive group."""
    parser = CommandLineParser()
    parser.add_argument_definitions(layout_selection_arguments(include_file=True))

    args = parser.parse_args(["--array_layout_file", "layout.ecsv"])
    assert args.array_layout_file == ["layout.ecsv"]
    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--array_layout_name", "CTAO-North-Alpha", "--array_layout_file", "layout.ecsv"]
        )


def test_layout_selection_can_be_explicitly_optional():
    """Applications that list layouts can declare layout selection as optional."""
    parser = CommandLineParser()
    parser.add_argument_definitions(layout_selection_arguments(required=False))

    args = parser.parse_args([])

    assert args.array_layout_name is None
    assert args.array_element_list is None


def test_add_argument_definitions_rejects_conflicting_exclusive_group_state():
    """One exclusive group cannot have conflicting requiredness declarations."""
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
