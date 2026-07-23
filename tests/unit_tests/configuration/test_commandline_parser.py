"""Tests for explicit command-line argument registration."""

import astropy.units as u
import pytest
from astropy.tests.helper import assert_quantity_allclose

from simtools.configuration.arguments import (
    ARRAY_LAYOUT_NAME,
    ENERGY_RANGE,
    PRIMARY,
    SIMULATION_MODELS_PATH,
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


def test_add_argument_definitions_records_by_version_arguments():
    """Only declarations supporting unresolved version mappings are recorded."""
    parser = CommandLineParser()
    parser.add_argument_definitions((ARRAY_LAYOUT_NAME, SITE))

    assert parser.preserve_by_version == {"array_layout_name"}


def test_add_argument_definitions_reuses_display_groups():
    """Arguments sharing a display group create one argparse group."""
    parser = CommandLineParser()
    parser.add_argument_definitions((SITE, ARRAY_LAYOUT_NAME))

    groups = [group for group in parser._action_groups if group.title == "simulation model"]
    assert len(groups) == 1
    assert {action.dest for action in groups[0]._group_actions} == {
        "array_layout_name",
        "site",
    }


def test_add_argument_definitions_registers_simulation_models_path(tmp_test_directory):
    """The file-backed model source is available as a database argument."""
    parser = CommandLineParser()
    parser.add_argument_definitions((SIMULATION_MODELS_PATH,))

    args = parser.parse_args(["--simulation_models_path", str(tmp_test_directory)])

    assert args.simulation_models_path == tmp_test_directory


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


def test_layout_selection_supports_parameter_files():
    """A model-parameter file can satisfy the required layout selection."""
    parser = CommandLineParser()
    parser.add_argument_definitions(layout_selection_arguments(include_parameter_file=True))

    args = parser.parse_args(["--array_layout_parameter_file", "layouts.json"])

    assert args.array_layout_parameter_file == "layouts.json"
    assert parser.preserve_by_version == {"array_layout_name"}


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
