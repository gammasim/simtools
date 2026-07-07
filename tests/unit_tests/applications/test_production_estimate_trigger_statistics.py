"""Tests for the production_estimate_trigger_statistics application."""

import pytest

from simtools.applications import production_estimate_trigger_statistics
from simtools.configuration.commandline_parser import CommandLineParser


def test_add_arguments_accepts_radius_override_and_energy_ranges():
    parser = CommandLineParser()
    production_estimate_trigger_statistics._add_arguments(parser)

    args = parser.parse_args(
        [
            "--input",
            "reference.hdf5",
            "--target_relative_uncertainty",
            "0.05",
            "--reduced_core_radius",
            "80 m",
            "--thrown_energy_min",
            "0.2 TeV",
            "--thrown_energy_max",
            "20 TeV",
        ]
    )

    assert args.input == "reference.hdf5"
    assert args.target_relative_uncertainty == pytest.approx(0.05)
    assert args.reduced_core_radius.value == pytest.approx(80.0)
    assert args.thrown_energy_min.value == pytest.approx(0.2)
    assert args.thrown_energy_max.value == pytest.approx(20.0)
