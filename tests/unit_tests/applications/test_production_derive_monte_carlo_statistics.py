"""Tests for the production_derive_monte_carlo_statistics application."""

import pytest

from simtools.applications import production_derive_monte_carlo_statistics
from simtools.configuration.commandline_parser import CommandLineParser


def test_add_arguments_accepts_radius_override_and_energy_ranges():
    parser = CommandLineParser()
    production_derive_monte_carlo_statistics._add_arguments(parser)

    args = parser.parse_args(
        [
            "--trigger_histogram_file",
            "reference.hdf5",
            "--target_relative_uncertainty",
            "0.05",
            "--reduced_core_radius",
            "80 m",
            "--optimization_energy_min",
            "0.2 TeV",
            "--optimization_energy_max",
            "20 TeV",
            "--plot_diagnostics",
        ]
    )

    assert args.trigger_histogram_file == "reference.hdf5"
    assert args.target_relative_uncertainty == pytest.approx(0.05)
    assert args.reduced_core_radius.value == pytest.approx(80.0)
    assert args.optimization_energy_min.value == pytest.approx(0.2)
    assert args.optimization_energy_max.value == pytest.approx(20.0)
    assert args.plot_diagnostics is True
