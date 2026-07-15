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
            "--reduced_view_cone_radius",
            "2 deg",
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
    assert args.reduced_view_cone_radius.value == pytest.approx(2.0)
    assert args.optimization_energy_min.value == pytest.approx(0.2)
    assert args.optimization_energy_max.value == pytest.approx(20.0)
    assert args.plot_diagnostics is True


def test_add_arguments_accepts_target_triggered_events():
    parser = CommandLineParser()
    production_derive_monte_carlo_statistics._add_arguments(parser)

    args = parser.parse_args(
        [
            "--trigger_histogram_file",
            "reference.hdf5",
            "--target_triggered_events",
            "25",
        ]
    )

    assert args.target_relative_uncertainty is None
    assert args.target_triggered_events == 25


def test_add_arguments_accepts_scientific_notation_target_triggered_events():
    parser = CommandLineParser()
    production_derive_monte_carlo_statistics._add_arguments(parser)

    args = parser.parse_args(
        [
            "--trigger_histogram_file",
            "reference.hdf5",
            "--target_triggered_events",
            "1e6",
        ]
    )

    assert args.target_triggered_events == 1000000
