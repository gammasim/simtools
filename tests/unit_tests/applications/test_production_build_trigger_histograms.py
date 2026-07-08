"""Tests for the production_build_trigger_histograms application."""

import pytest

from simtools.applications import production_build_trigger_histograms
from simtools.configuration.commandline_parser import CommandLineParser


def test_add_arguments_accepts_multi_pattern_and_binning_options():
    parser = CommandLineParser()
    production_build_trigger_histograms._add_arguments(parser)

    args = parser.parse_args(
        [
            "--event_data_file",
            "a*.hdf5",
            "b*.hdf5",
            "--energy_bins_per_decade",
            "8",
            "--angular_distance_bin_width",
            "0.25 deg",
        ]
    )

    assert args.event_data_file == ["a*.hdf5", "b*.hdf5"]
    assert args.energy_bins_per_decade == 8
    assert args.angular_distance_bin_width.value == pytest.approx(0.25)


def test_add_arguments_uses_default_angular_distance_bin_width():
    parser = CommandLineParser()
    production_build_trigger_histograms._add_arguments(parser)

    args = parser.parse_args(["--event_data_file", "a*.hdf5"])

    assert args.angular_distance_bin_width.value == pytest.approx(0.5)
