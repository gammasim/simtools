"""Tests for the production_build_trigger_reference application."""

import argparse

from simtools.applications import production_build_trigger_reference


def test_add_arguments_accepts_multi_pattern_and_binning_options():
    parser = argparse.ArgumentParser()
    production_build_trigger_reference._add_arguments(parser)

    args = parser.parse_args(
        [
            "--event_data_file",
            "a*.hdf5",
            "b*.hdf5",
            "--energy_bins_per_decade",
            "8",
            "--angular_distance_bin_count",
            "42",
        ]
    )

    assert args.event_data_file == ["a*.hdf5", "b*.hdf5"]
    assert args.energy_bins_per_decade == 8
    assert args.angular_distance_bin_count == 42
