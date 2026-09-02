"""Tests for the write_trigger_histograms application."""

import pytest

from simtools.applications import write_trigger_histograms
from simtools.configuration.commandline_parser import CommandLineParser


def test_add_arguments_accepts_multi_pattern_and_binning_options():
    parser = CommandLineParser()
    parser.add_argument_definitions(write_trigger_histograms._ARGUMENTS)

    args = parser.parse_args(
        [
            "--event_data_files",
            "a*.hdf5",
            "b*.hdf5",
            "--energy_bins_per_decade",
            "8",
            "--angular_distance_bin_width",
            "0.25 deg",
            "--core_distance_bin_width",
            "25 m",
        ]
    )

    assert args.event_data_files == ["a*.hdf5", "b*.hdf5"]
    assert args.energy_bins_per_decade == 8
    assert args.angular_distance_bin_width.value == pytest.approx(0.25)
    assert args.core_distance_bin_width.value == pytest.approx(25.0)
    assert args.max_workers == 1


def test_add_arguments_uses_default_angular_distance_bin_width():
    parser = CommandLineParser()
    parser.add_argument_definitions(write_trigger_histograms._ARGUMENTS)

    args = parser.parse_args(["--event_data_files", "a*.hdf5"])

    assert args.angular_distance_bin_width.value == pytest.approx(0.5)
    assert args.core_distance_bin_width.value == pytest.approx(20.0)
    assert args.minimum_triggered_telescopes == 2


def test_add_arguments_accepts_max_workers():
    parser = CommandLineParser()
    parser.add_argument_definitions(write_trigger_histograms._ARGUMENTS)

    args = parser.parse_args(["--event_data_files", "a*.hdf5", "--max_workers", "24"])

    assert args.max_workers == 24


def test_add_arguments_accepts_event_data_directory():
    parser = CommandLineParser()
    parser.add_argument_definitions(write_trigger_histograms._ARGUMENTS)

    args = parser.parse_args(["--event_data_directory", "reduced_event_data"])

    assert args.event_data_directory == "reduced_event_data"
    assert args.event_data_files is None


def test_add_arguments_accepts_production_path_and_selection():
    parser = CommandLineParser()
    parser.add_argument_definitions(write_trigger_histograms._ARGUMENTS)

    args = parser.parse_args(
        [
            "--production_path",
            "grid-output",
            "--select",
            "configuration.corsika_he_interaction=qgs3",
            "--select",
            "configuration.zenith_angle=20 deg",
        ]
    )

    assert args.production_path == "grid-output"
    assert args.select == [
        "configuration.corsika_he_interaction=qgs3",
        "configuration.zenith_angle=20 deg",
    ]


def test_post_parse_rejects_default_output_path_for_directory_mode(mocker):
    parser = mocker.Mock()

    write_trigger_histograms._post_parse(
        {
            "event_data_directory": "reduced_event_data",
            "array_layout_name": ["CTAO-North-Alpha"],
            "output_file": "write_trigger_histograms.hdf5",
            "output_file_from_default": True,
        },
        {"defaults": {"output_path"}},
        parser,
    )

    parser.error.assert_called_once_with(
        "'--output_path' is required with directory or production metadata input."
    )


@pytest.mark.parametrize("source", ["environment", "constructor", "yaml", "cli"])
def test_post_parse_accepts_explicit_output_path_for_directory_mode(mocker, source):
    parser = mocker.Mock()

    write_trigger_histograms._post_parse(
        {
            "event_data_directory": "reduced_event_data",
            "array_layout_name": ["CTAO-North-Alpha"],
            "output_file": "write_trigger_histograms.hdf5",
            "output_file_from_default": True,
        },
        {source: {"output_path"}},
        parser,
    )

    parser.error.assert_not_called()


def test_post_parse_rejects_explicit_output_file_for_directory_mode(mocker):
    parser = mocker.Mock()

    write_trigger_histograms._post_parse(
        {
            "event_data_directory": "reduced_event_data",
            "array_layout_name": ["CTAO-North-Alpha"],
            "output_file": "trigger_histograms.hdf5",
            "output_file_from_default": False,
        },
        {"cli": {"output_path"}},
        parser,
    )

    parser.error.assert_called_once_with(
        "'--output_file' cannot be used with directory or production metadata input."
    )


def test_post_parse_allows_production_metadata_without_layout_selection(mocker):
    parser = mocker.Mock()

    write_trigger_histograms._post_parse(
        {
            "production_path": "production",
            "output_file": "write_trigger_histograms.hdf5",
            "output_file_from_default": True,
        },
        {"cli": {"output_path"}},
        parser,
    )

    parser.error.assert_not_called()
