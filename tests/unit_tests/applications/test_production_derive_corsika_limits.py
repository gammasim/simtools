"""Tests for the production_derive_corsika_limits application."""

import pytest

from simtools.applications import production_derive_corsika_limits


def test_parser_accepts_compact_loss_configuration_and_array_selection():
    args = production_derive_corsika_limits.APPLICATION.build_parser().parse_args(
        [
            "--trigger_histogram_file",
            "trigger_histograms.hdf5",
            "--array_layout_name",
            "CTAO-North-Alpha",
            "CTAO-North-Beta",
            "--allowed_losses",
            "all,0.001,10",
            "--differential_loss_bins_per_decade",
            "5",
            "--output_file",
            "limits.ecsv",
            "--output_file_format",
            "ecsv",
        ]
    )

    assert args.trigger_histogram_file == "trigger_histograms.hdf5"
    assert args.array_layout_name == ["CTAO-North-Alpha", "CTAO-North-Beta"]
    assert args.allowed_losses == ["all,0.001,10"]
    assert args.differential_loss_bins_per_decade == 5
    assert args.output_file == "limits.ecsv"
    assert args.output_file_format == "ecsv"


def test_parser_rejects_negative_differential_loss_bins():
    with pytest.raises(SystemExit):
        production_derive_corsika_limits.APPLICATION.build_parser().parse_args(
            [
                "--trigger_histogram_file",
                "trigger_histograms.hdf5",
                "--allowed_losses",
                "all,0.001,10",
                "--differential_loss_bins_per_decade",
                "-1",
            ]
        )


def test_post_parse_rejects_invalid_allowed_loss_configuration(mocker):
    parser = mocker.Mock()

    production_derive_corsika_limits._post_parse(
        {"allowed_losses": ["core_distance,1.1,10", "angular_distance,0.1,10"]},
        {},
        parser,
    )

    parser.error.assert_called_once_with(
        "Invalid --allowed_losses value 'core_distance,1.1,10': "
        "fraction must be finite and in the interval [0, 1]"
    )
