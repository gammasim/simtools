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


def test_parser_accepts_trigger_histogram_directory():
    args = production_derive_corsika_limits.APPLICATION.build_parser().parse_args(
        [
            "--trigger_histogram_directory",
            "/data/trigger_histograms",
            "--allowed_losses",
            "all,0.001,10",
        ]
    )

    assert args.trigger_histogram_directory == "/data/trigger_histograms"
    assert args.trigger_histogram_file is None


def test_parser_accepts_selected_plot_layouts():
    args = production_derive_corsika_limits.APPLICATION.build_parser().parse_args(
        [
            "--trigger_histogram_file",
            "trigger_histograms.hdf5",
            "--allowed_losses",
            "all,0.001,10",
            "--plot_histograms",
            "CTAO-North-Alpha",
            "CTAO-North-Beta",
        ]
    )

    assert args.plot_histograms == ["CTAO-North-Alpha", "CTAO-North-Beta"]


def test_parser_accepts_bare_plot_histograms_flag_as_all_layouts():
    args = production_derive_corsika_limits.APPLICATION.build_parser().parse_args(
        [
            "--trigger_histogram_file",
            "trigger_histograms.hdf5",
            "--allowed_losses",
            "all,0.001,10",
            "--plot_histograms",
        ]
    )

    assert args.plot_histograms == []


def test_parser_accepts_reduced_histogram_plot_flag():
    args = production_derive_corsika_limits.APPLICATION.build_parser().parse_args(
        [
            "--trigger_histogram_file",
            "trigger_histograms.hdf5",
            "--allowed_losses",
            "all,0.001,10",
            "--plot_histograms",
            "CTAO-North-Alpha",
            "--plot_reduced_histograms",
        ]
    )

    assert args.plot_histograms == ["CTAO-North-Alpha"]
    assert args.plot_reduced_histograms is True


def test_parser_rejects_output_file_with_trigger_histogram_directory(mocker):
    parser = mocker.Mock()

    production_derive_corsika_limits._post_parse(
        {
            "trigger_histogram_directory": "/data/trigger_histograms",
            "output_file": "limits.ecsv",
            "output_file_from_default": False,
            "allowed_losses": ["all,0.001,10"],
        },
        {},
        parser,
    )

    parser.error.assert_called_once_with(
        "--output_file cannot be used with --trigger_histogram_directory."
    )


def test_parser_accepts_default_output_file_with_trigger_histogram_directory(mocker):
    parser = mocker.Mock()

    production_derive_corsika_limits._post_parse(
        {
            "trigger_histogram_directory": "/data/trigger_histograms",
            "output_file": "activity-id-simtools-production-derive-corsika-limits.ecsv",
            "output_file_from_default": True,
            "allowed_losses": ["all,0.001,10"],
        },
        {},
        parser,
    )

    parser.error.assert_not_called()


def test_parser_rejects_negative_differential_loss_bins(capsys):
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
    assert (
        "differential_loss_bins_per_decade must be a non-negative integer"
        in capsys.readouterr().err
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
