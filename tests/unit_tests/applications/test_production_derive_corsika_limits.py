"""Tests for the production_derive_corsika_limits application."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from simtools.applications import production_derive_corsika_limits
from simtools.configuration.commandline_parser import CommandLineParser


def test_add_arguments_parses_allowed_losses_and_defaults():
    parser = CommandLineParser()
    production_derive_corsika_limits._add_arguments(parser)

    args = parser.parse_args(
        [
            "--trigger_histogram_file",
            "trigger_histograms.hdf5",
            "--allowed_losses",
            "all,1e-6,10",
        ]
    )

    assert args.trigger_histogram_file == "trigger_histograms.hdf5"
    assert args.allowed_losses == ["all,1e-6,10"]
    assert args.energy_threshold_fraction == pytest.approx(0.01)
    assert args.differential_loss_bins_per_decade == 0
    assert args.plot_histograms is False


def test_main_builds_application_and_generates_limits():
    app_context = SimpleNamespace(args={"trigger_histogram_file": "trigger_histograms.hdf5"})

    with (
        patch(
            "simtools.applications.production_derive_corsika_limits.build_application",
            return_value=app_context,
        ) as mock_build,
        patch(
            "simtools.applications.production_derive_corsika_limits.generate_corsika_limits_grid"
        ) as mock_generate,
    ):
        production_derive_corsika_limits.main()

    mock_build.assert_called_once_with(
        initialization_kwargs={
            "db_config": False,
            "output": True,
        }
    )
    mock_generate.assert_called_once_with(app_context.args)
