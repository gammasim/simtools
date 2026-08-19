"""Tests for the production_derive_monte_carlo_statistics application."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from simtools.applications import production_derive_monte_carlo_statistics
from simtools.configuration.commandline_parser import CommandLineParser


@patch(
    "simtools.applications.production_derive_monte_carlo_statistics.estimate_monte_carlo_statistics"
)
@patch("simtools.applications.production_derive_monte_carlo_statistics.MetadataCollector")
@patch("simtools.application.definition.ApplicationDefinition.start")
def test_main_passes_top_level_metadata_to_estimator(
    mock_start, mock_metadata_collector, mock_estimate
):
    args = {"output_file": "mc_statistics.ecsv"}
    mock_start.return_value = SimpleNamespace(args=args, io_handler=Mock())
    metadata = {"cta": {"activity": {"name": "production_derive_monte_carlo_statistics"}}}
    mock_metadata_collector.return_value.get_top_level_metadata.return_value = metadata

    production_derive_monte_carlo_statistics.main()

    mock_metadata_collector.assert_called_once_with(args)
    mock_estimate.assert_called_once_with(metadata=metadata)


def test_add_arguments_accepts_radius_override_and_energy_ranges():
    parser = CommandLineParser()
    parser.add_argument_definitions(production_derive_monte_carlo_statistics._ARGUMENTS)

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
    assert args.spectral_index is None
    assert args.reduced_core_radius.value == pytest.approx(80.0)
    assert args.reduced_view_cone_radius.value == pytest.approx(2.0)
    assert args.optimization_energy_min.value == pytest.approx(0.2)
    assert args.optimization_energy_max.value == pytest.approx(20.0)
    assert args.plot_diagnostics is True


def test_add_arguments_accepts_target_triggered_events():
    parser = CommandLineParser()
    parser.add_argument_definitions(production_derive_monte_carlo_statistics._ARGUMENTS)

    args = parser.parse_args(
        [
            "--trigger_histogram_file",
            "reference.hdf5",
            "--target_triggered_events",
            "25",
        ]
    )

    assert args.target_relative_uncertainty is None
    assert args.spectral_index is None
    assert args.target_triggered_events == 25
