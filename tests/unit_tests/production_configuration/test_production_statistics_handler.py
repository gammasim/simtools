import json
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest

from simtools.production_configuration.derive_production_statistics_handler import (
    ProductionStatisticsHandler,
)


@pytest.fixture
def args_dict(tmp_path, metrics_file):
    """Fixture to provide a mock args_dict for testing."""
    return {
        "base_path": "tests/resources/production_dl2_fits/",
        "zeniths": [20, 40],
        "camera_offsets": [0.5, 1.0],
        "query_point": [1.0, 180.0, 20.0, 4.0, 0.5],
        "output_file": "production_statistics.json",
        "output_path": str(tmp_path),
        "metrics_file": str(metrics_file),
        "file_name_template": "prod6_LaPalma-{zenith}deg_gamma_cone.N.Am-4LSTs09MSTs_ID0_reduced.fits",
    }


@pytest.fixture
def metrics_file():
    """Fixture to return a metrics file."""
    return Path("tests/resources/production_simulation_config_metrics.yml")


def test_no_evaluators_initialized(args_dict, tmp_path):
    """Test behavior when no evaluators are initialized."""
    args_dict["zeniths"] = []  # No zeniths provided
    manager = ProductionStatisticsHandler(args_dict)

    manager.initialize_evaluators()

    # Check that no evaluators are initialized
    assert len(manager.evaluator_instances) == 0


def test_invalid_query_point(args_dict):
    """Test behavior with an invalid query point."""
    args_dict["query_point"] = [1, 2]  # Invalid query point
    manager = ProductionStatisticsHandler(args_dict)
    manager.initialize_evaluators()

    with pytest.raises(ValueError, match="Invalid query point format. Expected 5 values, got 2."):
        manager.perform_interpolation()


def test_missing_event_files(args_dict, tmp_path):
    """Test behavior when event files are missing."""
    args_dict["file_name_template"] = "non_existent_file.fits"
    manager = ProductionStatisticsHandler(args_dict)

    manager.initialize_evaluators()

    # Check that no evaluators are initialized due to missing files
    assert len(manager.evaluator_instances) == 0


def test_no_base_path(args_dict, tmp_path):
    """Test behavior when base_path is not provided."""
    args_dict["base_path"] = None
    manager = ProductionStatisticsHandler(args_dict)

    manager.initialize_evaluators()

    # Check that no evaluators are initialized
    assert len(manager.evaluator_instances) == 0


def test_empty_offsets(args_dict, tmp_path):
    """Test behavior when offsets are empty."""
    args_dict["camera_offsets"] = []  # Empty offsets
    manager = ProductionStatisticsHandler(args_dict)

    manager.initialize_evaluators()

    # Check that no evaluators are initialized
    assert len(manager.evaluator_instances) == 0


def test_no_query_point(args_dict, tmp_path):
    """Test behavior when query_point is missing."""
    args_dict.pop("query_point", None)  # Remove query_point
    manager = ProductionStatisticsHandler(args_dict)

    manager.initialize_evaluators()

    with pytest.raises(
        ValueError, match="Invalid query point format. Expected 5 values, got None."
    ):
        manager.perform_interpolation()


@patch("builtins.open", new_callable=mock_open)
def test_write_output(mock_open, args_dict, tmp_path):
    """Test the write_output method."""
    manager = ProductionStatisticsHandler(args_dict)

    production_statistics = np.array([10000])

    manager.write_output(production_statistics=production_statistics)

    mock_open.assert_any_call(Path(tmp_path) / args_dict["output_file"], "w", encoding="utf-8")

    handle = mock_open()
    written_data = "".join(call.args[0] for call in handle.write.call_args_list)

    expected_data = {
        "query_point": [1.0, 180.0, 20.0, 4.0, 0.5],
        "production_statistics": [10000],
    }
    assert json.loads(written_data) == expected_data


def test_run(args_dict, tmp_path):
    """Test the run method."""
    manager = ProductionStatisticsHandler(args_dict)

    manager.initialize_evaluators = MagicMock()
    manager.perform_interpolation = MagicMock()
    manager.write_output = MagicMock()

    manager.run()

    manager.initialize_evaluators.assert_called_once()
    manager.perform_interpolation.assert_called_once()
    manager.write_output.assert_called_once()


def test_perform_interpolation_not_initialized(args_dict, tmp_path):
    """Test perform_interpolation when no evaluators are initialized."""
    manager = ProductionStatisticsHandler(args_dict)

    manager.logger = MagicMock()

    manager.evaluator_instances = []

    result = manager.perform_interpolation()

    manager.logger.error.assert_called_once_with(
        "No evaluators initialized. Cannot perform interpolation."
    )
    assert result is None


@patch("simtools.production_configuration.interpolation_handler.InterpolationHandler.interpolate")
def test_perform_interpolation_with_valid_query_point(mock_interpolate, args_dict, tmp_path):
    manager = ProductionStatisticsHandler(args_dict)

    manager.evaluator_instances = [MagicMock()]

    mock_interpolate.return_value = np.array([12345])

    result = manager.perform_interpolation()

    expected_query_points = np.array([args_dict["query_point"]])

    actual_query_points = mock_interpolate.call_args[0][0]

    np.testing.assert_array_equal(actual_query_points, expected_query_points)

    assert np.array_equal(result, np.array([12345]))


def test_plot_production_statistics_comparison(args_dict, tmp_path):
    """Test the plot_production_statistics_comparison method."""
    manager = ProductionStatisticsHandler(args_dict)
    manager.interpolation_handler = MagicMock()
    mock_axes = MagicMock()
    mock_figure = MagicMock()
    manager.interpolation_handler.plot_comparison.return_value = mock_axes
    mock_axes.figure = mock_figure

    manager.plot_production_statistics_comparison()

    plot_path = Path(tmp_path) / "production_statistics_comparison.png"
    mock_figure.savefig.assert_called_once_with(plot_path)


@patch("matplotlib.pyplot.Figure.savefig")
@patch(
    "simtools.production_configuration.derive_production_statistics_handler.ProductionStatisticsHandler.initialize_evaluators"
)
@patch(
    "simtools.production_configuration.derive_production_statistics_handler.ProductionStatisticsHandler.perform_interpolation"
)
def test_run_with_plot_production_statistics(
    mock_perform_interpolation, mock_initialize_evaluators, mock_savefig, args_dict, tmp_path
):
    """Test the run method when plot_production_statistics is enabled."""
    args_dict["plot_production_statistics"] = True
    manager = ProductionStatisticsHandler(args_dict)

    mock_initialize_evaluators.return_value = None
    mock_perform_interpolation.return_value = np.array([10000])
    manager.interpolation_handler = MagicMock()
    mock_axes = MagicMock()
    mock_figure = MagicMock()
    manager.interpolation_handler.plot_comparison.return_value = mock_axes
    mock_axes.figure = mock_figure

    manager.run()

    plot_path = Path(tmp_path) / "production_statistics_comparison.png"
    mock_figure.savefig.assert_called_once_with(plot_path)

    mock_initialize_evaluators.assert_called_once()
    mock_perform_interpolation.assert_called_once()
