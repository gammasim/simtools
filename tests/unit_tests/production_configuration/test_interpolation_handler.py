from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
import pytest

from simtools.io_operations import ascii_handler
from simtools.production_configuration.calculate_statistical_uncertainties_grid_point import (
    StatisticalUncertaintyEvaluator,
)
from simtools.production_configuration.interpolation_handler import InterpolationHandler


@pytest.fixture
def test_fits_file():
    return (
        "tests/resources/production_dl2_fits/"
        "prod6_LaPalma-20deg_gamma_cone.N.Am-4LSTs09MSTs_ID0_reduced.fits"
    )


@pytest.fixture
def test_fits_file_2():
    return (
        "tests/resources/production_dl2_fits/"
        "prod6_LaPalma-40deg_gamma_cone.N.Am-4LSTs09MSTs_ID0_reduced.fits"
    )


@pytest.fixture
def metric():
    return ascii_handler.collect_data_from_file(
        "tests/resources/production_simulation_config_metrics.yml"
    )


@pytest.fixture
def handler(test_fits_file, test_fits_file_2, metric):
    """Create a basic InterpolationHandler for testing."""
    grid_point1 = (1, 180, 45, 0, 0.5)
    evaluator1 = StatisticalUncertaintyEvaluator(
        file_path=test_fits_file, metrics=metric, grid_point=grid_point1
    )
    grid_point2 = (1, 180, 60, 0, 0.5)
    evaluator2 = StatisticalUncertaintyEvaluator(
        file_path=test_fits_file_2, metrics=metric, grid_point=grid_point2
    )
    evaluator1.calculate_metrics()
    evaluator2.calculate_metrics()
    # Sample grid points production
    grid_points_production = [
        {
            "azimuth": {"value": 180, "unit": "deg"},
            "zenith_angle": {"value": 50, "unit": "deg"},
            "nsb": {"value": 0, "unit": "MHz"},
            "offset": {"value": 0.5, "unit": "deg"},
        }
    ]

    return InterpolationHandler(
        [evaluator1, evaluator2], metrics=metric, grid_points_production=grid_points_production
    )


def test_interpolation_handler_interpolate(handler):
    """Test interpolation with the InterpolationHandler."""
    interpolated_values = handler.interpolate()
    assert interpolated_values is not None
    assert len(interpolated_values) == len(handler.grid_points_production)


def test_interpolation_handler_build_data_array(handler):
    """Test the _build_data_array method of InterpolationHandler."""
    data, grid_points = handler._build_data_array()
    assert data.shape[0] == grid_points.shape[0]
    assert grid_points.shape[1] == 5  # energy, az, zen, nsb, offset


def test_interpolation_handler_remove_flat_dimensions():
    """Test the _remove_flat_dimensions method of InterpolationHandler."""
    grid_points = np.array(
        [
            [1, 180, 45, 0, 0.5],
            [1, 180, 45, 0, 0.5],
            [1, 180, 60, 0, 0.5],
        ]
    )
    handler = InterpolationHandler([], metrics={}, grid_points_production=[])
    reduced_grid_points, non_flat_mask = handler._remove_flat_dimensions(grid_points)
    assert reduced_grid_points.shape[1] < grid_points.shape[1]
    assert np.all(non_flat_mask == [False, False, True, False, False])


def test_interpolation_handler_plot_comparison(handler):
    """Test the plot_comparison method of InterpolationHandler."""
    handler.interpolate()

    # Default grid point index
    ax = handler.plot_comparison()
    assert ax is not None
    assert ax.get_title() == "Comparison of Interpolated and Reconstructed Events"
    assert ax.get_xlabel() == "Energy (TeV)"
    assert ax.get_ylabel() == "Event Count"

    lines = ax.get_lines()
    assert len(lines) == 2

    line_labels = [line.get_label() for line in lines]
    assert "Interpolated Production Statistics" in line_labels
    assert "Reconstructed Events" in line_labels

    plt.close()


def test_interpolation_handler_empty_evaluators():
    """Test behavior with empty evaluators list."""
    handler = InterpolationHandler([], metrics={}, grid_points_production=[])

    # Test that methods handle empty evaluators gracefully
    data, grid_points = handler._build_data_array()
    assert data.shape[0] == 0
    assert grid_points.shape[0] == 0

    # Test interpolate with empty evaluators
    result = handler.interpolate()
    assert isinstance(result, np.ndarray)
    assert result.size == 0

    # Test plot_comparison with empty evaluators
    ax = handler.plot_comparison()
    assert ax is not None
    plt.close()


def test_prepare_energy_independent_data(handler):
    """Test the _prepare_energy_independent_data method."""
    production_statistic, grid_points_no_energy = handler._prepare_energy_independent_data()

    assert isinstance(production_statistic, np.ndarray)
    assert isinstance(grid_points_no_energy, np.ndarray)
    assert production_statistic.dtype == float
    assert hasattr(handler, "_non_flat_mask")
    assert handler._non_flat_mask is not None


def test_prepare_production_grid_points(handler):
    """Test the _prepare_production_grid_points method."""
    handler._prepare_energy_independent_data()

    production_grid_points = handler._prepare_production_grid_points()
    assert isinstance(production_grid_points, np.ndarray)
    assert production_grid_points.ndim == 2
    assert production_grid_points.shape[0] == len(handler.grid_points_production)


def test_perform_interpolation(handler):
    """Test the _perform_interpolation method."""
    grid_points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    values = np.array([10, 20, 15, 25])
    query_points = np.array([[0.5, 0.5], [0.2, 0.7]])

    interpolated = handler._perform_interpolation(grid_points, values, query_points)
    assert isinstance(interpolated, np.ndarray)
    assert interpolated.shape[0] == query_points.shape[0]


def test_perform_interpolation_with_energy(handler):
    """Test the _perform_interpolation_with_energy method."""
    handler.interpolate()

    energy_dependent_results = handler._perform_interpolation_with_energy()
    assert isinstance(energy_dependent_results, np.ndarray)
    assert energy_dependent_results.ndim == 3  # [1, energy_bins, production_points]


def test_build_grid_points_no_energy(handler):
    """Test the build_grid_points_no_energy method."""
    data_list, grid_points = handler.build_grid_points_no_energy()

    assert len(data_list) == len(handler.evaluators)
    assert grid_points.shape[0] == len(handler.evaluators)
    assert grid_points.shape[1] == 4  # az, zen, nsb, offset


def test_handling_of_nonuniform_energy_grids(test_fits_file, test_fits_file_2, metric):
    """Test handling of non-uniform energy grids."""
    grid_point1 = (1, 180, 45, 0, 0.5)
    evaluator1 = StatisticalUncertaintyEvaluator(
        file_path=test_fits_file, metrics=metric, grid_point=grid_point1
    )
    grid_point2 = (1, 180, 60, 0, 0.5)
    evaluator2 = StatisticalUncertaintyEvaluator(
        file_path=test_fits_file_2, metrics=metric, grid_point=grid_point2
    )
    evaluator1.calculate_metrics()
    evaluator2.calculate_metrics()
    # Modify one evaluator's energy grid to force them to be different
    # This is just for testing - in real use they might naturally be different
    if np.array_equal(evaluator1.data["bin_edges_low"], evaluator2.data["bin_edges_low"]):
        evaluator2.data["bin_edges_low"] = evaluator2.data["bin_edges_low"] * 1.01
        evaluator2.data["bin_edges_high"] = evaluator2.data["bin_edges_high"] * 1.01

    grid_points_production = [
        {
            "azimuth": {"value": 180, "unit": "deg"},
            "zenith_angle": {"value": 50, "unit": "deg"},
            "nsb": {"value": 0, "unit": "MHz"},
            "offset": {"value": 0.5, "unit": "deg"},
        }
    ]

    handler = InterpolationHandler(
        [evaluator1, evaluator2], metrics=metric, grid_points_production=grid_points_production
    )

    result = handler.interpolate()
    assert result is not None

    ax = handler.plot_comparison()
    assert ax is not None
    plt.close()


def test_plot_comparison_with_grid_point_index(handler):
    """Test plot_comparison with specific grid point index."""
    handler.interpolate()

    ax = handler.plot_comparison(grid_point_index=0)
    assert ax is not None
    plt.close()

    ax = handler.plot_comparison(grid_point_index=999)
    assert ax is not None
    plt.close()


def test_flat_dimensions_edge_case():
    """Test edge case where all dimensions are flat."""
    grid_points = np.array(
        [
            [1, 180, 45, 0, 0.5],
            [1, 180, 45, 0, 0.5],
            [1, 180, 45, 0, 0.5],
        ]
    )
    handler = InterpolationHandler([], metrics={}, grid_points_production=[])
    reduced_grid_points, non_flat_mask = handler._remove_flat_dimensions(grid_points)

    # Should keep all dimensions when all are flat
    assert reduced_grid_points.shape[1] == grid_points.shape[1]
    assert np.all(non_flat_mask)


def test_empty_grid_points():
    """Test handling of empty grid points."""
    grid_points = np.array([])
    handler = InterpolationHandler([], metrics={}, grid_points_production=[])

    reduced_grid_points, non_flat_mask = handler._remove_flat_dimensions(grid_points.reshape(0, 0))
    assert reduced_grid_points.size == 0
    assert non_flat_mask.size == 0


def test_build_grid_points_no_energy_empty_evaluators():
    """Test build_grid_points_no_energy method with empty evaluators list."""
    handler = InterpolationHandler(evaluators=[], metrics={}, grid_points_production=[])

    handler._logger = MagicMock()

    data, grid_points = handler.build_grid_points_no_energy()

    handler._logger.error.assert_called_once_with(
        "No evaluators available for grid point building."
    )

    assert isinstance(data, np.ndarray)
    assert isinstance(grid_points, np.ndarray)
    assert data.size == 0
    assert grid_points.size == 0
    assert len(data) == 0
    assert len(grid_points) == 0
