import numpy as np
import pytest

import simtools.utils.general as gen
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
    return gen.collect_data_from_file("tests/resources/production_simulation_config_metrics.yml")


def test_interpolation_handler_interpolate(test_fits_file, test_fits_file_2, metric):
    """Test interpolation with the InterpolationHandler."""
    grid_point1 = (1, 180, 45, 0, 0.5)
    evaluator1 = StatisticalUncertaintyEvaluator(
        file_path=test_fits_file, metrics=metric, grid_point=grid_point1
    )
    grid_point2 = (1, 180, 60, 0, 0.5)
    evaluator2 = StatisticalUncertaintyEvaluator(
        file_path=test_fits_file_2, metrics=metric, grid_point=grid_point2
    )
    handler = InterpolationHandler([evaluator1, evaluator2], metrics=metric)
    query_point = np.array([[1, 180, 50, 0, 0.5]])
    interpolated_values = handler.interpolate(query_point)
    assert interpolated_values.shape[0] == query_point.shape[0]

    query_point = np.array([[1e-3, 180, 40, 0, 0.5]])
    interpolated_threshold = handler.interpolate_energy_threshold(query_point)
    assert isinstance(interpolated_threshold, float)


def test_interpolation_handler_build_data_array(test_fits_file, test_fits_file_2, metric):
    """Test the _build_data_array method of InterpolationHandler."""
    grid_point1 = (1, 180, 45, 0, 0.5)
    evaluator1 = StatisticalUncertaintyEvaluator(
        file_path=test_fits_file, metrics=metric, grid_point=grid_point1
    )
    grid_point2 = (1, 180, 60, 0, 0.5)
    evaluator2 = StatisticalUncertaintyEvaluator(
        file_path=test_fits_file_2, metrics=metric, grid_point=grid_point2
    )
    handler = InterpolationHandler([evaluator1, evaluator2], metrics=metric)
    data, grid_points = handler._build_data_array()
    assert data.shape[0] == grid_points.shape[0]
    assert grid_points.shape[1] == 5


def test_interpolation_handler_remove_flat_dimensions():
    """Test the _remove_flat_dimensions method of InterpolationHandler."""
    grid_points = np.array(
        [
            [1, 180, 45, 0, 0.5],
            [1, 180, 45, 0, 0.5],
            [1, 180, 60, 0, 0.5],
        ]
    )
    handler = InterpolationHandler([], metrics={})
    reduced_grid_points, non_flat_mask = handler._remove_flat_dimensions(grid_points)
    assert reduced_grid_points.shape[1] < grid_points.shape[1]
    assert np.all(non_flat_mask == [False, False, True, False, False])


def test_interpolation_handler_plot_comparison(test_fits_file, test_fits_file_2, metric):
    """Test the plot_comparison method of InterpolationHandler."""
    grid_point1 = (1, 180, 45, 0, 0.5)
    evaluator1 = StatisticalUncertaintyEvaluator(
        file_path=test_fits_file, metrics=metric, grid_point=grid_point1
    )
    grid_point2 = (1, 180, 60, 0, 0.7)
    evaluator2 = StatisticalUncertaintyEvaluator(
        file_path=test_fits_file_2, metrics=metric, grid_point=grid_point2
    )
    handler = InterpolationHandler([evaluator1, evaluator2], metrics=metric)
    query_point = np.array([[1, 180, 50, 0, 0.5]])
    handler.interpolate(query_point)

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
