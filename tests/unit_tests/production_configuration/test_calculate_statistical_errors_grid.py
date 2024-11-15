import astropy.units as u
import numpy as np
import pytest

from simtools.production_configuration.calculate_statistical_errors_grid_point import (
    StatisticalErrorEvaluator,
)
from simtools.production_configuration.event_scaler import EventScaler
from simtools.production_configuration.interpolation_handler import InterpolationHandler
from simtools.production_configuration.production_configuration_helper_functions import load_metrics


@pytest.fixture
def test_fits_file():
    return "tests/resources/production_dl2_fits/prod6_LaPalma-20deg_gamma_cone.N.Am-4LSTs09MSTs_ID0_reduced.fits"


@pytest.fixture
def test_fits_file2():
    return "tests/resources/production_dl2_fits/prod6_LaPalma-40deg_gamma_cone.N.Am-4LSTs09MSTs_ID0_reduced.fits"


@pytest.fixture
def metric():
    return load_metrics("tests/resources/production_simulation_config_metrics.yaml")


def test_initialization(test_fits_file, metric):
    """Test the initialization of the StatisticalErrorEvaluator."""
    evaluator = StatisticalErrorEvaluator(
        file_path=test_fits_file, file_type="On-source", metrics=metric
    )
    assert evaluator.file_path == test_fits_file
    assert evaluator.file_type == "On-source"
    assert isinstance(evaluator.data, dict)
    assert "event_energies_reco" in evaluator.data


def test_calculate_error_eff_area(test_fits_file, metric):
    """Test the calculation of effective area error."""
    evaluator = StatisticalErrorEvaluator(
        file_path=test_fits_file, file_type="On-source", metrics=metric
    )
    evaluator.calculate_metrics()
    errors = evaluator.calculate_error_eff_area()
    assert "relative_errors" in errors
    assert len(errors["relative_errors"]) > 0


def test_calculate_error_energy_estimate_bdt_reg_tree(test_fits_file, metric):
    """Test the calculation of energy estimate error."""
    evaluator = StatisticalErrorEvaluator(
        file_path=test_fits_file, file_type="On-source", metrics=metric
    )
    evaluator.calculate_metrics()
    error, sigma, delta = evaluator.calculate_error_energy_estimate_bdt_reg_tree()
    assert isinstance(sigma, list)
    assert isinstance(delta, list)


def test_missing_file():
    """Test initialization with a missing file."""
    file_path = "nonexistent_file.fits"
    file_type = "On-source"
    metrics = {"error_eff_area": {"target_error": {"value": 0.1, "unit": "dimensionless"}}}

    with pytest.raises(FileNotFoundError, match=f"Error loading file {file_path}:"):
        StatisticalErrorEvaluator(file_path, file_type, metrics)


def test_interpolation_handler(test_fits_file, test_fits_file2, metric):
    """Test interpolation with the InterpolationHandler."""
    grid_point1 = (1, 180, 45, 0, 0.5)
    evaluator1 = StatisticalErrorEvaluator(
        file_path=test_fits_file, file_type="On-source", metrics=metric, grid_point=grid_point1
    )
    grid_point2 = (1, 180, 60, 0, 0.5)
    evaluator2 = StatisticalErrorEvaluator(
        file_path=test_fits_file2, file_type="On-source", metrics=metric, grid_point=grid_point2
    )
    science_case = "example case"
    handler = InterpolationHandler(
        [evaluator1, evaluator2], science_case=science_case, metrics=metric
    )
    query_point = np.array([[1, 180, 50, 0, 0.5]])
    interpolated_values = handler.interpolate(query_point)
    assert interpolated_values.shape[0] == query_point.shape[0]

    query_point = np.array([[1e-3, 180, 40, 0, 0.5]])
    interpolated_threshold = handler.interpolate_energy_threshold(query_point)
    assert isinstance(interpolated_threshold, float)


def test_calculate_scaled_events(test_fits_file, metric):
    """Test the calculation of scaled events for a specific grid point using EventScaler."""

    evaluator = StatisticalErrorEvaluator(
        file_path=test_fits_file, file_type="On-source", metrics=metric
    )
    evaluator.grid_point = (1.5, 180, 45, 0, 0.5)

    event_scaler = EventScaler(evaluator, science_case="science case 1", metrics=metric)

    scaled_events = event_scaler.scale_events()

    assert isinstance(scaled_events, u.Quantity)
    assert scaled_events.value == pytest.approx(33027, rel=1e-0)
    assert scaled_events.unit == u.ct


def test_calculate_metrics(test_fits_file, metric):
    """Test the calculation of metrics."""
    evaluator = StatisticalErrorEvaluator(
        file_path=test_fits_file, file_type="On-source", metrics=metric
    )

    evaluator.calculate_error_energy_estimate_bdt_reg_tree = lambda: (
        0.33,
        [0.1, 0.2],
        [0.01, 0.02],
    )

    evaluator.calculate_metrics()

    expected_values = np.array([3.18110350e-10, 4.65906785e-09, 1.03266065e-08])
    computed_values = evaluator.error_eff_area["relative_errors"].value[: len(expected_values)]
    assert computed_values == pytest.approx(expected_values, rel=1e-2)

    assert evaluator.error_energy_estimate_bdt_reg_tree == pytest.approx(0.33, rel=1e-2)

    expected_results = {
        "error_eff_area": evaluator.error_eff_area,
        "error_energy_estimate_bdt_reg_tree": evaluator.error_energy_estimate_bdt_reg_tree,
    }
    assert evaluator.metric_results == expected_results


@pytest.fixture
def setup_evaluator(metric):
    file_path = "path_to_fits_file"
    file_type = "On-source"
    grid_point = (1.0, 45.0, 30.0, 0.1, 0.05)

    evaluator = StatisticalErrorEvaluator(
        file_path, file_type, metrics=metric, grid_point=grid_point
    )

    evaluator.metric_results = {
        "error_eff_area": {"relative_errors": np.array([0.04, 0.05, 0.06])},
        "error_sig_eff_gh": 0.02,
        "error_energy_estimate_bdt_reg_tree": 0.03,
        "error_gamma_ray_psf": 0.01,
        "error_image_template_methods": 0.04,
    }

    return evaluator


def test_calculate_overall_metric_average():
    evaluator = StatisticalErrorEvaluator(
        file_path="tests/resources/production_dl2_fits/prod6_LaPalma-20deg_gamma_cone.N.Am-4LSTs09MSTs_ID0_reduced.fits",
        file_type="On-source",
        metrics={"error_eff_area": {"target_error": {"value": 0.1, "unit": "dimensionless"}}},
    )
    evaluator.data = {"metric_values": np.array([0.1, 0.2, 0.3, 0.4])}
    evaluator.metric_results = {
        "error_eff_area": {"relative_errors": np.array([0.1, 0.2, 0.3, 0.4])}
    }
    overall_metric = evaluator.calculate_overall_metric(metric="average")
    expected_metric = 0.4

    assert np.isclose(
        overall_metric, expected_metric
    ), f"Expected {expected_metric}, got {overall_metric}"


def test_calculate_overall_metric_maximum():
    evaluator = StatisticalErrorEvaluator(
        file_path="tests/resources/production_dl2_fits/prod6_LaPalma-20deg_gamma_cone.N.Am-4LSTs09MSTs_ID0_reduced.fits",
        file_type="On-source",
        metrics={"error_eff_area": {"target_error": {"value": 0.1, "unit": "dimensionless"}}},
    )
    evaluator.data = {"metric_values": np.array([0.1, 0.2, 0.3, 0.4])}
    evaluator.metric_results = {
        "error_eff_area": {"relative_errors": np.array([0.1, 0.2, 0.3, 0.4])}
    }
    overall_metric = evaluator.calculate_overall_metric(metric="maximum")
    expected_metric = (
        0.4  # max and average are the same in this case since there is only one metric
    )

    assert np.isclose(
        overall_metric, expected_metric
    ), f"Expected {expected_metric}, got {overall_metric}"


def test_create_bin_edges(test_fits_file, metric):
    """Test the creation of unique energy bin edges."""
    evaluator = StatisticalErrorEvaluator(
        file_path=test_fits_file, file_type="On-source", metrics=metric
    )

    evaluator.data = {
        "bin_edges_low": np.array([1.0, 2.0, 3.0]),
        "bin_edges_high": np.array([2.0, 3.0, 4.0]),
    }

    bin_edges = evaluator.create_bin_edges()
    expected_bin_edges = np.array([1.0, 2.0, 3.0, 4.0])

    assert isinstance(bin_edges, np.ndarray)
    assert np.array_equal(
        bin_edges, expected_bin_edges
    ), f"Expected {expected_bin_edges}, got {bin_edges}"


def test_compute_efficiency_and_errors(test_fits_file, metric):
    evaluator = StatisticalErrorEvaluator(
        file_path=test_fits_file, file_type="On-source", metrics=metric
    )

    triggered_event_counts = np.array([10, 20, 5, 0]) * u.ct
    simulated_event_counts = np.array([100, 200, 50, 0]) * u.ct

    efficiencies, relative_errors = evaluator.compute_efficiency_and_errors(
        triggered_event_counts, simulated_event_counts
    )

    expected_efficiencies = np.array([0.1, 0.1, 0.1, 0.0]) * u.dimensionless_unscaled
    expected_relative_errors = np.array([0.03, 0.0212132, 0.04242641, 0.0]) / u.ct**0.5

    assert np.allclose(
        efficiencies, expected_efficiencies, atol=1e-2
    ), f"Expected efficiencies {expected_efficiencies}, but got {efficiencies}"
    assert np.allclose(
        relative_errors, expected_relative_errors, atol=1e-2
    ), f"Expected relative errors {expected_relative_errors}, but got {relative_errors}"


def test_calculate_overall_metric_invalid_metric(test_fits_file):
    evaluator = StatisticalErrorEvaluator(
        file_path=test_fits_file,
        file_type="On-source",
        metrics={"invalid_metric": {"target_error": {"value": 0.1, "unit": "dimensionless"}}},
    )

    with pytest.raises(ValueError, match="Invalid metric specified"):
        evaluator.calculate_metrics()
