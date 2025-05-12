from unittest.mock import patch

import astropy.units as u
import numpy as np
import pytest

import simtools.utils.general as gen
from simtools.production_configuration.calculate_statistical_uncertainties_grid_point import (
    StatisticalUncertaintyEvaluator,
)
from simtools.production_configuration.derive_production_statistics import (
    ProductionStatisticsDerivator,
)


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


def test_initialization(test_fits_file, metric):
    """Test the initialization of the StatisticalUncertaintyEvaluator."""
    evaluator = StatisticalUncertaintyEvaluator(file_path=test_fits_file, metrics=metric)
    assert isinstance(evaluator.data, dict)
    assert "event_energies_reco" in evaluator.data


def test_calculate_uncertainty_effective_area(test_fits_file, metric):
    """Test the calculation of effective area uncertainty."""
    evaluator = StatisticalUncertaintyEvaluator(file_path=test_fits_file, metrics=metric)
    evaluator.calculate_metrics()
    uncertainties = evaluator.calculate_uncertainty_effective_area()
    assert "relative_uncertainties" in uncertainties
    assert len(uncertainties["relative_uncertainties"]) > 0


def test_calculate_energy_estimate(test_fits_file, metric):
    """Test the calculation of energy estimate uncertainty."""
    evaluator = StatisticalUncertaintyEvaluator(file_path=test_fits_file, metrics=metric)
    evaluator.calculate_metrics()
    uncertainty, sigma, delta = evaluator.calculate_energy_estimate()
    assert isinstance(sigma, list)
    assert isinstance(delta, list)


def test_missing_file():
    """Test initialization with a missing file."""
    file_path = "nonexistent_file.fits"
    metrics = {
        "uncertainty_effective_area": {
            "target_uncertainty": {"value": 0.1, "unit": "dimensionless"}
        }
    }

    with pytest.raises(FileNotFoundError, match=f"Error loading file {file_path}:"):
        StatisticalUncertaintyEvaluator(file_path, metrics)


def test_calculate_production_statistics(test_fits_file, metric):
    """Test the calculation of production statistics for a specific grid point using ProductionStatisticsHandler."""

    evaluator = StatisticalUncertaintyEvaluator(file_path=test_fits_file, metrics=metric)
    evaluator.grid_point = (1.5, 180, 45, 0, 0.5)

    derive_production_statistics = ProductionStatisticsDerivator(evaluator, metrics=metric)

    production_statistics = derive_production_statistics.derive_statistics()

    assert isinstance(production_statistics, u.Quantity)
    assert production_statistics.value == pytest.approx(41249903535849.58, rel=1e-0)
    assert production_statistics.unit == u.ct


def test_calculate_metrics(test_fits_file, metric):
    """Test the calculation of metrics."""
    evaluator = StatisticalUncertaintyEvaluator(file_path=test_fits_file, metrics=metric)

    evaluator.calculate_energy_estimate = lambda: (
        0.33,
        [0.1, 0.2],
        [0.01, 0.02],
    )

    evaluator.calculate_metrics()

    expected_values = np.array([0.40824829, 0.31622776, 0.1796053])
    computed_values = evaluator.uncertainty_effective_area["relative_uncertainties"].value[
        : len(expected_values)
    ]
    assert computed_values == pytest.approx(expected_values, rel=1e-2)

    assert evaluator.energy_estimate == pytest.approx(0.33, rel=1e-2)

    expected_results = {
        "uncertainty_effective_area": evaluator.uncertainty_effective_area,
        "energy_estimate": evaluator.energy_estimate,
    }
    assert evaluator.metric_results == expected_results


@pytest.fixture
def setup_evaluator(metric):
    file_path = "path_to_fits_file"
    grid_point = (1.0, 45.0, 30.0, 0.1, 0.05)

    evaluator = StatisticalUncertaintyEvaluator(file_path, metrics=metric, grid_point=grid_point)

    evaluator.metric_results = {
        "uncertainty_effective_area": {"relative_uncertainties": np.array([0.04, 0.05, 0.06])},
        "uncertainty_sig_eff_gh": 0.02,
        "energy_estimate": 0.03,
        "uncertainty_gamma_ray_psf": 0.01,
        "uncertainty_image_template_methods": 0.04,
    }

    return evaluator


def test_calculate_overall_metric_average(test_fits_file):
    evaluator = StatisticalUncertaintyEvaluator(
        file_path=test_fits_file,
        metrics={
            "uncertainty_effective_area": {
                "target_uncertainty": {"value": 0.1, "unit": "dimensionless"}
            }
        },
    )
    evaluator.data = {"metric_values": np.array([0.1, 0.2, 0.3, 0.4])}
    evaluator.metric_results = {
        "uncertainty_effective_area": {"relative_uncertainties": np.array([0.1, 0.2, 0.3, 0.4])}
    }
    overall_metric = evaluator.calculate_overall_metric(metric="average")
    expected_metric = 0.4

    assert np.isclose(overall_metric, expected_metric), (
        f"Expected {expected_metric}, got {overall_metric}"
    )


def test_calculate_overall_metric_maximum(test_fits_file):
    evaluator = StatisticalUncertaintyEvaluator(
        file_path=test_fits_file,
        metrics={
            "uncertainty_effective_area": {
                "target_uncertainty": {"value": 0.1, "unit": "dimensionless"}
            }
        },
    )
    evaluator.data = {"metric_values": np.array([0.1, 0.2, 0.3, 0.4])}
    evaluator.metric_results = {
        "uncertainty_effective_area": {"relative_uncertainties": np.array([0.1, 0.2, 0.3, 0.4])}
    }
    overall_metric = evaluator.calculate_overall_metric(metric="maximum")
    expected_metric = (
        0.4  # max and average are the same in this case since there is only one metric
    )

    assert np.isclose(overall_metric, expected_metric), (
        f"Expected {expected_metric}, got {overall_metric}"
    )


def test_create_bin_edges(test_fits_file, metric):
    """Test the creation of unique energy bin edges."""
    evaluator = StatisticalUncertaintyEvaluator(file_path=test_fits_file, metrics=metric)

    evaluator.data = {
        "bin_edges_low": np.array([1.0, 2.0, 3.0]),
        "bin_edges_high": np.array([2.0, 3.0, 4.0]),
    }

    bin_edges = evaluator.create_bin_edges()
    expected_bin_edges = np.array([1.0, 2.0, 3.0, 4.0])

    assert isinstance(bin_edges, np.ndarray)
    assert np.array_equal(bin_edges, expected_bin_edges), (
        f"Expected {expected_bin_edges}, got {bin_edges}"
    )


def test_compute_efficiency_and_uncertainties(test_fits_file, metric):
    evaluator = StatisticalUncertaintyEvaluator(file_path=test_fits_file, metrics=metric)

    reconstructed_event_counts = np.array([10, 20, 5, 0]) * u.ct
    simulated_event_counts = np.array([100, 200, 50, 0]) * u.ct

    efficiencies, relative_uncertainties = evaluator.compute_efficiency_and_uncertainties(
        reconstructed_event_counts, simulated_event_counts
    )

    expected_efficiencies = np.array([0.1, 0.1, 0.1, 0.0]) * u.dimensionless_unscaled
    expected_relative_uncertainties = np.array([0.3, 0.21213203, 0.42426407, 0.0])

    assert np.allclose(efficiencies, expected_efficiencies, atol=1e-2), (
        f"Expected efficiencies {expected_efficiencies}, but got {efficiencies}"
    )
    assert np.allclose(relative_uncertainties, expected_relative_uncertainties, atol=1e-2), (
        f"Expected relative uncertainties {expected_relative_uncertainties}, but got {relative_uncertainties}"
    )

    with pytest.raises(
        ValueError, match="Reconstructed event counts exceed simulated event counts."
    ):
        evaluator.compute_efficiency_and_uncertainties(20.0, 10.0)


def test_calculate_overall_metric_invalid_metric(test_fits_file):
    evaluator = StatisticalUncertaintyEvaluator(
        file_path=test_fits_file,
        metrics={"invalid_metric": {"target_uncertainty": {"value": 0.1, "unit": "dimensionless"}}},
    )

    with pytest.raises(ValueError, match="Invalid metric specified"):
        evaluator.calculate_metrics()


def test_set_grid_point_single_azimuth_zenith(caplog):
    with patch.object(StatisticalUncertaintyEvaluator, "load_data_from_file", return_value={}):
        evaluator = StatisticalUncertaintyEvaluator(file_path="", metrics={})
        events_data = {"PNT_AZ": np.array([45.0]), "PNT_ALT": np.array([45.0])}
        evaluator._set_grid_point(events_data)
        assert evaluator.grid_point == (1 * u.TeV, 45 * u.deg, 45 * u.deg, 0, 0 * u.deg)

        events_data = {"PNT_AZ": np.array([45.0, 90.0]), "PNT_ALT": np.array([45.0])}
        with pytest.raises(ValueError, match=r"^Multiple values found for azimuth"):
            evaluator._set_grid_point(events_data)

        events_data = {"PNT_AZ": np.array([45.0]), "PNT_ALT": np.array([45.0, 60.0])}
        with pytest.raises(ValueError, match=r"^Multiple values found for azimuth"):
            evaluator._set_grid_point(events_data)

        evaluator.grid_point = (1 * u.TeV, 45 * u.deg, 45 * u.deg, 0, 0 * u.deg)
        events_data = {"PNT_AZ": np.array([90.0]), "PNT_ALT": np.array([30.0])}
        evaluator._set_grid_point(events_data)
        assert evaluator.grid_point == (1 * u.TeV, 90 * u.deg, 60 * u.deg, 0, 0 * u.deg)
