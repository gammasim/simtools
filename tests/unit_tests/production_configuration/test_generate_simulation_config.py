from unittest.mock import MagicMock

import numpy as np
import pytest

import simtools.utils.general as gen
from simtools.production_configuration.calculate_statistical_errors_grid_point import (
    StatisticalErrorEvaluator,
)
from simtools.production_configuration.generate_simulation_config import SimulationConfig

PATH_FITS = (
    "tests/resources/production_dl2_fits/"
    "prod6_LaPalma-20deg_gamma_cone.N.Am-4LSTs09MSTs_ID0_reduced.fits"
)


@pytest.fixture
def metric():
    return gen.collect_data_from_file("tests/resources/production_simulation_config_metrics.yml")


@pytest.fixture
def mock_statistical_error_evaluator():
    metric_results = {
        "uncertainty_effective_area": {
            "relative_errors": np.array(
                [0.00000000e00, 1.31778332e-08, 4.74675897e-08, 1.17238332e-07]
            ),
            "uncertainties": np.array([0.01]),
        },
        "error_sig_eff_gh": 0.02,
        "energy_estimate": 0.2,
        "error_gamma_ray_psf": 0.01,
        "error_image_template_methods": 0.05,
    }

    mock_evaluator = MagicMock(spec=StatisticalErrorEvaluator)
    mock_evaluator.metric_results = metric_results
    mock_evaluator.calculate_metrics.return_value = None
    mock_evaluator.calculate_required_events = MagicMock(return_value=1e5)

    # Mock data attribute
    mock_evaluator.data = {
        "simulated_event_histogram": [100, 200, 300],
        "core_range": 1000,
        "viewcone": 10,
    }

    # Mock file loading function
    mock_evaluator.load_data_from_file = MagicMock(return_value=None)

    return mock_evaluator


def test_initialization(mock_statistical_error_evaluator):
    grid_point = {"azimuth": 0.0, "elevation": 0.0}
    config = SimulationConfig(grid_point, PATH_FITS, "point-like")
    config.evaluator = mock_statistical_error_evaluator

    assert config.grid_point == grid_point
    assert config.file_path == PATH_FITS
    assert config.file_type == "point-like"


def test_configure_simulation(mock_statistical_error_evaluator):
    grid_point = {"azimuth": 30.0, "elevation": 40.0}
    metrics = {
        "uncertainty_effective_area": {
            "target_error": {"value": 0.1, "unit": "dimensionless"},
            "energy_range": {"value": [0.04, 200], "unit": "TeV"},
        },
        "energy_estimate": {
            "target_error": {"value": 0.2, "unit": "dimensionless"},
            "energy_range": {"value": [0.04, 200], "unit": "TeV"},
        },
    }
    config = SimulationConfig(grid_point, PATH_FITS, "Off-source", metrics)
    config.evaluator = mock_statistical_error_evaluator

    params = config.configure_simulation()
    assert isinstance(params, dict)
    assert np.isclose(params.get("number_of_events").value, 1099997427622.6555, atol=1e-2)


def test_calculate_core_scatter_area(mock_statistical_error_evaluator):
    grid_point = {"azimuth": 45.0, "elevation": 60.0}
    config = SimulationConfig(grid_point, PATH_FITS, "point-like")
    config.evaluator = mock_statistical_error_evaluator

    # Mocking the method calculate_core_scatter_area
    config.calculate_core_scatter_area = MagicMock(return_value=25.0)

    core_area = config.calculate_core_scatter_area()

    assert core_area > 0


def test_calculate_viewcone(mock_statistical_error_evaluator):
    grid_point = {"azimuth": 15.0, "elevation": 25.0}
    config = SimulationConfig(grid_point, PATH_FITS, "Off-source")
    config.evaluator = mock_statistical_error_evaluator

    # Mocking the method calculate_viewcone
    config.calculate_viewcone = MagicMock(return_value={"view_angle": 45.0})

    viewcone_params = config.calculate_viewcone()

    assert isinstance(viewcone_params, dict)
    assert viewcone_params.get("view_angle") == 45.0


def test_edge_cases(mock_statistical_error_evaluator, metric):
    grid_point = {"azimuth": 0.0, "elevation": 0.0}
    config = SimulationConfig(
        grid_point=grid_point,
        file_path=PATH_FITS,
        file_type="point-like",
        metrics=metric,
    )
    config.evaluator = mock_statistical_error_evaluator

    params = config.configure_simulation()

    expected_number_of_events = 27499935690566.387
    assert np.isclose(params.get("number_of_events").value, expected_number_of_events, atol=1e0)
