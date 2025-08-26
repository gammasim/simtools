"""Test module for derive_production_statistics.py."""

from unittest.mock import MagicMock, patch

import astropy.units as u
import numpy as np
import pytest

from simtools.production_configuration.derive_production_statistics import (
    ProductionStatisticsDerivator,
)


@pytest.fixture
def mock_evaluator():
    """Create a mock evaluator with all required attributes."""
    mock = MagicMock()
    mock.metric_results = {
        "uncertainty_effective_area": {"relative_uncertainties": np.array([0.1, 0.2, 0.3])}
    }
    mock.energy_bin_edges = np.array([1.0, 10.0, 100.0, 1000.0]) * u.TeV
    mock.data = {"simulated_event_histogram": np.array([1000, 2000, 3000])}
    mock.create_bin_edges.return_value = np.array([1.0, 10.0, 100.0, 1000.0]) * u.TeV
    return mock


@pytest.fixture
def mock_metrics():
    """Create mock metrics dictionary with required structure."""
    return {
        "uncertainty_effective_area": {
            "target_uncertainty": {"value": 0.05},
            "energy_range": {"value": [1.0, 1000.0], "unit": "TeV"},
        }
    }


@pytest.fixture
def derivator(mock_evaluator, mock_metrics):
    """Create a ProductionStatisticsDerivator instance with mock dependencies."""
    return ProductionStatisticsDerivator(mock_evaluator, mock_metrics)


def test_derive_statistics_debugging_output(derivator):
    """Test the debugging output in derive_statistics method."""
    with patch("builtins.print") as mock_print:
        result = derivator.derive_statistics(return_sum=True)

        assert result > 0

        if mock_print.call_count > 0:
            call_args_list = [str(call[0][0]) for call in mock_print.call_args_list]
            assert any("Base events" in arg for arg in call_args_list)
            assert any("Scaling factors shape" in arg for arg in call_args_list)
            assert any("Scaled events shape" in arg for arg in call_args_list)
            assert any("metrics" in arg for arg in call_args_list)


def test_compute_scaling_factor(derivator, mock_evaluator, mock_metrics):
    """Test the _compute_scaling_factor method."""
    scaling_factors = derivator._compute_scaling_factor()

    # Verify shape matches relative uncertainties
    assert scaling_factors.shape == (3,)

    # Verify scaling factors are calculated correctly
    # For bins with uncertainty > 0, scaling = (uncertainty/target)^2
    # Target uncertainty is 0.05, so scaling for 0.1 should be (0.1/0.05)^2 = 4
    # We need to check if the array contains values close to expected values
    expected_scaling = np.array([4.0, 16.0, 36.0])  # (0.1/0.05)^2, (0.2/0.05)^2, (0.3/0.05)^2
    np.testing.assert_almost_equal(scaling_factors, expected_scaling)


def test_derive_statistics_with_return_sum_true(derivator):
    """Test derive_statistics with return_sum=True."""
    result = derivator.derive_statistics(return_sum=True)

    # Verify result is a scalar
    assert isinstance(result, int | float | np.number)

    # Expected result: sum of base events * max scaling factor
    # Max scaling factor is 36.0 for uncertainty of 0.3
    # Base events are [1000, 2000, 3000]
    # So sum should be 36.0 * (1000 + 2000 + 3000) = 36.0 * 6000 = 216000
    expected_sum = 36.0 * 6000
    np.testing.assert_almost_equal(result, expected_sum)


def test_derive_statistics_with_return_sum_false(derivator):
    """Test derive_statistics with return_sum=False."""
    result = derivator.derive_statistics(return_sum=False)

    # Verify result is an array with the correct shape
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)

    # Expected result: base events * max scaling factor for each bin
    # Max scaling factor is 36.0 for uncertainty of 0.3
    # Base events are [1000, 2000, 3000]
    # So result should be [36000, 72000, 108000]
    expected_result = np.array([1000, 2000, 3000]) * 36.0
    np.testing.assert_almost_equal(result, expected_result)


def test_calculate_production_statistics_at_grid_point(derivator, mock_evaluator):
    """Test calculate_production_statistics_at_grid_point method."""
    # Set up a grid point with energy in the middle bin
    grid_point = (50 * u.TeV, 0, 0, 0, 0)

    # Call the method
    result = derivator.calculate_production_statistics_at_grid_point(grid_point)

    # Energy 50 TeV should fall in the second bin (10-100 TeV)
    # Base events at this bin is 2000
    # Scaling factor should be (0.2/0.05)^2 = 16.0
    # Expected result: 2000 * 16.0 = 32000
    expected_result = 2000 * 16.0
    np.testing.assert_almost_equal(result, expected_result)

    # Verify bin lookup was done correctly
    mock_evaluator.create_bin_edges.assert_called_once()


def test_calculate_production_statistics_at_invalid_grid_point(derivator):
    """Test calculate_production_statistics_at_grid_point with invalid energy."""
    # Set up a grid point with energy outside the bin range
    grid_point = (10000 * u.TeV, 0, 0, 0, 0)

    # Energy 10000 TeV is outside the bin range (1-1000 TeV)
    # Should raise ValueError
    with pytest.raises(ValueError, match="Energy .* is outside the range"):
        derivator.calculate_production_statistics_at_grid_point(grid_point)


def test_compute_scaling_factor_with_zero_uncertainties(mock_evaluator, mock_metrics):
    """Test _compute_scaling_factor with some zero uncertainties."""
    mock_evaluator.metric_results = {
        "uncertainty_effective_area": {"relative_uncertainties": np.array([0.0, 0.2, 0.0])}
    }

    derivator = ProductionStatisticsDerivator(mock_evaluator, mock_metrics)

    scaling_factors = derivator._compute_scaling_factor()

    expected_scaling = np.array([0.0, 16.0, 0.0])
    np.testing.assert_almost_equal(scaling_factors, expected_scaling)

    # Zero uncertainties should have zero scaling factors
    assert scaling_factors[0] == pytest.approx(0.0)
    assert scaling_factors[2] == pytest.approx(0.0)


def test_derive_statistics(derivator):
    """Test the derive_statistics method."""
    # Call the method with return_sum=True
    result_sum = derivator.derive_statistics(return_sum=True)

    # Verify result is a scalar and has the expected value
    assert isinstance(result_sum, int | float | np.number)
    expected_sum = 36.0 * 6000  # Max scaling factor * sum of base events
    np.testing.assert_almost_equal(result_sum, expected_sum)

    # Call the method with return_sum=False
    result_array = derivator.derive_statistics(return_sum=False)

    # Verify result is an array with the correct shape and values
    assert isinstance(result_array, np.ndarray)
    assert result_array.shape == (3,)
    expected_result = np.array([1000, 2000, 3000]) * 36.0
    np.testing.assert_almost_equal(result_array, expected_result)

    with patch("builtins.print") as mock_print:
        derivator.derive_statistics(return_sum=True)
        if mock_print.call_count > 0:
            call_args_list = [str(call[0][0]) for call in mock_print.call_args_list]
            assert any("Base events" in arg for arg in call_args_list)
            assert any("Scaling factors shape" in arg for arg in call_args_list)
            assert any("Scaled events shape" in arg for arg in call_args_list)
            assert any("metrics" in arg for arg in call_args_list)
