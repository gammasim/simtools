"""Unit tests for multi_illuminator_simulator module."""

import os
from unittest.mock import MagicMock, patch

import pytest

from simtools.simtel.multi_illuminator_simulator import (
    MultiIlluminatorSimulator,
    _simulate_illuminator_telescope_pair,
)


@pytest.fixture
def simple_visibility_data():
    """
    Create a simple visibility dict with 2 illuminators and 3 telescopes.

    Returns
    -------
    dict
        Visibility data dictionary.
    """
    return {
        "columns": ["illuminator_id", "telescope_id", "visible"],
        "rows": [
            ["ILLS-01", "MSTS-01", True],
            ["ILLS-01", "MSTS-02", True],
            ["ILLS-01", "MSTS-03", False],
            ["ILLS-02", "MSTS-01", False],
            ["ILLS-02", "MSTS-02", True],
            ["ILLS-02", "MSTS-03", True],
        ],
    }


@pytest.fixture
def base_config():
    """
    Create a base configuration for simulations.

    Returns
    -------
    dict
        Configuration dictionary.
    """
    return {
        "site": "South",
        "model_version": "6.0.0",
        "number_of_events": 100,
        "light_source_type": "illuminator",
    }


def test_initialization(simple_visibility_data, base_config):
    """Test MultiIlluminatorSimulator initialization."""
    simulator = MultiIlluminatorSimulator(
        visibility_data=simple_visibility_data,
        config=base_config,
        label="test_sim",
        max_workers=2,
    )

    assert simulator.visibility is not None
    assert simulator.visibility.n_illuminators == 2
    assert simulator.visibility.n_telescopes == 3
    assert simulator.visibility.n_valid_pairs == 4
    assert simulator.base_config == base_config
    assert simulator.label == "test_sim"
    assert simulator.max_workers == 2
    assert simulator.results is None


def test_initialization_default_label(simple_visibility_data, base_config):
    """Test initialization with default label."""
    simulator = MultiIlluminatorSimulator(
        visibility_data=simple_visibility_data,
        config=base_config,
    )

    assert simulator.label == "multi_illuminator"


def test_determine_max_workers_explicit(simple_visibility_data, base_config):
    """Test _determine_max_workers with explicit value."""
    simulator = MultiIlluminatorSimulator(
        visibility_data=simple_visibility_data,
        config=base_config,
        max_workers=4,
    )

    assert simulator.max_workers == 4


def test_determine_max_workers_default(simple_visibility_data, base_config):
    """Test _determine_max_workers with default (60% of cores)."""
    simulator = MultiIlluminatorSimulator(
        visibility_data=simple_visibility_data,
        config=base_config,
        max_workers=None,
    )

    expected_workers = max(1, int((os.cpu_count() or 1) * 0.6))
    assert simulator.max_workers == expected_workers


def test_determine_max_workers_all_cores(simple_visibility_data, base_config):
    """Test _determine_max_workers with 0 (use all cores)."""
    simulator = MultiIlluminatorSimulator(
        visibility_data=simple_visibility_data,
        config=base_config,
        max_workers=0,
    )

    expected_workers = os.cpu_count() or 1
    assert simulator.max_workers == expected_workers


@patch("simtools.simtel.multi_illuminator_simulator.process_pool_map_ordered")
def test_simulate_all_pairs(mock_pool, simple_visibility_data, base_config):
    """Test simulate() runs all valid pairs."""
    # Mock successful results
    mock_pool.return_value = [
        {"illuminator": "ILLS-01", "telescope": "MSTS-01", "success": True, "error": None},
        {"illuminator": "ILLS-01", "telescope": "MSTS-02", "success": True, "error": None},
        {"illuminator": "ILLS-02", "telescope": "MSTS-02", "success": True, "error": None},
        {"illuminator": "ILLS-02", "telescope": "MSTS-03", "success": True, "error": None},
    ]

    simulator = MultiIlluminatorSimulator(
        visibility_data=simple_visibility_data,
        config=base_config,
        max_workers=2,
    )

    results = simulator.simulate()

    assert len(results) == 4
    assert all(r["success"] for r in results)
    mock_pool.assert_called_once()

    # Check max_workers was passed
    call_kwargs = mock_pool.call_args[1]
    assert call_kwargs["max_workers"] == 2

    # Check job specs passed to the pool
    call_args = mock_pool.call_args[0]
    job_specs = call_args[1]
    assert len(job_specs) == 4
    pairs = {(js["illuminator"], js["telescope"]) for js in job_specs}
    expected_pairs = {
        ("ILLS-01", "MSTS-01"),
        ("ILLS-01", "MSTS-02"),
        ("ILLS-02", "MSTS-02"),
        ("ILLS-02", "MSTS-03"),
    }
    assert pairs == expected_pairs
    assert all(js["site"] == "South" for js in job_specs)
    assert all(js["config"] == base_config for js in job_specs)


@patch("simtools.simtel.multi_illuminator_simulator.process_pool_map_ordered")
def test_simulate_filter_by_illuminators(mock_pool, simple_visibility_data, base_config):
    """Test simulate() with illuminator filtering."""
    mock_pool.return_value = [
        {"illuminator": "ILLS-01", "telescope": "MSTS-01", "success": True, "error": None},
        {"illuminator": "ILLS-01", "telescope": "MSTS-02", "success": True, "error": None},
    ]

    simulator = MultiIlluminatorSimulator(
        visibility_data=simple_visibility_data,
        config=base_config,
    )

    results = simulator.simulate(illuminators=["ILLS-01"])

    # Should only simulate pairs with ILLS-01
    assert len(results) == 2
    assert all(r["illuminator"] == "ILLS-01" for r in results)


@patch("simtools.simtel.multi_illuminator_simulator.process_pool_map_ordered")
def test_simulate_filter_by_telescopes(mock_pool, simple_visibility_data, base_config):
    """Test simulate() with telescope filtering."""
    mock_pool.return_value = [
        {"illuminator": "ILLS-01", "telescope": "MSTS-02", "success": True, "error": None},
        {"illuminator": "ILLS-02", "telescope": "MSTS-02", "success": True, "error": None},
    ]

    simulator = MultiIlluminatorSimulator(
        visibility_data=simple_visibility_data,
        config=base_config,
    )

    results = simulator.simulate(telescopes=["MSTS-02"])

    # Should only simulate pairs with MSTS-02
    assert len(results) == 2
    assert all(r["telescope"] == "MSTS-02" for r in results)


@patch("simtools.simtel.multi_illuminator_simulator.process_pool_map_ordered")
def test_simulate_filter_both(mock_pool, simple_visibility_data, base_config):
    """Test simulate() with both illuminator and telescope filtering."""
    mock_pool.return_value = [
        {"illuminator": "ILLS-01", "telescope": "MSTS-01", "success": True, "error": None},
    ]

    simulator = MultiIlluminatorSimulator(
        visibility_data=simple_visibility_data,
        config=base_config,
    )

    results = simulator.simulate(illuminators=["ILLS-01"], telescopes=["MSTS-01"])

    assert len(results) == 1
    assert results[0]["illuminator"] == "ILLS-01"
    assert results[0]["telescope"] == "MSTS-01"


@patch("simtools.simtel.multi_illuminator_simulator.process_pool_map_ordered")
def test_simulate_no_valid_pairs_after_filtering(mock_pool, simple_visibility_data, base_config):
    """Test simulate() when filtering results in no valid pairs."""
    simulator = MultiIlluminatorSimulator(
        visibility_data=simple_visibility_data,
        config=base_config,
    )

    # MSTS-03 not visible to ILLS-01
    results = simulator.simulate(illuminators=["ILLS-01"], telescopes=["MSTS-03"])

    assert len(results) == 0
    mock_pool.assert_not_called()


@patch("simtools.simtel.multi_illuminator_simulator.process_pool_map_ordered")
def test_get_summary_success(mock_pool, simple_visibility_data, base_config):
    """Test get_summary() with successful simulations."""
    mock_pool.return_value = [
        {"illuminator": "ILLS-01", "telescope": "MSTS-01", "success": True, "error": None},
        {"illuminator": "ILLS-01", "telescope": "MSTS-02", "success": True, "error": None},
        {"illuminator": "ILLS-02", "telescope": "MSTS-02", "success": False, "error": "Test error"},
        {"illuminator": "ILLS-02", "telescope": "MSTS-03", "success": True, "error": None},
    ]

    simulator = MultiIlluminatorSimulator(
        visibility_data=simple_visibility_data,
        config=base_config,
    )

    simulator.simulate()
    summary = simulator.get_summary()

    assert summary["total"] == 4
    assert summary["successful"] == 3
    assert summary["failed"] == 1
    assert summary["success_rate"] == 0.75


def test_get_summary_before_simulation(simple_visibility_data, base_config):
    """Test get_summary() raises error when called before simulate()."""
    simulator = MultiIlluminatorSimulator(
        visibility_data=simple_visibility_data,
        config=base_config,
    )

    with pytest.raises(RuntimeError, match="No simulations have been run yet"):
        simulator.get_summary()


@patch("simtools.simtel.multi_illuminator_simulator.process_pool_map_ordered")
def test_get_failed_pairs(mock_pool, simple_visibility_data, base_config):
    """Test get_failed_pairs() returns correct failures."""
    mock_pool.return_value = [
        {"illuminator": "ILLS-01", "telescope": "MSTS-01", "success": True, "error": None},
        {"illuminator": "ILLS-01", "telescope": "MSTS-02", "success": False, "error": "Error 1"},
        {"illuminator": "ILLS-02", "telescope": "MSTS-02", "success": False, "error": "Error 2"},
        {"illuminator": "ILLS-02", "telescope": "MSTS-03", "success": True, "error": None},
    ]

    simulator = MultiIlluminatorSimulator(
        visibility_data=simple_visibility_data,
        config=base_config,
    )

    simulator.simulate()
    failed_pairs = simulator.get_failed_pairs()

    assert len(failed_pairs) == 2
    assert ("ILLS-01", "MSTS-02") in failed_pairs
    assert ("ILLS-02", "MSTS-02") in failed_pairs


def test_get_failed_pairs_before_simulation(simple_visibility_data, base_config):
    """Test get_failed_pairs() raises error when called before simulate()."""
    simulator = MultiIlluminatorSimulator(
        visibility_data=simple_visibility_data,
        config=base_config,
    )

    with pytest.raises(RuntimeError, match="No simulations have been run yet"):
        simulator.get_failed_pairs()


@patch("simtools.simtel.multi_illuminator_simulator.process_pool_map_ordered")
def test_get_failed_results(mock_pool, simple_visibility_data, base_config):
    """Test get_failed_results() returns detailed failure information."""
    mock_pool.return_value = [
        {"illuminator": "ILLS-01", "telescope": "MSTS-01", "success": True, "error": None},
        {"illuminator": "ILLS-01", "telescope": "MSTS-02", "success": False, "error": "Error 1"},
        {"illuminator": "ILLS-02", "telescope": "MSTS-02", "success": False, "error": "Error 2"},
        {"illuminator": "ILLS-02", "telescope": "MSTS-03", "success": True, "error": None},
    ]

    simulator = MultiIlluminatorSimulator(
        visibility_data=simple_visibility_data,
        config=base_config,
    )

    simulator.simulate()
    failed_results = simulator.get_failed_results()

    assert len(failed_results) == 2
    assert all(not r["success"] for r in failed_results)
    assert failed_results[0]["error"] == "Error 1"
    assert failed_results[1]["error"] == "Error 2"


def test_get_failed_results_before_simulation(simple_visibility_data, base_config):
    """Test get_failed_results() raises error when called before simulate()."""
    simulator = MultiIlluminatorSimulator(
        visibility_data=simple_visibility_data,
        config=base_config,
    )

    with pytest.raises(RuntimeError, match="No simulations have been run yet"):
        simulator.get_failed_results()


@patch("simtools.simtel.multi_illuminator_simulator.SimulatorLightEmission")
def test_simulate_illuminator_telescope_pair_success(mock_sim_class):
    """Test _simulate_illuminator_telescope_pair() worker function with successful simulation."""
    # Mock SimulatorLightEmission instance
    mock_sim = MagicMock()
    mock_sim.simulate.return_value = None
    mock_sim.validate_simulations.return_value = {"status": "success"}
    mock_sim_class.return_value = mock_sim

    job_spec = {
        "illuminator": "ILLS-01",
        "telescope": "MSTS-01",
        "site": "South",
        "label": "test",
        "config": {
            "site": "South",
            "model_version": "6.0.0",
        },
    }

    result = _simulate_illuminator_telescope_pair(job_spec)

    assert result["illuminator"] == "ILLS-01"
    assert result["telescope"] == "MSTS-01"
    assert result["success"] is True
    assert result["error"] is None
    assert result["validation_result"] == {"status": "success"}

    # Verify SimulatorLightEmission was called correctly
    mock_sim_class.assert_called_once()
    call_kwargs = mock_sim_class.call_args[1]
    assert call_kwargs["telescope"] == "MSTS-01"
    assert call_kwargs["label"] == "test"
    assert call_kwargs["light_emission_config"]["light_source"] == "ILLS-01"
    assert call_kwargs["light_emission_config"]["telescope"] == "MSTS-01"

    mock_sim.simulate.assert_called_once()
    mock_sim.validate_simulations.assert_called_once()


@patch("simtools.simtel.multi_illuminator_simulator.SimulatorLightEmission")
def test_simulate_illuminator_telescope_pair_failure(mock_sim_class):
    """Test _simulate_illuminator_telescope_pair() worker function with failed simulation."""
    # Mock SimulatorLightEmission to raise an exception
    mock_sim_class.side_effect = ValueError("Simulation failed")

    job_spec = {
        "illuminator": "ILLS-01",
        "telescope": "MSTS-01",
        "site": "South",
        "label": "test",
        "config": {
            "site": "South",
            "model_version": "6.0.0",
        },
    }

    result = _simulate_illuminator_telescope_pair(job_spec)

    assert result["illuminator"] == "ILLS-01"
    assert result["telescope"] == "MSTS-01"
    assert result["success"] is False
    assert "Simulation failed" in result["error"]
    assert result["validation_result"] is None


@patch("simtools.simtel.multi_illuminator_simulator.process_pool_map_ordered")
def test_simulate_empty_visibility_table(mock_pool, base_config):
    """Test simulate() with visibility data having no valid pairs."""
    visibility_data = {
        "columns": ["illuminator_id", "telescope_id", "visible"],
        "rows": [
            ["ILLS-01", "MSTS-01", False],
            ["ILLS-01", "MSTS-02", False],
            ["ILLS-02", "MSTS-01", False],
            ["ILLS-02", "MSTS-02", False],
        ],
    }

    simulator = MultiIlluminatorSimulator(
        visibility_data=visibility_data,
        config=base_config,
    )

    results = simulator.simulate()

    assert len(results) == 0
    mock_pool.assert_not_called()


@patch("simtools.model.site_model.SiteModel")
def test_load_visibility_from_site_model(mock_site_model_class, base_config):
    """Test that visibility data is loaded from SiteModel when not provided."""
    mock_site_model = MagicMock()
    mock_site_model.get_parameter_value.return_value = {
        "columns": ["illuminator_id", "telescope_id", "visible"],
        "rows": [
            ["ILLS-01", "MSTS-01", True],
            ["ILLS-02", "MSTS-01", True],
        ],
    }
    mock_site_model_class.return_value = mock_site_model

    simulator = MultiIlluminatorSimulator(config=base_config)

    assert simulator.visibility.n_valid_pairs == 2
    mock_site_model_class.assert_called_once_with(site="South", model_version="6.0.0")
    mock_site_model.get_parameter_value.assert_called_once_with("illuminator_telescope_visibility")
