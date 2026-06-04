"""Unit tests for multi_illuminator_simulator module."""

import os
from unittest.mock import MagicMock, patch

import astropy.units as u
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
    """Test max_workers initialization with explicit value."""
    simulator = MultiIlluminatorSimulator(
        visibility_data=simple_visibility_data,
        config=base_config,
        max_workers=4,
    )

    assert simulator.max_workers == 4


def test_determine_max_workers_default(simple_visibility_data, base_config):
    """Test max_workers initialization with default (60% of cores)."""
    simulator = MultiIlluminatorSimulator(
        visibility_data=simple_visibility_data,
        config=base_config,
        max_workers=None,
    )

    expected_workers = max(1, int((os.cpu_count() or 1) * 0.6))
    assert simulator.max_workers == expected_workers


def test_determine_max_workers_all_cores(simple_visibility_data, base_config):
    """Test max_workers initialization with 0 (use all cores)."""
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

    results = simulator.simulate(wavelengths=[355 * u.nm])

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
    # Config gets modified with wavelength, so check base config keys are present
    assert all(all(js["config"].get(k) == v for k, v in base_config.items()) for js in job_specs)
    assert all("wavelength" in js["config"] for js in job_specs)


@patch(
    "simtools.simtel.simulator_light_emission."
    "SimulatorLightEmission.get_available_wavelengths_from_config"
)
@patch("simtools.simtel.multi_illuminator_simulator.process_pool_map_ordered")
def test_simulate_filter_by_illuminators(
    mock_pool, mock_get_wavelengths, simple_visibility_data, base_config
):
    """Test simulate() with illuminator filtering and auto-fetch wavelengths."""
    # Mock wavelengths from model
    mock_get_wavelengths.return_value = [355 * u.nm]

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

    # Should have fetched wavelengths with config augmented from filtered pairs
    mock_get_wavelengths.assert_called_once()
    called_config = mock_get_wavelengths.call_args[0][0]
    assert called_config["telescope"] in ["MSTS-01", "MSTS-02"]  # One of the filtered pairs
    assert called_config["light_source"] == "ILLS-01"  # The filtered illuminator


@patch(
    "simtools.simtel.simulator_light_emission."
    "SimulatorLightEmission.get_available_wavelengths_from_config"
)
@patch("simtools.simtel.multi_illuminator_simulator.process_pool_map_ordered")
def test_simulate_filter_by_telescopes(
    mock_pool, mock_get_wavelengths, simple_visibility_data, base_config
):
    """Test simulate() with telescope filtering and auto-fetch wavelengths."""
    # Mock wavelengths from model
    mock_get_wavelengths.return_value = [355 * u.nm]

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

    # Should have fetched wavelengths with config augmented from filtered pairs
    mock_get_wavelengths.assert_called_once()
    called_config = mock_get_wavelengths.call_args[0][0]
    assert called_config["telescope"] == "MSTS-02"  # The filtered telescope
    assert called_config["light_source"] in ["ILLS-01", "ILLS-02"]  # One of the illuminators


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

    results = simulator.simulate(
        wavelengths=[355 * u.nm], illuminators=["ILLS-01"], telescopes=["MSTS-01"]
    )

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
    results = simulator.simulate(
        wavelengths=[355 * u.nm], illuminators=["ILLS-01"], telescopes=["MSTS-03"]
    )

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

    simulator.simulate(wavelengths=[355 * u.nm])
    summary = simulator.get_summary()

    assert summary["total"] == 4
    assert summary["successful"] == 3
    assert summary["failed"] == 1
    assert summary["success_rate"] == pytest.approx(0.75)


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

    simulator.simulate(wavelengths=[355 * u.nm])
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

    simulator.simulate(wavelengths=[355 * u.nm])
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
    mock_sim.validate_simulations.return_value = None
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

    results = simulator.simulate(wavelengths=[355 * u.nm])

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


@patch("simtools.simtel.multi_illuminator_simulator.process_pool_map_ordered")
@patch("simtools.simtel.simulator_light_emission.SimulatorLightEmission")
def test_simulate_with_wavelengths_provided(
    mock_sim_class, mock_pool, simple_visibility_data, base_config
):
    """Test simulate() with explicit wavelengths."""
    import astropy.units as u

    # Mock return value: 4 pairs x 2 wavelengths = 8 results
    mock_pool.return_value = [
        {
            "illuminator": ill,
            "telescope": tel,
            "wavelength": wl,
            "success": True,
            "error": None,
        }
        for wl in [355 * u.nm, 473 * u.nm]
        for ill, tel in [
            ("ILLS-01", "MSTS-01"),
            ("ILLS-01", "MSTS-02"),
            ("ILLS-02", "MSTS-02"),
            ("ILLS-02", "MSTS-03"),
        ]
    ]

    simulator = MultiIlluminatorSimulator(
        config=base_config,
        visibility_data=simple_visibility_data,
        label="test",
        max_workers=2,
    )

    wavelengths = [355 * u.nm, 473 * u.nm]
    results = simulator.simulate(wavelengths=wavelengths)

    # Should create jobs for 4 pairs x 2 wavelengths = 8 jobs
    assert len(results) == 8
    mock_pool.assert_called_once()
    job_specs = mock_pool.call_args[0][1]
    assert len(job_specs) == 8

    # Verify wavelengths are distributed correctly
    wavelengths_in_jobs = [job["wavelength"] for job in job_specs]
    assert wavelengths_in_jobs.count(355 * u.nm) == 4
    assert wavelengths_in_jobs.count(473 * u.nm) == 4

    # Verify wavelengths are in configs
    configs = [job["config"] for job in job_specs]
    assert all("wavelength" in config for config in configs)
    assert sum(1 for config in configs if config["wavelength"] == 355 * u.nm) == 4
    assert sum(1 for config in configs if config["wavelength"] == 473 * u.nm) == 4

    # Verify base label is passed
    labels = [job["label"] for job in job_specs]
    assert all(label == "test" for label in labels)


@patch("simtools.simtel.multi_illuminator_simulator.process_pool_map_ordered")
@patch(
    "simtools.simtel.simulator_light_emission."
    "SimulatorLightEmission.get_available_wavelengths_from_config"
)
def test_simulate_with_wavelengths_from_model(
    mock_get_wavelengths, mock_pool, simple_visibility_data, base_config
):
    """Test simulate() fetches wavelengths from model when not provided."""
    import astropy.units as u

    # Mock wavelengths from model
    model_wavelengths = [266 * u.nm, 355 * u.nm, 473 * u.nm, 532 * u.nm]
    mock_get_wavelengths.return_value = model_wavelengths

    # Mock return value: 4 pairs x 4 wavelengths = 16 results
    mock_pool.return_value = [
        {
            "illuminator": ill,
            "telescope": tel,
            "wavelength": wl,
            "success": True,
            "error": None,
        }
        for wl in model_wavelengths
        for ill, tel in [
            ("ILLS-01", "MSTS-01"),
            ("ILLS-01", "MSTS-02"),
            ("ILLS-02", "MSTS-02"),
            ("ILLS-02", "MSTS-03"),
        ]
    ]

    simulator = MultiIlluminatorSimulator(
        config=base_config,
        visibility_data=simple_visibility_data,
        label="test",
    )

    # Call without wavelengths - should fetch from model
    results = simulator.simulate(wavelengths=None)

    # Should have called get_available_wavelengths_from_config with augmented config
    mock_get_wavelengths.assert_called_once()
    called_config = mock_get_wavelengths.call_args[0][0]

    # Verify config was augmented with telescope and light_source from first pair
    assert called_config["telescope"] == "MSTS-01"
    assert called_config["light_source"] == "ILLS-01"
    assert called_config["site"] == base_config["site"]
    assert called_config["model_version"] == base_config["model_version"]

    # Should create jobs for 4 pairs x 4 wavelengths = 16 jobs
    assert len(results) == 16
    job_specs = mock_pool.call_args[0][1]
    assert len(job_specs) == 16


@patch("simtools.simtel.multi_illuminator_simulator.process_pool_map_ordered")
def test_simulate_with_wavelength_in_config(mock_pool, simple_visibility_data, base_config):
    """Test simulate() uses wavelength from base_config if present."""
    import astropy.units as u

    # Add wavelength to config
    config_with_wl = base_config.copy()
    config_with_wl["wavelength"] = 355 * u.nm

    mock_pool.return_value = [
        {
            "illuminator": "ILLS-01",
            "telescope": "MSTS-01",
            "wavelength": 355 * u.nm,
            "success": True,
            "error": None,
        },
        {
            "illuminator": "ILLS-01",
            "telescope": "MSTS-02",
            "wavelength": 355 * u.nm,
            "success": True,
            "error": None,
        },
        {
            "illuminator": "ILLS-02",
            "telescope": "MSTS-02",
            "wavelength": 355 * u.nm,
            "success": True,
            "error": None,
        },
        {
            "illuminator": "ILLS-02",
            "telescope": "MSTS-03",
            "wavelength": 355 * u.nm,
            "success": True,
            "error": None,
        },
    ]

    simulator = MultiIlluminatorSimulator(
        config=config_with_wl,
        visibility_data=simple_visibility_data,
        label="test",
    )

    # Call without wavelengths - should use the one from config
    results = simulator.simulate(wavelengths=None)

    # Should use wavelength from config (4 pairs x 1 wavelength = 4 jobs)
    assert len(results) == 4
    job_specs = mock_pool.call_args[0][1]
    assert len(job_specs) == 4
    assert all(job["wavelength"] == 355 * u.nm for job in job_specs)


@patch("simtools.simtel.multi_illuminator_simulator.process_pool_map_ordered")
def test_simulate_with_list_of_wavelengths_in_config(
    mock_pool, simple_visibility_data, base_config
):
    """Test simulate() handles list of wavelengths in config."""
    import astropy.units as u

    # Add list of wavelengths to config
    config_with_wl = base_config.copy()
    config_with_wl["wavelength"] = [355 * u.nm, 473 * u.nm]

    # Mock return value: 4 pairs x 2 wavelengths = 8 results
    mock_pool.return_value = [
        {
            "illuminator": ill,
            "telescope": tel,
            "wavelength": wl,
            "success": True,
            "error": None,
        }
        for wl in [355 * u.nm, 473 * u.nm]
        for ill, tel in [
            ("ILLS-01", "MSTS-01"),
            ("ILLS-01", "MSTS-02"),
            ("ILLS-02", "MSTS-02"),
            ("ILLS-02", "MSTS-03"),
        ]
    ]

    simulator = MultiIlluminatorSimulator(
        config=config_with_wl,
        visibility_data=simple_visibility_data,
        label="test",
    )

    results = simulator.simulate(wavelengths=None)

    # Should use both wavelengths from config (4 pairs x 2 wavelengths = 8 jobs)
    assert len(results) == 8
    job_specs = mock_pool.call_args[0][1]
    assert len(job_specs) == 8


@patch("simtools.simtel.multi_illuminator_simulator.process_pool_map_ordered")
@patch(
    "simtools.simtel.simulator_light_emission."
    "SimulatorLightEmission.get_available_wavelengths_from_config"
)
def test_simulate_wavelengths_parameter_overrides_config(
    mock_get_wavelengths, mock_pool, simple_visibility_data, base_config
):
    """Test that explicit wavelengths parameter overrides config."""
    import astropy.units as u

    # Add wavelength to config
    config_with_wl = base_config.copy()
    config_with_wl["wavelength"] = 355 * u.nm

    mock_pool.return_value = [
        {
            "illuminator": "ILLS-01",
            "telescope": "MSTS-01",
            "wavelength": 473 * u.nm,
            "success": True,
            "error": None,
        },
        {
            "illuminator": "ILLS-01",
            "telescope": "MSTS-02",
            "wavelength": 473 * u.nm,
            "success": True,
            "error": None,
        },
    ]

    simulator = MultiIlluminatorSimulator(
        config=config_with_wl,
        visibility_data=simple_visibility_data,
        label="test",
    )

    # Explicit parameter should override config
    results = simulator.simulate(wavelengths=[473 * u.nm])

    # Should NOT call get_available_wavelengths_from_config
    mock_get_wavelengths.assert_not_called()

    # Should use the explicitly provided wavelength
    assert len(results) == 2
    job_specs = mock_pool.call_args[0][1]
    assert all(job["wavelength"] == 473 * u.nm for job in job_specs)


@patch("simtools.simtel.multi_illuminator_simulator.process_pool_map_ordered")
def test_simulate_wavelength_labels_formatted_correctly(
    mock_pool, simple_visibility_data, base_config
):
    """Test that wavelength labels are formatted correctly (no decimal points)."""
    import astropy.units as u

    mock_pool.return_value = []

    simulator = MultiIlluminatorSimulator(
        config=base_config,
        visibility_data=simple_visibility_data,
        label="mytest",
    )

    simulator.simulate(wavelengths=[355.5 * u.nm, 473 * u.nm])

    job_specs = mock_pool.call_args[0][1]

    # Verify wavelengths are passed in configs (label formatting happens in SimulatorLightEmission)
    configs = [job["config"] for job in job_specs]
    wavelengths_in_configs = [config["wavelength"] for config in configs]
    assert any(wl == 355.5 * u.nm for wl in wavelengths_in_configs)
    assert any(wl == 473 * u.nm for wl in wavelengths_in_configs)

    # Verify base label is preserved
    labels = [job["label"] for job in job_specs]
    assert all(label == "mytest" for label in labels)
