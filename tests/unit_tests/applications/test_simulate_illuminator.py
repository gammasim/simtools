#!/usr/bin/python3

"""Tests for simulate_illuminator application."""

from unittest.mock import Mock, patch

import astropy.units as u
import pytest


@patch("simtools.applications.simulate_illuminator.MultiIlluminatorSimulator")
@patch("simtools.applications.simulate_illuminator.build_application")
def test_main_single_pair_mode(mock_build_app, mock_simulator_class):
    """Test main function in single-pair mode."""
    from simtools.applications.simulate_illuminator import main

    # Setup mock application context
    mock_context = Mock()
    mock_context.args = {
        "light_source": "ILLN-01",
        "telescope": "MSTN-04",
        "simulate_all": False,
        "wavelength": [355 * u.nm],
        "label": "test_label",
        "max_workers": None,
        "site": "North",
        "model_version": "7.0.0",
    }
    mock_build_app.return_value = mock_context

    # Setup mock simulator
    mock_simulator = Mock()
    mock_simulator_class.return_value = mock_simulator

    # Run main
    main()

    # Verify simulator was created correctly
    mock_simulator_class.assert_called_once()
    call_kwargs = mock_simulator_class.call_args[1]
    assert call_kwargs["config"] == mock_context.args
    assert call_kwargs["label"] == "test_label"

    # Verify simulate was called with correct parameters (single-pair filters)
    mock_simulator.simulate.assert_called_once()
    call_kwargs = mock_simulator.simulate.call_args[1]
    assert call_kwargs["wavelengths"] == [355 * u.nm]
    assert call_kwargs["illuminators"] == ["ILLN-01"]
    assert call_kwargs["telescopes"] == ["MSTN-04"]


@patch("simtools.applications.simulate_illuminator.MultiIlluminatorSimulator")
@patch("simtools.applications.simulate_illuminator.build_application")
def test_main_multi_pair_mode(mock_build_app, mock_simulator_class):
    """Test main function in multi-pair mode (simulate_all)."""
    from simtools.applications.simulate_illuminator import main

    # Setup mock application context
    mock_context = Mock()
    mock_context.args = {
        "light_source": None,
        "telescope": None,
        "simulate_all": True,
        "wavelength": [355 * u.nm, 473 * u.nm],
        "label": "multi_test",
        "max_workers": 4,
        "site": "North",
        "model_version": "7.0.0",
    }
    mock_build_app.return_value = mock_context

    # Setup mock simulator
    mock_simulator = Mock()
    mock_simulator_class.return_value = mock_simulator

    # Run main
    main()

    # Verify simulate was called with no filters (all pairs)
    mock_simulator.simulate.assert_called_once()
    call_kwargs = mock_simulator.simulate.call_args[1]
    assert call_kwargs["wavelengths"] == [355 * u.nm, 473 * u.nm]
    assert call_kwargs["illuminators"] is None
    assert call_kwargs["telescopes"] is None


@patch("simtools.applications.simulate_illuminator.MultiIlluminatorSimulator")
@patch("simtools.applications.simulate_illuminator.build_application")
def test_main_multi_pair_with_filters(mock_build_app, mock_simulator_class):
    """Test main function in multi-pair mode with filters."""
    from simtools.applications.simulate_illuminator import main

    # Setup mock application context with filters in multi-pair mode
    mock_context = Mock()
    mock_context.args = {
        "light_source": "ILLN-01",  # Filter to specific illuminator
        "telescope": None,  # All telescopes
        "simulate_all": True,
        "wavelength": None,
        "label": "filtered",
        "max_workers": 2,
        "site": "North",
        "model_version": "7.0.0",
    }
    mock_build_app.return_value = mock_context

    # Setup mock simulator
    mock_simulator = Mock()
    mock_simulator_class.return_value = mock_simulator

    # Run main
    main()

    # Verify simulate was called with partial filter
    mock_simulator.simulate.assert_called_once()
    call_kwargs = mock_simulator.simulate.call_args[1]
    assert call_kwargs["illuminators"] == ["ILLN-01"]
    assert call_kwargs["telescopes"] is None


@patch("simtools.applications.simulate_illuminator.build_application")
def test_main_single_pair_missing_light_source(mock_build_app):
    """Test main function exits when light_source is missing in single-pair mode."""
    from simtools.applications.simulate_illuminator import main

    # Setup mock application context without light_source
    mock_context = Mock()
    mock_context.args = {
        "light_source": None,
        "telescope": "MSTN-04",
        "simulate_all": False,
    }
    mock_build_app.return_value = mock_context

    # Verify sys.exit is called
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert "light_source" in str(exc_info.value)
    assert "telescope are required" in str(exc_info.value)


@patch("simtools.applications.simulate_illuminator.build_application")
def test_main_single_pair_missing_telescope(mock_build_app):
    """Test main function exits when telescope is missing in single-pair mode."""
    from simtools.applications.simulate_illuminator import main

    # Setup mock application context without telescope
    mock_context = Mock()
    mock_context.args = {
        "light_source": "ILLN-01",
        "telescope": None,
        "simulate_all": False,
    }
    mock_build_app.return_value = mock_context

    # Verify sys.exit is called
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert "telescope" in str(exc_info.value)
    assert "required" in str(exc_info.value)


@patch("simtools.applications.simulate_illuminator.build_application")
def test_main_single_pair_missing_both(mock_build_app):
    """Test main function exits when both parameters are missing in single-pair mode."""
    from simtools.applications.simulate_illuminator import main

    # Setup mock application context without both parameters
    mock_context = Mock()
    mock_context.args = {
        "light_source": None,
        "telescope": None,
        "simulate_all": False,
    }
    mock_build_app.return_value = mock_context

    # Verify sys.exit is called
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert "required" in str(exc_info.value)
