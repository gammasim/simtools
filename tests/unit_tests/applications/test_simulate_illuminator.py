#!/usr/bin/python3

"""Tests for simulate_illuminator application."""

from unittest.mock import Mock, patch

import astropy.units as u


@patch("simtools.applications.simulate_illuminator.MultiIlluminatorSimulator")
@patch("simtools.application.definition.ApplicationDefinition.start")
def test_main_single_pair_mode(mock_application_start, mock_simulator_class):
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
    mock_application_start.return_value = mock_context

    # Setup mock simulator with successful result
    mock_simulator = Mock()
    mock_simulator.simulate.return_value = [{"success": True}]
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
