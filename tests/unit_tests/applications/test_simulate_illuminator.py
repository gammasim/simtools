#!/usr/bin/python3

"""Tests for simulate_illuminator application."""

from unittest.mock import Mock, patch

import astropy.units as u
import pytest

_ILLUMINATOR = "ILLN-01"
_TELESCOPE = "MSTN-04"


@patch("simtools.applications.simulate_illuminator.MultiIlluminatorSimulator")
@patch("simtools.application.definition.ApplicationDefinition.start")
def test_main_single_pair_mode(mock_application_start, mock_simulator_class):
    from simtools.applications.simulate_illuminator import main

    # Setup mock application context
    mock_context = Mock()
    mock_context.args = {
        "light_source": _ILLUMINATOR,
        "telescope": _TELESCOPE,
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
    assert call_kwargs["illuminators"] == [_ILLUMINATOR]
    assert call_kwargs["telescopes"] == [_TELESCOPE]


@patch("simtools.applications.simulate_illuminator.MultiIlluminatorSimulator")
@patch("simtools.application.definition.ApplicationDefinition.start")
def test_main_reports_failed_simulations(mock_application_start, mock_simulator_class):
    from simtools.applications.simulate_illuminator import main

    mock_context = Mock()
    mock_context.args = {
        "light_source": _ILLUMINATOR,
        "telescope": _TELESCOPE,
        "simulate_all": False,
        "wavelength": [355 * u.nm],
        "label": "test_label",
        "max_workers": None,
        "site": "North",
        "model_version": "7.0.0",
    }
    mock_application_start.return_value = mock_context
    mock_simulator_class.return_value.simulate.return_value = [
        {
            "illuminator": _ILLUMINATOR,
            "telescope": _TELESCOPE,
            "success": False,
            "error": "Parameter illuminator_tower_height was not found",
        }
    ]

    with pytest.raises(SystemExit, match="illuminator_tower_height was not found"):
        main()


@patch("simtools.applications.simulate_illuminator.MultiIlluminatorSimulator")
@patch("simtools.application.definition.ApplicationDefinition.start")
def test_main_reports_empty_multi_pair_batch(mock_application_start, mock_simulator_class):
    from simtools.applications.simulate_illuminator import main

    mock_context = Mock()
    mock_context.args = {
        "light_source": None,
        "telescope": None,
        "simulate_all": True,
        "wavelength": [355 * u.nm],
    }
    mock_application_start.return_value = mock_context
    mock_simulator = mock_simulator_class.return_value
    mock_simulator.simulate.return_value = []
    mock_simulator.visibility.n_valid_pairs = 0

    with pytest.raises(SystemExit, match=r"light_source=all, telescope=all.*0 valid pairs"):
        main()


def _pair_selection_args(simulate_all, light_source=None, telescope=None):
    """Return the arguments read by the pair-selection validation."""
    return {"simulate_all": simulate_all, "light_source": light_source, "telescope": telescope}


@pytest.mark.parametrize(
    ("simulate_all", "light_source", "telescope", "is_valid"),
    [
        (True, None, None, True),
        (True, _ILLUMINATOR, None, True),
        (False, _ILLUMINATOR, _TELESCOPE, True),
        (False, None, None, False),
        (False, _ILLUMINATOR, None, False),
        (False, None, _TELESCOPE, False),
    ],
)
def test_validate_pair_selection(simulate_all, light_source, telescope, is_valid):
    from simtools.applications.simulate_illuminator import _validate_pair_selection

    parser = Mock()
    _validate_pair_selection(
        _pair_selection_args(simulate_all, light_source, telescope), {}, parser
    )

    if is_valid:
        parser.error.assert_not_called()
    else:
        parser.error.assert_called_once()
        assert "required for single-pair mode" in parser.error.call_args.args[0]
