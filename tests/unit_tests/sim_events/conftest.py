"""Synthetic eventio objects shared by reader and reduced-writer tests."""

from unittest.mock import MagicMock

import pytest
from eventio.simtel import (
    ArrayEvent,
    MCEvent,
    MCRunHeader,
    MCShower,
    TrackingPosition,
    TriggerInformation,
)


def create_mc_run_header():
    """Create mock MC run header."""
    mock_header = MagicMock(spec=MCRunHeader)
    mock_header.parse.return_value = {
        "run": 123,
        "n_use": 2,  # Important: Must be >= 1
        "direction": [0.0, 70.0 / 57.3],
        "E_range": [0.003, 330.0],
        "energy_spectrum_slope": -2.0,
        "viewcone": [0.0, 10.0],
        "core_range": [0.0, 1000.0],
    }
    return mock_header


def create_mc_shower(shower_id=1):
    """Create mock MC shower."""
    mock_shower = MagicMock(spec=MCShower)
    mock_shower.parse.return_value = {
        "energy": 1.0,
        "azimuth": 0.1,
        "altitude": 0.1,
        "shower": shower_id,  # Must match shower_num in mc_event
        "primary_id": 1,
    }
    return mock_shower


def create_mc_event(shower_num=1, event_id=42):
    """Create mock MC event."""
    mock_event = MagicMock(spec=MCEvent)
    mock_event.parse.return_value = {
        "shower_num": shower_num,  # Must match shower in mc_shower
        "event_id": event_id,
        "xcore": 0.1,
        "ycore": 0.1,
        "aweight": 1.0,
    }
    return mock_event


def create_array_event():
    """Create mock array event."""
    mock_event = MagicMock(spec=ArrayEvent)
    mock_trigger = MagicMock(spec=TriggerInformation)
    mock_trigger.parse.return_value = {"triggered_telescopes": [1, 2, 3]}
    mock_tracking = MagicMock(spec=TrackingPosition)
    mock_tracking.parse.return_value = {"altitude_raw": 0.5, "azimuth_raw": 1.2}
    mock_event.__iter__.return_value = [mock_trigger, mock_tracking]
    mock_event.event_id = 42
    return mock_event


@pytest.fixture
def eventio_objects():
    """Provide constructors for synthetic native events."""
    return {
        "run_header": create_mc_run_header,
        "shower": create_mc_shower,
        "event": create_mc_event,
        "array_event": create_array_event,
    }
