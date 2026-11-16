"""Tests for the shared simulation stage types."""

import pytest

from simtools.runners import stage_types, stages


def test_shared_stage_types_are_available_from_stages():
    assert stages.StreamTo is stage_types.StreamTo
    assert stages.CHERENKOV_PHOTONS == stage_types.CHERENKOV_PHOTONS
    assert stages.CAMERA_EVENTS == stage_types.CAMERA_EVENTS
    assert stages.STDIN == stage_types.STDIN

    stream = stage_types.StreamTo(("run",))
    assert stream.commands == ("run",)
    with pytest.raises(AttributeError):
        stream.commands = ()
