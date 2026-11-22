"""Tests for the shared simulation stage types."""

import pytest

from simtools.runners import stage_types, stages
from simtools.simulation_software.programs import STDIO


def test_shared_stage_types_are_available_from_stages():
    assert stages.StreamTo is stage_types.StreamTo
    assert stages.STDIN == stage_types.STDIN


def test_stdin_is_the_stream_marker_of_programs():
    assert stage_types.STDIN == STDIO == "-"

    stream = stage_types.StreamTo(("run",))
    assert stream.commands == ("run",)
    with pytest.raises(AttributeError):
        stream.commands = ()
