"""Test format selection independently of installed simulation software."""

import pytest

from simtools.sim_events.formats import registry
from simtools.sim_events.formats.eventio_reader import EventioReader


def test_default_reader():
    reader = registry.get_reader("simulation.iact")
    assert isinstance(reader, EventioReader)
    assert reader.file_name == "simulation.iact"
    assert "eventio" in registry.available_formats()


def test_unknown_reader():
    with pytest.raises(ValueError, match=r"Unknown simulation-file format.*Available formats"):
        registry.get_reader("simulation.anything", "missing")
