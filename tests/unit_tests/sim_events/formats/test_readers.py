"""Test format selection independently of installed simulation software."""

import inspect

import pytest

from simtools.sim_events.formats import readers
from simtools.sim_events.formats.eventio_reader import EventioReader


def test_default_reader():
    reader = readers.get_reader("simulation.iact")
    assert isinstance(reader, EventioReader)
    assert reader.file_name == "simulation.iact"
    assert "eventio" in readers.available_formats()


def test_unknown_reader():
    with pytest.raises(ValueError, match=r"Unknown simulation-file format.*Available formats"):
        readers.get_reader("simulation.anything", "missing")


def test_eventio_reader_matches_protocol_signatures():
    method_names = (
        "read_run_number",
        "count_events",
        "iter_records",
        "read_metadata",
        "read_simulation_parameters",
    )

    for method_name in method_names:
        protocol_method = getattr(readers.EventReader, method_name)
        reader_method = getattr(EventioReader, method_name)
        assert inspect.signature(reader_method) == inspect.signature(protocol_method)
