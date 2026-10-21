"""Test format selection independently of installed simulation software."""

from types import SimpleNamespace

import pytest

from simtools.sim_events import file_info
from simtools.sim_events.formats import registry
from simtools.sim_events.formats.eventio_reader import EventioReader


@pytest.fixture
def isolated_readers(monkeypatch):
    """Keep registrations local to each test."""
    monkeypatch.setattr(registry, "_READERS", registry._READERS.copy())


def test_default_reader():
    reader = registry.get_reader("simulation.iact")
    assert isinstance(reader, EventioReader)
    assert reader.file_name == "simulation.iact"
    assert "eventio" in registry.available_formats()


def test_register_reader_and_file_information(isolated_readers):
    reader = SimpleNamespace(read_run_number=lambda: 42, count_events=lambda: (3, 6))
    registry.register_reader("example", lambda file: reader)
    assert registry.get_reader("arbitrary.extension", "example") is reader
    assert file_info.get_run_number("arbitrary.extension", "example") == 42
    assert file_info.get_simulated_events("arbitrary.extension", "example") == (3, 6)
    assert "example" in registry.available_formats()


@pytest.mark.parametrize(("name", "factory"), [("", lambda file: None), ("example", None)])
def test_invalid_registration(name, factory):
    with pytest.raises(ValueError, match="format name and callable"):
        registry.register_reader(name, factory)


def test_duplicate_registration():
    with pytest.raises(ValueError, match="already registered"):
        registry.register_reader("eventio", lambda file: None)


def test_unknown_reader():
    with pytest.raises(ValueError, match=r"Unknown simulation-file format.*Available formats"):
        registry.get_reader("simulation.anything", "missing")
