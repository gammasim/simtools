#!/usr/bin/python3

import pytest

from simtools.simtel.simtel_event_reader import read_events


class FakeSimTelFile:
    def __init__(self, tel_id, tel_descriptions, events_data):
        self.telescope_descriptions = tel_descriptions
        self._events = events_data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __iter__(self):
        return iter(self._events)


def _setup_mocks(monkeypatch, tel_id, tel_descriptions=None, events_data=None):
    monkeypatch.setattr(
        "simtools.simtel.simtel_event_reader.get_sim_telarray_telescope_id",
        lambda telescope, file_name: tel_id,
    )
    if tel_descriptions is not None:
        monkeypatch.setattr(
            "simtools.simtel.simtel_event_reader.SimTelFile",
            lambda *args, **kwargs: FakeSimTelFile(tel_id, tel_descriptions, events_data or []),
        )


def test_read_events_telescope_not_found(monkeypatch, caplog):
    _setup_mocks(monkeypatch, None)
    event_ids, tel_desc, events = read_events("file.simtel", "LST", 0, 1)
    assert (event_ids, tel_desc, events) == (None, None, None)
    assert "Telescope type 'LST' not found in file 'file.simtel'." in caplog.text


def test_read_events_tel_id_missing_in_descriptions(monkeypatch, caplog):
    _setup_mocks(monkeypatch, 42, {}, [])
    event_ids, tel_desc, events = read_events("file.simtel", "MST", 0, 1)
    assert (event_ids, tel_desc, events) == (None, None, None)
    assert "Telescope ID '42' not found in file 'file.simtel'." in caplog.text


@pytest.mark.parametrize(
    ("first_event", "max_events", "expected_ids", "expected_events"),
    [
        (1, 2, [1, 2], ["evt1", "evt2"]),
        (None, 1, [0], ["evt0"]),
    ],
)
def test_read_events_with_event_range(
    monkeypatch, first_event, max_events, expected_ids, expected_events
):
    tel_id = 7
    events_data = [{"telescope_events": {tel_id: f"evt{i}"}} for i in range(3)]
    _setup_mocks(monkeypatch, tel_id, {tel_id: {"name": "TEST"}}, events_data)
    event_ids, tel_desc, events = read_events("file.simtel", "SST", first_event, max_events)
    assert event_ids == expected_ids
    assert tel_desc == {"name": "TEST"}
    assert events == expected_events
