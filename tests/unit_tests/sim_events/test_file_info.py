#!/usr/bin/python3

from types import SimpleNamespace
from unittest import mock

import numpy as np
from eventio import iact
from eventio.simtel import MCRunHeader, MCShower, RunHeader

from simtools.sim_events.file_info import (
    get_corsika_run_and_event_headers,
    get_corsika_run_number,
    get_simulated_events,
)


def test_get_corsika_run_number_with_run_header(mocker):
    run_header = mock.MagicMock(spec=RunHeader)
    run_header.parse.return_value = {"run": 10}
    mc_run_header = mock.MagicMock(spec=MCRunHeader)
    mc_run_header.parse.return_value = {"n_use": 5}
    mc_shower = mock.MagicMock(spec=MCShower)
    mc_shower.parse.return_value = {"primary_id": 1}
    eventio_file = mocker.patch("simtools.sim_events.file_info.EventIOFile")
    eventio_file.return_value.__enter__.return_value = [run_header, mc_run_header, mc_shower]

    assert get_corsika_run_number("synthetic.simtel.zst") == 10


def test_get_simulated_events(mocker):
    events = [SimpleNamespace(header=SimpleNamespace(type=2020)) for _ in range(2)] + [
        SimpleNamespace(header=SimpleNamespace(type=2021)) for _ in range(3)
    ]
    eventio_file = mocker.patch("simtools.sim_events.file_info.EventIOFile")
    eventio_file.return_value.__enter__.return_value = events

    n_showers, n_events = get_simulated_events("synthetic.simtel.zst")
    assert n_showers == 2
    assert n_events == 3


def test_get_simulated_events_corsika_iact(mocker):
    events = [SimpleNamespace(header=SimpleNamespace(type=1202)) for _ in range(2)]
    eventio_file = mocker.patch("simtools.sim_events.file_info.EventIOFile")
    eventio_file.return_value.__enter__.return_value = events

    n_showers, n_events = get_simulated_events("synthetic.corsika.zst")
    assert n_showers == 2
    assert n_events == 0  # CORSIKA IACT files don't have MCEvent objects


def test_get_corsika_run_and_event_headers(mocker):
    run_header = mock.MagicMock(spec=iact.RunHeader)
    run_header.parse.return_value = np.array((7,), dtype=[("run_number", "i4")])
    event_header = mock.MagicMock(spec=iact.EventHeader)
    event_header.parse.return_value = np.array((1,), dtype=[("event_number", "i4")])
    eventio_file = mocker.patch("simtools.sim_events.file_info.EventIOFile")
    eventio_file.return_value.__enter__.return_value = [run_header, event_header]

    parsed_run_header, parsed_event_header = get_corsika_run_and_event_headers(
        "synthetic.corsika.zst"
    )

    assert "run_number" in parsed_run_header.dtype.names
    assert parsed_run_header["run_number"] == 7
    assert "event_number" in parsed_event_header.dtype.names
