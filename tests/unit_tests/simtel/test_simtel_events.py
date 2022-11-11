#!/usr/bin/python3

import logging

import astropy.units as u
import pytest

from simtools.simtel.simtel_events import SimtelEvents

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def test_files(io_handler):
    test_files = list()
    test_files.append(
        io_handler.get_input_data_file(
            file_name="run201_proton_za20deg_azm0deg_North_TestLayout_test-prod.simtel.zst",
            test=True,
        )
    )
    test_files.append(
        io_handler.get_input_data_file(
            file_name="run202_proton_za20deg_azm0deg_North_TestLayout_test-prod.simtel.zst",
            test=True,
        )
    )
    return test_files


def test_reading_files(test_files):
    simtel_events = SimtelEvents(input_files=test_files)

    assert len(simtel_events.input_files) == 2


def test_loading_files(test_files):
    simtel_events = SimtelEvents()

    assert len(simtel_events.input_files) == 0

    simtel_events.load_input_files(test_files)
    assert len(simtel_events.input_files) == 2


def test_loading_header(test_files):
    simtel_events = SimtelEvents(input_files=test_files)
    simtel_events.load_header_and_summary()

    assert 4000.0 == pytest.approx(simtel_events.count_simulated_events())


def test_select_events(test_files):
    simtel_events = SimtelEvents(input_files=test_files)
    events = simtel_events.select_events()

    assert len(events) == 7


def test_units(test_files):
    simtel_events = SimtelEvents(input_files=test_files)

    # core_max without units
    with pytest.raises(TypeError):
        simtel_events.count_simulated_events(energy_range=[0.3 * u.TeV, 300 * u.TeV], core_max=1500)

    # energy_range without units
    with pytest.raises(TypeError):
        simtel_events.count_simulated_events(energy_range=[0.3, 300], core_max=1500 * u.m)

    # energy_range with wrong units
    with pytest.raises(TypeError):
        simtel_events.count_simulated_events(
            energy_range=[0.3 * u.m, 300 * u.m], core_max=1500 * u.m
        )
