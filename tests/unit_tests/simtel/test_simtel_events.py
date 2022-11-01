#!/usr/bin/python3

import logging

import astropy.units as u
import pytest

from simtools.simtel.simtel_events import SimtelEvents

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def testFiles(io_handler):
    testFiles = list()
    testFiles.append(
        io_handler.get_input_data_file(
            fileName="run201_proton_za20deg_azm0deg-North-Prod5_test-production-5-mini.simtel.zst",
            test=True,
        )
    )
    testFiles.append(
        io_handler.get_input_data_file(
            fileName="run202_proton_za20deg_azm0deg-North-Prod5_test-production-5-mini.simtel.zst",
            test=True,
        )
    )
    return testFiles


def test_reading_files(testFiles):
    simtel_events = SimtelEvents(inputFiles=testFiles)

    assert len(simtel_events.inputFiles) == 2


def test_loading_files(testFiles):
    simtel_events = SimtelEvents()

    assert len(simtel_events.inputFiles) == 0

    simtel_events.load_input_files(testFiles)
    assert len(simtel_events.inputFiles) == 2


def test_loading_header(testFiles):
    simtel_events = SimtelEvents(inputFiles=testFiles)
    simtel_events.load_header_and_summary()

    assert 4000.0 == pytest.approx(simtel_events.count_simulated_events())


def test_select_events(testFiles):
    simtel_events = SimtelEvents(inputFiles=testFiles)
    events = simtel_events.select_events()

    assert len(events) == 7


def test_units(testFiles):
    simtel_events = SimtelEvents(inputFiles=testFiles)

    # coreMax without units
    with pytest.raises(TypeError):
        simtel_events.count_simulated_events(energyRange=[0.3 * u.TeV, 300 * u.TeV], coreMax=1500)

    # energyRange without units
    with pytest.raises(TypeError):
        simtel_events.count_simulated_events(energyRange=[0.3, 300], coreMax=1500 * u.m)

    # energyRange with wrong units
    with pytest.raises(TypeError):
        simtel_events.count_simulated_events(energyRange=[0.3 * u.m, 300 * u.m], coreMax=1500 * u.m)
