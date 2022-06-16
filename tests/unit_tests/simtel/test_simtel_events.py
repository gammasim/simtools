#!/usr/bin/python3

import logging
import pytest

import astropy.units as u

import simtools.io_handler as io
from simtools.simtel.simtel_events import SimtelEvents

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def testFiles():
    testFiles = list()
    testFiles.append(
        io.getTestDataFile(
            "run201_proton_za20deg_azm0deg-North-Prod5_test-production-5-mini.simtel.zst"
        )
    )
    testFiles.append(
        io.getTestDataFile(
            "run202_proton_za20deg_azm0deg-North-Prod5_test-production-5-mini.simtel.zst"
        )
    )
    return testFiles


def test_reading_files(testFiles):
    simtel_events = SimtelEvents(inputFiles=testFiles)

    assert len(simtel_events.inputFiles) == 2


def test_loading_files(testFiles):
    simtel_events = SimtelEvents()

    assert len(simtel_events.inputFiles) == 0

    simtel_events.loadInputFiles(testFiles)
    assert len(simtel_events.inputFiles) == 2


def test_loading_header(testFiles):
    simtel_events = SimtelEvents(inputFiles=testFiles)
    simtel_events.loadHeaderAndSummary()

    assert 4000.0 == pytest.approx(simtel_events.countSimulatedEvents())


def test_select_events(testFiles):
    simtel_events = SimtelEvents(inputFiles=testFiles)
    events = simtel_events.selectEvents()

    assert len(events) == 7


def test_units(testFiles):
    simtel_events = SimtelEvents(inputFiles=testFiles)

    # coreMax without units
    with pytest.raises(TypeError):
        simtel_events.countSimulatedEvents(
            energyRange=[0.3 * u.TeV, 300 * u.TeV], coreMax=1500)

    # energyRange without units
    with pytest.raises(TypeError):
        simtel_events.countSimulatedEvents(
            energyRange=[0.3, 300], coreMax=1500 * u.m)

    # energyRange with wrong units
    with pytest.raises(TypeError):
        simtel_events.countSimulatedEvents(
            energyRange=[0.3 * u.m, 300 * u.m],
            coreMax=1500 * u.m
        )
