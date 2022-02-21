#!/usr/bin/python3

import logging
import unittest

import astropy.units as u

import simtools.io_handler as io
from simtools.simtel.simtel_events import SimtelEvents

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class TestSimtelEvents(unittest.TestCase):
    def setUp(self):
        self.testFiles = list()
        self.testFiles.append(
            io.getTestDataFile(
                "run201_proton_za20deg_azm0deg-North-Prod5_test-production-5-mini.simtel.zst"
            )
        )
        self.testFiles.append(
            io.getTestDataFile(
                "run202_proton_za20deg_azm0deg-North-Prod5_test-production-5-mini.simtel.zst"
            )
        )

    def test_reading_files(self):
        simtel_events = SimtelEvents(inputFiles=self.testFiles)
        self.assertEqual(len(simtel_events.inputFiles), 2)

    def test_loading_files(self):
        simtel_events = SimtelEvents()
        self.assertEqual(len(simtel_events.inputFiles), 0)

        simtel_events.loadInputFiles(self.testFiles)
        self.assertEqual(len(simtel_events.inputFiles), 2)

    def test_loading_header(self):
        simtel_events = SimtelEvents(inputFiles=self.testFiles)
        simtel_events.loadHeaderAndSummary()

    def test_select_events(self):
        simtel_events = SimtelEvents(inputFiles=self.testFiles)
        events = simtel_events.selectEvents()
        self.assertEqual(len(events), 7)

    def test_units(self):
        simtel_events = SimtelEvents(inputFiles=self.testFiles)
        # simtel_events.selectEvents()

        # coreMax without units
        with self.assertRaises(TypeError):
            simtel_events.countSimulatedEvents(energyRange=[0.3 * u.TeV, 300 * u.TeV], coreMax=1500)

        # energyRange without units
        with self.assertRaises(TypeError):
            simtel_events.countSimulatedEvents(energyRange=[0.3, 300], coreMax=1500 * u.m)

        # energyRange with wrong units
        with self.assertRaises(TypeError):
            simtel_events.countSimulatedEvents(
                energyRange=[0.3 * u.m, 300 * u.m],
                coreMax=1500 * u.m
            )


if __name__ == "__main__":
    unittest.main()
