#!/usr/bin/python3

import logging
import unittest

import simtools.io_handler as io
from simtools.simtel.simtel_events import SimtelEvents

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class TestSimtelEvents(unittest.TestCase):

    def setUp(self):
        self.testFiles = list()
        self.testFiles.append(io.getTestDataFile(
            'run201_proton_za20deg_azm0deg-North-Prod5_test-production-5-mini.simtel.zst')
        )
        self.testFiles.append(io.getTestDataFile(
            'run202_proton_za20deg_azm0deg-North-Prod5_test-production-5-mini.simtel.zst')
        )

    # def test_reading_files(self):
    #     simtel_events = SimtelEvents(inputFiles=self.testFiles)
    #     self.assertEqual(len(simtel_events.inputFiles), 2)

    # def test_loading_files(self):
    #     simtel_events = SimtelEvents()
    #     self.assertEqual(len(simtel_events.inputFiles), 0)

    #     simtel_events.loadInputFiles(self.testFiles)
    #     self.assertEqual(len(simtel_events.inputFiles), 2)

    # def test_loading_header(self):
    #     simtel_events = SimtelEvents(inputFiles=self.testFiles)
    #     simtel_events.loadHeader()

    def test_larger_files(self):
        directory = '/lustre/fs24/group/cta/users/prado/gammasim-files/simtel-data/North/gamma/data'
        files = [
            directory + '/run93_gamma_za20deg_azm0deg-North-Prod5_test-production-5.simtel.zst',
            directory + '/run94_gamma_za20deg_azm0deg-North-Prod5_test-production-5.simtel.zst'
        ]

        simtel_events = SimtelEvents(inputFiles=files)
        print(simtel_events._mcHeader)


if __name__ == '__main__':
    unittest.main()
