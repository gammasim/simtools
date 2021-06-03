#!/usr/bin/python3

import logging
import unittest
from copy import copy

import astropy.units as u

import simtools.io_handler as io
from simtools.array_simulator import ArraySimulator, MissingRequiredEntryInArrayConfig

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

class TestArraySimulator(unittest.TestCase):

    def setUp(self):
        self.label = 'test-array-simulator'
        self.arrayConfigData = {
            'simtelDataDirectory': '.',
            'primary': 'gamma',
            'zenith': 20 * u.deg,
            'azimuth': 0 * u.deg,
            'viewcone': [0 * u.deg, 0 * u.deg],
            # ArrayModel
            'site': 'North',
            'layoutName': '1LST',
            'modelVersion': 'Prod5',
            'default': {
                'LST': '1'
            },
            'M-01': 'FlashCam-D'
        }
        self.arraySimulator = ArraySimulator(
            label=self.label,
            configData=self.arrayConfigData
        )
        self.corsikaFile = io.getTestDataFile(
            'run1_proton_za20deg_azm0deg-North-1LST_trigger_rates.corsika.zst'
        )

    def test_guess_run(self):
        run = self.arraySimulator._guessRunFromFile('run12345_bla_ble')
        self.assertEqual(run, 12345)

        # Invalid run number - returns 1
        run = self.arraySimulator._guessRunFromFile('run1test2_bla_ble')
        self.assertEqual(run, 1)

    def test_invalid_array_data(self):
        newArrayConfigData = copy(self.arrayConfigData)
        newArrayConfigData.pop('site')
        with self.assertRaises(MissingRequiredEntryInArrayConfig):
            newArraySimulator = ArraySimulator(
                label=self.label,
                configData=newArrayConfigData
            )

    def test_run(self):
        self.arraySimulator.run(inputFileList=self.corsikaFile)

    def test_submitting(self):
        self.arraySimulator.submit(inputFileList=self.corsikaFile, submitCommand='more ')

    def test_list_of_files(self):
        self.arraySimulator.submit(inputFileList=self.corsikaFile, submitCommand='more ', test=True)

        self.arraySimulator.printListOfOutputFiles()
        self.arraySimulator.printListOfLogFiles()
        self.arraySimulator.printListOfInputFiles()

        inputFiles = self.arraySimulator.getListOfInputFiles()
        self.assertEqual(inputFiles[0], str(self.corsikaFile))


if __name__ == '__main__':
    unittest.main()
