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

    # def test_invalid_shower_data(self):
    #     newArrayConfigData = copy(self.arrayConfigData)
    #     newArrayConfigData.pop('site')
    #     with self.assertRaises(MissingRequiredEntryInArrayConfig):
    #         newArraySimulator = ArraySimulator(
    #             label=self.label,
    #             configData=newArrayConfigData
    #         )

    def test_run(self):
        self.corsikaFile = io.getTestDataFile(
            'run1_proton_za20deg_azm0deg-North-1LST_trigger_rates.corsika.zst'
        )
        self.arraySimulator.run(inputFileList=str(self.corsikaFile))

    # def test_no_corsika_data(self):
    #     newShowerConfigData = copy(self.showerConfigData)
    #     newShowerConfigData.pop('corsikaDataDirectory')
    #     newShowerSimulator = ShowerSimulator(
    #         label=self.label,
    #         showerConfigData=newShowerConfigData
    #     )
    #     newShowerSimulator.runs
    #     files = newShowerSimulator.getListOfOutputFiles(runList=[3])
    #     self.assertTrue('/' + self.label + '/' in files[0])

    # def test_submitting(self):
    #     self.showerSimulator.submit(runList=[2], submitCommand='more ')

    # def test_runs_range(self):
    #     self.showerSimulator.submit(runRange=[4, 8], submitCommand='more ')

    # def test_get_list_of_files(self):
    #     files = self.showerSimulator.getListOfOutputFiles()
    #     self.assertEqual(len(files), len(self.showerSimulator.runs))

    #     # Giving new runs
    #     files = self.showerSimulator.getListOfOutputFiles(runList=[2, 5, 7])
    #     self.assertEqual(len(files), 3)


if __name__ == '__main__':
    unittest.main()
