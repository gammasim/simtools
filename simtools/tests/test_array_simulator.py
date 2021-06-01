#!/usr/bin/python3

import logging
import unittest
from copy import copy

import astropy.units as u

from simtools.array_simulator import ArraySimulator

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
            'viewcone': 0 * u.deg,
            # ArrayModel
            'site': 'North',
            'layoutName': '1LST',
            'modelVersion': 'Prod5',
            'default': {
                'LST': '1'
            }
        }
        self.arraySimulator = ArraySimulator(
            label=self.label,
            configData=self.arrayConfigData
        )

    # def test_invalid_shower_data(self):
    #     newShowerConfigData = copy(self.showerConfigData)
    #     newShowerConfigData.pop('site')
    #     with self.assertRaises(MissingRequiredEntryInShowerConfig):
    #         newShowerSimulator = ShowerSimulator(
    #             label=self.label,
    #             showerConfigData=newShowerConfigData
    #         )
    #         newShowerSimulator.runs

    # def test_runs_invalid_input(self):
    #     newShowerConfigData = copy(self.showerConfigData)
    #     newShowerConfigData['runList'] = [1, 2.5, 'bla']  # Invalid run list
    #     with self.assertRaises(InvalidRunsToSimulate):
    #         newShowerSimulator = ShowerSimulator(
    #             label=self.label,
    #             showerConfigData=newShowerConfigData
    #         )
    #         newShowerSimulator.runs

    # def test_runs_input(self):
    #     newShowerConfigData = copy(self.showerConfigData)
    #     newShowerConfigData['runList'] = [1, 2, 4]
    #     newShowerConfigData['runRange'] = [5, 8]
    #     newShowerSimulator = ShowerSimulator(
    #         label=self.label,
    #         showerConfigData=newShowerConfigData
    #     )
    #     self.assertEqual(newShowerSimulator.runs, [1, 2, 4, 5, 6, 7])

    #     # With overlap
    #     newShowerConfigData['runList'] = [1, 3, 4]
    #     newShowerConfigData['runRange'] = [3, 7]
    #     newShowerSimulator = ShowerSimulator(
    #         label=self.label,
    #         showerConfigData=newShowerConfigData
    #     )
    #     self.assertEqual(newShowerSimulator.runs, [1, 3, 4, 5, 6])

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
