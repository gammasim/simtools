#!/usr/bin/python3

import logging
import unittest
from copy import copy

import astropy.units as u

from simtools.shower_simulator import ShowerSimulator, InvalidRunsToSimulate

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

class TestShowerSimulator(unittest.TestCase):

    def setUp(self):
        self.showerConfigData = {
            'corsikaDataDirectory': './corsika-data',
            'site': 'South',
            'layoutName': 'Prod5',
            'runList': [3, 4],
            'runRange': [6, 10],
            'nshow': 10,
            'primary': 'gamma',
            'erange': [100 * u.GeV, 1 * u.TeV],
            'eslope': -2,
            'zenith': 20 * u.deg,
            'azimuth': 0 * u.deg,
            'viewcone': 0 * u.deg,
            'cscat': [10, 1500 * u.m, 0]
        }
        self.showerSimulator = ShowerSimulator(
            label='test-shower-simulator',
            showerConfigData=self.showerConfigData
        )

    def test_runs_invalid_input(self):
        newShowerConfigData = copy(self.showerConfigData)
        newShowerConfigData['runList'] = [1, 2.5, 'bla']  # Invalid run list
        with self.assertRaises(InvalidRunsToSimulate):
            newShowerSimulator = ShowerSimulator(
                label='test-shower-simulator',
                showerConfigData=newShowerConfigData
            )
            newShowerSimulator.runs

    def test_runs_input(self):
        newShowerConfigData = copy(self.showerConfigData)
        newShowerConfigData['runList'] = [1, 2, 4]
        newShowerConfigData['runRange'] = [5, 8]
        newShowerSimulator = ShowerSimulator(
            label='test-shower-simulator',
            showerConfigData=newShowerConfigData
        )
        self.assertEqual(newShowerSimulator.runs, [1, 2, 4, 5, 6, 7])

        # With overlap
        newShowerConfigData['runList'] = [1, 3, 4]
        newShowerConfigData['runRange'] = [3, 7]
        newShowerSimulator = ShowerSimulator(
            label='test-shower-simulator',
            showerConfigData=newShowerConfigData
        )
        self.assertEqual(newShowerSimulator.runs, [1, 3, 4, 5, 6])

    def test_submitting(self):
        self.showerSimulator.submit(runList=[2], submitCommand='more ')

    def test_runs_range(self):
        self.showerSimulator.submit(runRange=[4, 8], submitCommand='more ')

    def test_get_list_of_files(self):
        files = self.showerSimulator.getListOfOutputFiles()
        self.assertEqual(len(files), len(self.showerSimulator.runs))

        # Giving new runs
        files = self.showerSimulator.getListOfOutputFiles(runList=[2, 5, 7])
        self.assertEqual(len(files), 3)


if __name__ == '__main__':
    unittest.main()
