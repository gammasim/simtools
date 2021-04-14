#!/usr/bin/python3

import logging
import unittest

import astropy.units as u

from simtools.corsika.corsika_runner import CorsikaRunner

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class TestCorsikaRunner(unittest.TestCase):

    def setUp(self):
        self.corsikaConfigData = {
            'corsikaDataDirectory': './corsika-data',
            'nshow': 10,
            'primary': 'gamma',
            'erange': [100 * u.GeV, 1 * u.TeV],
            'eslope': -2,
            'zenith': 20 * u.deg,
            'azimuth': 0 * u.deg,
            'viewcone': 0 * u.deg,
            'cscat': [10, 1500 * u.m, 0]
        }

        self.corsikaRunner = CorsikaRunner(
            site='south',
            layoutName='Prod5',
            label='test-corsika-runner',
            corsikaConfigData=self.corsikaConfigData
        )

    def test_get_run_script(self):
        runNumber = 3
        script = self.corsikaRunner.getRunScriptFile(runNumber)
        self.assertTrue(script.exists())

    def test_get_run_script_with_invalid_run(self):
        for run in [-2, 'test']:
            with self.assertRaises(ValueError):
                _ = self.corsikaRunner.getRunScriptFile(run)


if __name__ == '__main__':
    unittest.main()
