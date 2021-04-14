#!/usr/bin/python3

import logging
import unittest
from copy import copy

from astropy import units as u

import simtools.io_handler as io
from simtools.corsika.corsika_config import (
    CorsikaConfig,
    InvalidCorsikaInput,
    MissingRequiredInputInCorsikaConfigData
)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class TestCorsikaConfig(unittest.TestCase):

    def setUp(self):
        logger.info('setUp')
        self.corsikaConfigData = {
            'nshow': 100,
            'nrun': 10,
            'zenith': 20 * u.deg,
            'viewcone': 5 * u.deg,
            'erange': [10 * u.GeV, 10 * u.TeV],
            'eslope': -2,
            'phi': 0 * u.deg,
            'cscat': [10, 1500 * u.m, 0],
            'primary': 'proton'
        }
        self.corsikaConfig = CorsikaConfig(
            site='Paranal',
            layoutName='4LST',
            label='test-corsika-config',
            corsikaConfigData=self.corsikaConfigData
        )

    def test_repr(self):
        logger.info('test_repr')
        text = repr(self.corsikaConfig)
        self.assertTrue('site' in text)

    def test_user_parameters(self):
        logger.info('test_user_parameters')
        self.assertEqual(self.corsikaConfig.getUserParameter('nshow'), 100)
        self.assertEqual(self.corsikaConfig.getUserParameter('thetap'), [20, 20])
        self.assertEqual(self.corsikaConfig.getUserParameter('erange'), [10., 10000.])

    def test_export_input_file(self):
        logger.info('test_export_input_file')
        self.corsikaConfig.exportInputFile()
        inputFile = self.corsikaConfig.getInputFile()
        self.assertTrue(inputFile.exists())

    def test_wrong_par_in_config_data(self):
        logger.info('test_wrong_primary_name')
        newConfigData = copy(self.corsikaConfigData)
        newConfigData['wrong_par'] = 20 * u.m
        with self.assertRaises(InvalidCorsikaInput):
            corsikaConfig = CorsikaConfig(
                site='LaPalma',
                layoutName='1LST',
                label='test-corsika-config',
                corsikaConfigData=newConfigData
            )
            corsikaConfig.printUserParameters()

    def test_units_of_config_data(self):
        logger.info('test_units_of_config_data')
        newConfigData = copy(self.corsikaConfigData)
        newConfigData['zenith'] = 20 * u.m
        with self.assertRaises(InvalidCorsikaInput):
            corsikaConfig = CorsikaConfig(
                site='LaPalma',
                layoutName='1LST',
                label='test-corsika-config',
                corsikaConfigData=newConfigData
            )
            corsikaConfig.printUserParameters()

    def test_len_of_config_data(self):
        logger.info('test_len_of_config_data')
        newConfigData = copy(self.corsikaConfigData)
        newConfigData['erange'] = [20 * u.TeV]
        with self.assertRaises(InvalidCorsikaInput):
            corsikaConfig = CorsikaConfig(
                site='LaPalma',
                layoutName='1LST',
                label='test-corsika-config',
                corsikaConfigData=newConfigData
            )
            corsikaConfig.printUserParameters()

    def test_wrong_primary_name(self):
        logger.info('test_wrong_primary_name')
        newConfigData = copy(self.corsikaConfigData)
        newConfigData['primary'] = 'rock'
        with self.assertRaises(InvalidCorsikaInput):
            corsikaConfig = CorsikaConfig(
                site='LaPalma',
                layoutName='1LST',
                label='test-corsika-config',
                corsikaConfigData=newConfigData
            )
            corsikaConfig.printUserParameters()

    def test_missing_input(self):
        logger.info('test_missing_input')
        newConfigData = copy(self.corsikaConfigData)
        newConfigData.pop('primary')
        with self.assertRaises(MissingRequiredInputInCorsikaConfigData):
            corsikaConfig = CorsikaConfig(
                site='LaPalma',
                layoutName='1LST',
                label='test-corsika-config',
                corsikaConfigData=newConfigData
            )
            corsikaConfig.printUserParameters()

    def test_set_user_parameters(self):
        logger.info('test_set_user_parameters')
        newConfigData = copy(self.corsikaConfigData)
        newConfigData['zenith'] = 0 * u.deg
        newCorsikaConfig = copy(self.corsikaConfig)
        newCorsikaConfig.setUserParameters(newConfigData)
        self.assertEqual(newCorsikaConfig.getUserParameter('thetap'), [0, 0])

    def test_config_data_from_yaml_file(self):
        logger.info('test_config_data_from_yaml_file')
        corsikaConfigFile = io.getTestDataFile('corsikaConfigTest.yml')
        cc = CorsikaConfig(
            site='Paranal',
            layoutName='4LST',
            label='test-corsika-config',
            corsikaConfigFile=corsikaConfigFile
        )
        cc.printUserParameters()


if __name__ == '__main__':
    unittest.main()

    # tt = TestCorsikaConfig()
    # tt.setUp()
    # tt.test_set_user_parameters()
