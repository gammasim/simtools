#!/usr/bin/python3

import logging
import unittest

from astropy import units as u

import simtools.io_handler as io
from simtools.corsika.corsika_config import CorsikaConfig

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class TestCorsikaConfig(unittest.TestCase):

    def setUp(self):
        logger.info('setUp')
        corsikaConfigData = {
            'nshow': 100,
            'nrun': 10,
            'wrong_par': 200,
            'zenith': 20 * u.deg,
            'viewcone': 5 * u.deg,
            'erange': [0.01 * u.GeV, 10 * u.GeV],
            'eslope': -2,
            'phi': 0 * u.deg,
            'cscat': [10, 1500 * u.m, 0],
            'primary': 'proton'
        }
        self.corsikaConfig = CorsikaConfig(
            site='Paranal',
            layoutName='4LST',
            label='test-corsika-config',
            corsikaConfigData=corsikaConfigData
        )

    def test_repr(self):
        logger.info('test_repr')
        text = repr(self.corsikaConfig)
        self.assertTrue('site' in text)

    def test_print(self):
        print('TESTTESTTEST')

# def test_general():

#     corsikaConfigData = {
#         'nshow': 100,
#         'nrun': 10,
#         'wrong_par': 200,
#         'zenith': 20 * u.deg,
#         'viewcone': 5 * u.deg,
#         'erange': [0.01 * u.GeV, 10 * u.GeV],
#         'eslope': -2,
#         'phi': 0 * u.deg,
#         'cscat': [10, 1500 * u.m, 0],
#         'primary': 'proton'
#     }

#     cc = CorsikaConfig(
#         site='Paranal',
#         layoutName='4LST',
#         label='test-corsika-config',
#         corsikaConfigData=corsikaConfigData
#     )
#     cc.printParameters()
#     cc.exportInputFile()

#     corsikaConfigData2 = {
#         'nshow': 1000,
#         'nrun': 11,
#         'zenith': [0 * u.deg, 60 * u.deg],
#         'viewcone': [0 * u.deg, 10 * u.deg],
#         'erange': [0.01 * u.TeV, 10 * u.TeV],
#         'eslope': -2,
#         'phi': 0 * u.deg,
#         'cscat': [10, 1500 * u.m, 0],
#         'primary': 'proton'
#     }
#     cc2 = CorsikaConfig(
#         site='LaPalma',
#         layoutName='1SST',
#         label='test-corsika-config',
#         corsikaConfigData=corsikaConfigData2
#     )
#     cc2.exportInputFile()

#     corsikaConfigData3 = {
#         'nshow': 1000,
#         'nrun': 11,
#         'zenith': [0 * u.deg, 60 * u.deg],
#         'viewcone': [0 * u.deg, 10 * u.deg],
#         'erange': [0.01 * u.TeV, 10 * u.TeV],
#         'eslope': -2,
#         'phi': 0 * u.deg,
#         'cscat': [10, 1500 * u.m, 0],
#         'primary': 'electron'
#     }

#     cc3 = CorsikaConfig(
#         site='LaPalma',
#         layoutName='1MST',
#         label='test-corsika-config',
#         corsikaConfigData=corsikaConfigData3
#     )
#     # Testing default parameters
#     assert cc3._userParameters['RUNNR'] == [11]
#     assert cc3._userParameters['EVTNR'] == [1]
#     cc3.exportInputFile()


# def test_config_data_from_yaml_file():
#     corsikaConfigFile = io.getTestDataFile('corsikaConfigTest.yml')
#     cc = CorsikaConfig(
#         site='Paranal',
#         layoutName='4LST',
#         label='test-corsika-config',
#         corsikaConfigFile=corsikaConfigFile
#     )
#     cc.printParameters()


# def test_units():
#     corsikaConfigData = {
#         'nshow': 100,
#         'nrun': 10,
#         'zenith': 0.1 * u.rad,
#         'viewcone': 5 * u.deg,
#         'erange': [0.01 * u.TeV, 10 * u.TeV],
#         'eslope': -2,
#         'phi': 0 * u.deg,
#         'cscat': [10, 1500 * u.m, 0],
#         'primary': 'proton'
#     }
#     cc = CorsikaConfig(
#         site='Paranal',
#         layoutName='4LST',
#         label='test-corsika-config',
#         corsikaConfigData=corsikaConfigData
#     )
#     cc.exportInputFile()


# def test_running_corsika_externally():

#     corsikaConfigData = {
#         'nshow': 10,
#         'nrun': 10,
#         'zenith': 20 * u.deg,
#         'viewcone': 0 * u.deg,
#         'erange': [0.05 * u.TeV, 10 * u.TeV],
#         'eslope': -2,
#         'phi': 0 * u.deg,
#         'cscat': [1, 400 * u.m, 0],
#         'primary': 'gamma'
#     }
#     cc = CorsikaConfig(
#         site='South',
#         layoutName='4LST',
#         label='test-corsika-config',
#         corsikaConfigData=corsikaConfigData
#     )
#     cc.exportInputFile()


if __name__ == '__main__':
    unittest.main()

    # tt = TestCorsikaConfig()
    # tt.setUp()
    # tt.test_print()

    # test_general()
    # test_config_data_from_yaml_file()
    # test_units()
    # test_running_corsika_externally()
    # pass
