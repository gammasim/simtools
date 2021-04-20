#!/usr/bin/python3

import logging
import unittest

from simtools.simtel.simtel_config_writer import SimtelConfigWriter
from simtools.model.array_model import ArrayModel
from simtools.model.telescope_model import TelescopeModel

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class TestSimtelConfigWriter(unittest.TestCase):

    def test_write_array_config_file(self):
        arrayConfigData = {
            'site': 'North',
            'layoutName': '1LST',
            'modelVersion': 'Prod5',
            'default': {
                'LST': '1'
            }
        }
        arrayModel = ArrayModel(
            label='test',
            arrayConfigData=arrayConfigData
        )
        arrayModel.exportSimtelArrayConfigFile()

    def test_write_tel_config_file(self):
        telescopeModel = TelescopeModel(
            telescopeName='North-LST-1',
            modelVersion='Current',
            label='test-lst'
        )
        telescopeModel.exportConfigFile()


if __name__ == '__main__':
    unittest.main()

    # tt = TestSimtelConfigWriter()
    # tt.setUp()
    # tt.test_write_array_config_file()
