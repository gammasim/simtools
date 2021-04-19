#!/usr/bin/python3

import logging
import unittest

from simtools.simtel.simtel_config_writer import SimtelConfigWriter
from simtools.model.array_model import ArrayModel

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class TestSimtelConfigWriter(unittest.TestCase):

    def setUp(self):
        arrayConfigData = {
            'site': 'North',
            'layoutName': '4LST',
            'modelVersion': 'Prod5',
            'default': {
                'LST': '1'
            }
        }
        self.ArrayModel = ArrayModel(
            label='test',
            arrayConfigData=arrayConfigData
        )

    def test_write_array_config_file():
        self.ArrayModel.exportSimtelArrayConfigFile()


if __name__ == '__main__':
    # unittest.main()

    tt = TestSimtelConfigWriter()
    tt.setUp()
    tt.test_write_array_config_file()
