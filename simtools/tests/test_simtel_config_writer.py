#!/usr/bin/python3

import logging
import unittest

import simtools.io_handler as io
from simtools.simtel.simtel_config_writer import SimtelConfigWriter
from simtools.model.telescope_model import TelescopeModel
from simtools.layout.layout_array import LayoutArray

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def file_has_text(file, text):
    with open(file, 'r') as ff:
        for ll in ff:
            if text in ll:
                return True
    return False

class TestSimtelConfigWriter(unittest.TestCase):

    def setUp(self):
        self.simtelConfigWriter = SimtelConfigWriter(
            site='North',
            modelVersion='Current',
            label='test-simtel-config-writer',
            telescopeName='TestTelecope'
        )
        self.telescopeModel = TelescopeModel(
            telescopeName='North-LST-1',
            modelVersion='Current',
            label='test-telescope-model'
        )
        self.layout = LayoutArray.fromLayoutArrayName('South-4LST')

    def test_write_array_config_file(self):
        file = io.getTestOutputFile('simtel-config-writer_array.txt')
        self.simtelConfigWriter.writeArrayConfigFile(
            configFilePath=file,
            layout=self.layout,
            telescopeModel=[self.telescopeModel] * 4,
            siteParameters={}
        )
        self.assertTrue(file_has_text(file, 'TELESCOPE == 1'))

    def test_write_tel_config_file(self):
        file = io.getTestOutputFile('simtel-config-writer_telescope.txt')
        self.simtelConfigWriter.writeTelescopeConfigFile(
            configFilePath=file,
            parameters={'par': {'Value': 1}}
        )
        self.assertTrue(file_has_text(file, 'par = 1'))


if __name__ == '__main__':
    unittest.main()
