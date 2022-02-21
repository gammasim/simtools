#!/usr/bin/python3

import logging
import unittest

import simtools.io_handler as io
from simtools.util.general import fileHasText
from simtools.simtel.simtel_config_writer import SimtelConfigWriter
from simtools.model.telescope_model import TelescopeModel
from simtools.layout.layout_array import LayoutArray

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class TestSimtelConfigWriter(unittest.TestCase):
    def setUp(self):
        self.simtelConfigWriter = SimtelConfigWriter(
            site="North",
            modelVersion="Current",
            label="test-simtel-config-writer",
            telescopeModelName="TestTelecope",
        )
        self.telescopeModel = TelescopeModel(
            site="North",
            telescopeModelName="LST-1",
            modelVersion="Current",
            label="test-telescope-model",
        )
        self.layout = LayoutArray.fromLayoutArrayName("South-4LST")

    def test_write_array_config_file(self):
        file = io.getTestOutputFile("simtel-config-writer_array.txt")
        self.simtelConfigWriter.writeArrayConfigFile(
            configFilePath=file,
            layout=self.layout,
            telescopeModel=[self.telescopeModel] * 4,
            siteParameters={},
        )
        self.assertTrue(fileHasText(file, "TELESCOPE == 1"))

    def test_write_tel_config_file(self):
        file = io.getTestOutputFile("simtel-config-writer_telescope.txt")
        self.simtelConfigWriter.writeTelescopeConfigFile(
            configFilePath=file, parameters={"par": {"Value": 1}}
        )
        self.assertTrue(fileHasText(file, "par = 1"))


if __name__ == "__main__":
    unittest.main()
