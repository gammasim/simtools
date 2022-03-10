#!/usr/bin/python3

import pytest
import logging
import unittest

import simtools.config as cfg
from simtools.config import ParameterNotFoundInConfigFile
from simtools.util.tests import has_config_file, CONFIG_FILE_MSG


logging.getLogger().setLevel(logging.DEBUG)


class TestConfig(unittest.TestCase):

    def test_get_parameters(self):
        parameters = (
            "modelFilesLocations",
            "dataLocation",
            "outputLocation",
            "simtelPath",
            "useMongoDB",
            "mongoDBConfigFile",
        )
        for par in parameters:
            with self.subTest(msg="Testing get {}".format(par)):
                cfg.get(par)

    def test_get_non_existing_parameter(self):
        with self.assertRaises(ParameterNotFoundInConfigFile):
            cfg.get("NonExistingEntry")

    def test_input_options(self):
        cfg.setConfigFileName("config.yml")
        print("cfg.CONFIG_FILE_NAME: {}".format(cfg.CONFIG_FILE_NAME))
        print("modelFilesLocations: {}".format(cfg.get("modelFilesLocations")))

    @pytest.mark.skipif(not has_config_file(), reason=CONFIG_FILE_MSG)
    def test_find_file(self):
        files = ("mirror_MST_D80.dat", "parValues-LST.yml")
        for file in files:
            with self.subTest(msg="Testing find file {}".format(file)):
                cfg.findFile(file)


if __name__ == "__main__":
    unittest.main()
