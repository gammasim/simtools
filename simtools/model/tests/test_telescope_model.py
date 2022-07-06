#!/usr/bin/python3

import pytest
import logging
import unittest

import simtools.io_handler as io
from simtools.model.telescope_model import TelescopeModel, InvalidParameter
from simtools.util.tests import has_db_connection, DB_CONNECTION_MSG


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class TestTelescopeModel(unittest.TestCase):
    def setUp(self):
        cfgFile = io.getTestDataFile("CTA-North-LST-1-Current_test-telescope-model.cfg")
        self.label = "test-telescope-model"
        self.telModel = TelescopeModel.fromConfigFile(
            site="North",
            telescopeModelName="LST-1",
            label=self.label,
            configFileName=cfgFile,
        )

    def test_handling_parameters(self):
        logger.info(
            "Old mirror_reflection_random_angle:{}".format(
                self.telModel.getParameterValue("mirror_reflection_random_angle")
            )
        )
        logger.info("Changing mirror_reflection_random_angle")
        new_mrra = "0.0080 0 0"
        self.telModel.changeParameter("mirror_reflection_random_angle", new_mrra)
        self.assertEqual(
            self.telModel.getParameterValue("mirror_reflection_random_angle"), new_mrra
        )

        logging.info("Adding new_parameter")
        new_par = "23"
        self.telModel.addParameter("new_parameter", new_par)
        self.assertEqual(self.telModel.getParameterValue("new_parameter"), new_par)

        with self.assertRaises(InvalidParameter):
            self.telModel.getParameter("bla_bla")

    def test_flen_type(self):
        flenInfo = self.telModel.getParameter("focal_length")
        logger.info(
            "Focal Length = {}, type = {}".format(flenInfo["Value"], flenInfo["Type"])
        )
        self.assertIsInstance(flenInfo["Value"], float)

    @pytest.mark.skipif(not has_db_connection(), reason=DB_CONNECTION_MSG)
    def test_cfg_file(self):
        # Exporting
        self.telModel.exportConfigFile()

        logger.info("Config file: {}".format(self.telModel.getConfigFile()))

        # Importing
        cfgFile = self.telModel.getConfigFile()
        tel = TelescopeModel.fromConfigFile(
            site="south",
            telescopeModelName="sst-d",
            label="test-sst",
            configFileName=cfgFile,
        )
        tel.exportConfigFile()

    @pytest.mark.skipif(not has_db_connection(), reason=DB_CONNECTION_MSG)
    def test_updating_export_model_files(self):
        """
        It was found in derive_mirror_rnda_angle that the DB was being
        accessed each time the model was changed, because the model
        files were being re-exported. A flag called _isExportedModelFilesUpToDate
        was added to prevent this behavior. This test is meant to assure
        it is working properly.
        """

        # We need a brand new telescopeModel to avoid interference
        tel = TelescopeModel(
            site="North",
            telescopeModelName="LST-1",
            modelVersion="Current",
            label="test-telescope-model-2",
        )

        logger.debug(
            "tel._isExportedModelFiles should be False because exportConfigFile"
            " was not called yet."
        )
        self.assertFalse(tel._isExportedModelFilesUpToDate)

        # Exporting config file
        tel.exportConfigFile()
        logger.debug(
            "tel._isExportedModelFiles should be True because exportConfigFile"
            " was called."
        )
        self.assertTrue(tel._isExportedModelFilesUpToDate)

        # Changing a non-file parameter
        logger.info(
            "Changing a parameter that IS NOT a file - mirror_reflection_random_angle"
        )
        tel.changeParameter("mirror_reflection_random_angle", "0.0080 0 0")
        logger.debug(
            "tel._isExportedModelFiles should still be True because the changed "
            "parameter was not a file"
        )
        self.assertTrue(tel._isExportedModelFilesUpToDate)

        # Testing the DB connection
        logger.info("DB should NOT be read next.")
        tel.exportConfigFile()

        # Changing a parameter that is a file
        logger.debug(
            "Changing a parameter that IS a file - camera_config_file"
        )
        tel.changeParameter(
            "camera_config_file",
            tel.getParameterValue("camera_config_file")
        )
        logger.debug(
            "tel._isExportedModelFiles should be False because a parameter that "
            "is a file was changed."
        )
        self.assertFalse(tel._isExportedModelFilesUpToDate)

        # Testing the DB connection
        logger.info("DB should be read next.")
        tel.exportConfigFile()


if __name__ == "__main__":
    unittest.main()
