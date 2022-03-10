#!/usr/bin/python3

import pytest
import logging
import unittest
import subprocess
from pathlib import Path

import simtools.config as cfg
import simtools.io_handler as io
from simtools import db_handler
from simtools.util.tests import has_db_connection, DB_CONNECTION_MSG


class TestDBHandler(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)

        self.db = db_handler.DatabaseHandler()
        self.testDataDirectory = io.getTestOutputDirectory()
        self.DB_CTA_SIMULATION_MODEL = "CTA-Simulation-Model"

    @pytest.mark.skipif(not has_db_connection(), reason=DB_CONNECTION_MSG)
    def test_reading_db_lst(self):

        self.logger.info("----Testing reading LST-----")
        pars = self.db.getModelParameters("north", "lst-1", "Current")
        if cfg.get("useMongoDB"):
            assert pars["parabolic_dish"]["Value"] == 1
            assert pars["camera_pixels"]["Value"] == 1855
        else:
            assert pars["parabolic_dish"] == 1
            assert pars["camera_pixels"] == 1855

        self.db.exportModelFiles(pars, self.testDataDirectory)
        self.logger.info("Listing files written in {}".format(self.testDataDirectory))
        subprocess.call(["ls -lh {}".format(self.testDataDirectory)], shell=True)

    @pytest.mark.skipif(not has_db_connection(), reason=DB_CONNECTION_MSG)
    def test_reading_db_mst_nc(self):

        self.logger.info("----Testing reading MST-NectarCam-----")
        pars = self.db.getModelParameters("north", "mst-NectarCam-D", "Current")
        if cfg.get("useMongoDB"):
            assert pars["camera_pixels"]["Value"] == 1855
        else:
            assert pars["camera_pixels"] == 1855

        self.logger.info("Listing files written in {}".format(self.testDataDirectory))
        subprocess.call(["ls -lh {}".format(self.testDataDirectory)], shell=True)

        self.logger.info(
            "Removing the files written in {}".format(self.testDataDirectory)
        )
        subprocess.call(["rm -f {}/*".format(self.testDataDirectory)], shell=True)

    @pytest.mark.skipif(not has_db_connection(), reason=DB_CONNECTION_MSG)
    def test_reading_db_mst_fc(self):

        self.logger.info("----Testing reading MST-FlashCam-----")
        pars = self.db.getModelParameters("north", "mst-FlashCam-D", "Current")
        if cfg.get("useMongoDB"):
            assert pars["camera_pixels"]["Value"] == 1764
        else:
            assert pars["camera_pixels"] == 1764

        self.logger.info("Listing files written in {}".format(self.testDataDirectory))
        subprocess.call(["ls -lh {}".format(self.testDataDirectory)], shell=True)

        self.logger.info(
            "Removing the files written in {}".format(self.testDataDirectory)
        )
        subprocess.call(["rm -f {}/*".format(self.testDataDirectory)], shell=True)

    @pytest.mark.skipif(not has_db_connection(), reason=DB_CONNECTION_MSG)
    def test_reading_db_sst(self):

        self.logger.info("----Testing reading SST-----")
        pars = self.db.getModelParameters("south", "sst-D", "Current")
        if cfg.get("useMongoDB"):
            assert pars["camera_pixels"]["Value"] == 2048
        else:
            assert pars["camera_pixels"] == 2048

        self.logger.info("Listing files written in {}".format(self.testDataDirectory))
        subprocess.call(["ls -lh {}".format(self.testDataDirectory)], shell=True)

        self.logger.info(
            "Removing the files written in {}".format(self.testDataDirectory)
        )
        subprocess.call(["rm -f {}/*".format(self.testDataDirectory)], shell=True)

    @pytest.mark.skipif(not has_db_connection(), reason=DB_CONNECTION_MSG)
    def test_copy_telescope_db(self):

        self.logger.info("----Testing copying a whole telescope-----")
        self.db.copyTelescope(
            self.DB_CTA_SIMULATION_MODEL,
            "North-LST-1",
            "Current",
            "North-LST-Test",
            "sandbox",
        )
        self.db.copyDocuments(
            self.DB_CTA_SIMULATION_MODEL,
            "metadata",
            {"Entry": "Simulation-Model-Tags"},
            "sandbox",
        )
        pars = self.db.readMongoDB(
            "sandbox", "North-LST-Test", "Current", self.testDataDirectory, False
        )
        assert pars["camera_pixels"]["Value"] == 1855

        self.logger.info(
            "Testing deleting a query (a whole telescope in this case and metadata)"
        )
        query = {"Telescope": "North-LST-Test"}
        self.db.deleteQuery("sandbox", "telescopes", query)
        query = {"Entry": "Simulation-Model-Tags"}
        self.db.deleteQuery("sandbox", "metadata", query)

    @pytest.mark.skipif(not has_db_connection(), reason=DB_CONNECTION_MSG)
    def test_adding_parameter_version_db(self):

        self.logger.info("----Testing adding a new version of a parameter-----")
        self.db.copyTelescope(
            self.DB_CTA_SIMULATION_MODEL,
            "North-LST-1",
            "Current",
            "North-LST-Test",
            "sandbox",
        )
        self.db.addParameter(
            "sandbox", "North-LST-Test", "camera_config_version", "test", 42
        )
        pars = self.db.readMongoDB(
            "sandbox", "North-LST-Test", "test", self.testDataDirectory, False
        )
        assert pars["camera_config_version"]["Value"] == 42

        query = {"Telescope": "North-LST-Test"}
        self.db.deleteQuery("sandbox", "telescopes", query)

    @pytest.mark.skipif(not has_db_connection(), reason=DB_CONNECTION_MSG)
    def test_update_parameter_db(self):

        self.logger.info("----Testing updating a parameter-----")
        self.db.copyTelescope(
            self.DB_CTA_SIMULATION_MODEL,
            "North-LST-1",
            "Current",
            "North-LST-Test",
            "sandbox",
        )
        self.db.addParameter(
            "sandbox", "North-LST-Test", "camera_config_version", "test", 42
        )
        self.db.updateParameter(
            "sandbox", "North-LST-Test", "test", "camera_config_version", 999
        )
        pars = self.db.readMongoDB(
            "sandbox", "North-LST-Test", "test", self.testDataDirectory, False
        )
        assert pars["camera_config_version"]["Value"] == 999

        query = {"Telescope": "North-LST-Test"}
        self.db.deleteQuery("sandbox", "telescopes", query)

    @pytest.mark.skipif(not has_db_connection(), reason=DB_CONNECTION_MSG)
    def test_adding_new_parameter_db(self):

        self.logger.info("----Testing adding a new parameter-----")
        self.db.copyTelescope(
            self.DB_CTA_SIMULATION_MODEL,
            "North-LST-1",
            "Current",
            "North-LST-Test",
            "sandbox",
        )
        self.db.addNewParameter(
            "sandbox", "North-LST-Test", "test", "camera_config_version_test", 999
        )
        pars = self.db.readMongoDB(
            "sandbox", "North-LST-Test", "test", self.testDataDirectory, False
        )
        assert pars["camera_config_version_test"]["Value"] == 999

        query = {"Telescope": "North-LST-Test"}
        self.db.deleteQuery("sandbox", "telescopes", query)

    @pytest.mark.skipif(not has_db_connection(), reason=DB_CONNECTION_MSG)
    def test_reading_db_sites(self):

        self.logger.info("----Testing reading La Palma parameters-----")
        pars = self.db.getSiteParameters("North", "Current")
        if cfg.get("useMongoDB"):
            assert pars["altitude"]["Value"] == 2158
        else:
            assert pars["altitude"] == 2158

        self.logger.info("Listing files written in {}".format(self.testDataDirectory))
        subprocess.call(["ls -lh {}".format(self.testDataDirectory)], shell=True)

        self.logger.info(
            "Removing the files written in {}".format(self.testDataDirectory)
        )
        subprocess.call(["rm -f {}/*".format(self.testDataDirectory)], shell=True)

        self.logger.info("----Testing reading Paranal parameters-----")
        pars = self.db.getSiteParameters("South", "Current")
        if cfg.get("useMongoDB"):
            assert pars["altitude"]["Value"] == 2147
        else:
            assert pars["altitude"] == 2147

        self.logger.info("Listing files written in {}".format(self.testDataDirectory))
        subprocess.call(["ls -lh {}".format(self.testDataDirectory)], shell=True)

        self.logger.info(
            "Removing the files written in {}".format(self.testDataDirectory)
        )
        subprocess.call(["rm -f {}/*".format(self.testDataDirectory)], shell=True)

        return

    @pytest.mark.skipif(not has_db_connection(), reason=DB_CONNECTION_MSG)
    def test_separating_get_and_write(self):
        pars = self.db.getModelParameters("north", "lst-1", "prod4")

        self.logger.info("Listing files written in {}".format(self.testDataDirectory))
        subprocess.call(["ls -lh {}".format(self.testDataDirectory)], shell=True)

        self.db.exportModelFiles(pars, self.testDataDirectory)

        self.logger.info("Listing files written in {}".format(self.testDataDirectory))
        subprocess.call(["ls -lh {}".format(self.testDataDirectory)], shell=True)

    @pytest.mark.skipif(not has_db_connection(), reason=DB_CONNECTION_MSG)
    def test_insert_files_db(self):

        self.logger.info("----Testing inserting files to the DB-----")
        self.logger.info(
            "Creating a temporary file in {}".format(self.testDataDirectory)
        )
        fileName = Path(self.testDataDirectory).joinpath("test_file.dat")
        with open(fileName, "w") as f:
            f.write("# This is a test file")

        file_id = self.db.insertFileToDB(fileName, "sandbox")
        assert file_id == self.db._getFileMongoDB("sandbox", "test_file.dat")._id

        subprocess.call(["rm -f {}".format(fileName)], shell=True)

        self.logger.info("Dropping the temporary files in the sandbox")
        self.db.dbClient["sandbox"]["fs.chunks"].drop()
        self.db.dbClient["sandbox"]["fs.files"].drop()


if __name__ == "__main__":

    unittest.main()
