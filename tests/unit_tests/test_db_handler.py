#!/usr/bin/python3

import pytest
import logging
# import subprocess
from pathlib import Path

import simtools.config as cfg
import simtools.io_handler as io
from simtools import db_handler

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def db(set_db):
    db = db_handler.DatabaseHandler()
    return db


def test_reading_db_lst(db):

    logger.info("----Testing reading LST-----")
    pars = db.getModelParameters("north", "lst-1", "Current")
    if cfg.get("useMongoDB"):
        assert pars["parabolic_dish"]["Value"] == 1
        assert pars["camera_pixels"]["Value"] == 1855
    else:
        assert pars["parabolic_dish"] == 1
        assert pars["camera_pixels"] == 1855


def test_reading_db_mst_nc(db):

    logger.info("----Testing reading MST-NectarCam-----")
    pars = db.getModelParameters("north", "mst-NectarCam-D", "Current")
    if cfg.get("useMongoDB"):
        assert pars["camera_pixels"]["Value"] == 1855
    else:
        assert pars["camera_pixels"] == 1855


def test_reading_db_mst_fc(db):

    logger.info("----Testing reading MST-FlashCam-----")
    pars = db.getModelParameters("north", "mst-FlashCam-D", "Current")
    if cfg.get("useMongoDB"):
        assert pars["camera_pixels"]["Value"] == 1764
    else:
        assert pars["camera_pixels"] == 1764


def test_reading_db_sst(db):

    logger.info("----Testing reading SST-----")
    pars = db.getModelParameters("south", "sst-D", "Current")
    if cfg.get("useMongoDB"):
        assert pars["camera_pixels"]["Value"] == 2048
    else:
        assert pars["camera_pixels"] == 2048


def test_copy_telescope_db(db):

    logger.info("----Testing copying a whole telescope-----")
    db.copyTelescope(
        db.DB_CTA_SIMULATION_MODEL,
        "North-LST-1",
        "Current",
        "North-LST-Test",
        "sandbox",
    )
    db.copyDocuments(
        db.DB_CTA_SIMULATION_MODEL,
        "metadata",
        {"Entry": "Simulation-Model-Tags"},
        "sandbox",
    )
    pars = db.readMongoDB(
        "sandbox", "North-LST-Test", "Current", io.getTestOutputDirectory(), False
    )
    assert pars["camera_pixels"]["Value"] == 1855

    logger.info(
        "Testing deleting a query (a whole telescope in this case and metadata)"
    )
    query = {"Telescope": "North-LST-Test"}
    db.deleteQuery("sandbox", "telescopes", query)
    query = {"Entry": "Simulation-Model-Tags"}
    db.deleteQuery("sandbox", "metadata", query)

    # After deleting the copied telescope
    # we always expect to get a ValueError (query returning zero results)
    with pytest.raises(ValueError):
        db.readMongoDB(
            "sandbox", "North-LST-Test", "Current", io.getTestOutputDirectory(), False
        )


def test_adding_parameter_version_db(db):

    logger.info("----Testing adding a new version of a parameter-----")
    db.copyTelescope(
        db.DB_CTA_SIMULATION_MODEL,
        "North-LST-1",
        "Current",
        "North-LST-Test",
        "sandbox",
    )
    db.addParameter(
        "sandbox", "North-LST-Test", "camera_config_version", "test", 42
    )
    pars = db.readMongoDB(
        "sandbox", "North-LST-Test", "test", io.getTestOutputDirectory(), False
    )
    assert pars["camera_config_version"]["Value"] == 42

    # Cleanup
    query = {"Telescope": "North-LST-Test"}
    db.deleteQuery("sandbox", "telescopes", query)


def test_update_parameter_db(db):

    logger.info("----Testing updating a parameter-----")
    db.copyTelescope(
        db.DB_CTA_SIMULATION_MODEL,
        "North-LST-1",
        "Current",
        "North-LST-Test",
        "sandbox",
    )
    db.addParameter(
        "sandbox", "North-LST-Test", "camera_config_version", "test", 42
    )
    db.updateParameter(
        "sandbox", "North-LST-Test", "test", "camera_config_version", 999
    )
    pars = db.readMongoDB(
        "sandbox", "North-LST-Test", "test", io.getTestOutputDirectory(), False
    )
    assert pars["camera_config_version"]["Value"] == 999

    # Cleanup
    query = {"Telescope": "North-LST-Test"}
    db.deleteQuery("sandbox", "telescopes", query)


def test_adding_new_parameter_db(db):

    logger.info("----Testing adding a new parameter-----")
    db.copyTelescope(
        db.DB_CTA_SIMULATION_MODEL,
        "North-LST-1",
        "Current",
        "North-LST-Test",
        "sandbox",
    )
    db.addNewParameter(
        "sandbox", "North-LST-Test", "test", "camera_config_version_test", 999
    )
    pars = db.readMongoDB(
        "sandbox", "North-LST-Test", "test", io.getTestOutputDirectory(), False
    )
    assert pars["camera_config_version_test"]["Value"] == 999

    # Cleanup
    query = {"Telescope": "North-LST-Test"}
    db.deleteQuery("sandbox", "telescopes", query)


def test_update_parameter_field_db(db):

    logger.info("----Testing modifying a field of a parameter-----")
    db.copyTelescope(
        db.DB_CTA_SIMULATION_MODEL,
        "North-LST-1",
        "Current",
        "North-LST-Test",
        "sandbox",
    )
    db.copyDocuments(
        db.DB_CTA_SIMULATION_MODEL,
        "metadata",
        {"Entry": "Simulation-Model-Tags"},
        "sandbox",
    )

    db.updateParameterField(
        dbName="sandbox",
        telescope="North-LST-Test",
        version="Current",
        parameter="camera_pixels",
        field="Applicable",
        newValue=False
    )
    pars = db.readMongoDB(
        "sandbox", "North-LST-Test", "Current", io.getTestOutputDirectory(), False
    )
    assert pars["camera_pixels"]["Applicable"] is False

    # Cleanup
    query = {"Telescope": "North-LST-Test"}
    db.deleteQuery("sandbox", "telescopes", query)
    query = {"Entry": "Simulation-Model-Tags"}
    db.deleteQuery("sandbox", "metadata", query)


def test_reading_db_sites(db):

    logger.info("----Testing reading La Palma parameters-----")
    pars = db.getSiteParameters("North", "Current")
    if cfg.get("useMongoDB"):
        assert pars["altitude"]["Value"] == 2158
    else:
        assert pars["altitude"] == 2158

    logger.info("----Testing reading Paranal parameters-----")
    pars = db.getSiteParameters("South", "Current")
    if cfg.get("useMongoDB"):
        assert pars["altitude"]["Value"] == 2147
    else:
        assert pars["altitude"] == 2147


def test_separating_get_and_write(db):

    logger.info("----Testing getting parameters and exporting model files-----")
    pars = db.getModelParameters("north", "lst-1", "Current")

    fileList = list()
    for parNow in pars.values():
        if parNow["File"]:
            fileList.append(parNow["Value"])
    db.exportModelFiles(pars, io.getTestOutputDirectory())
    logger.debug("Checking files were written to {}".format(io.getTestOutputDirectory()))
    for fileNow in fileList:
        assert (io.getTestOutputDirectory() / fileNow).exists()


def test_insert_files_db(db):

    logger.info("----Testing inserting files to the DB-----")
    logger.info(
        "Creating a temporary file in {}".format(io.getTestOutputDirectory())
    )
    fileName = io.getTestOutputDirectory() / "test_file.dat"
    with open(fileName, "w") as f:
        f.write("# This is a test file")

    file_id = db.insertFileToDB(fileName, "sandbox")
    assert file_id == db._getFileMongoDB("sandbox", "test_file.dat")._id

    logger.info("Dropping the temporary files in the sandbox")
    db.dbClient["sandbox"]["fs.chunks"].drop()
    db.dbClient["sandbox"]["fs.files"].drop()
